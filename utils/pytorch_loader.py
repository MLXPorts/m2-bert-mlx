#!/usr/bin/env python
"""
Simple PyTorch checkpoint loader for MLX.
Handles .pt and .bin files (zip archives containing pickled tensors).
"""
import pickle
import zipfile
from pathlib import Path
from typing import Dict

import mlx.core as mx


def load_pytorch_bin(file_path: Path) -> Dict[str, mx.array]:
    """
    Load PyTorch .pt/.bin checkpoint into MLX arrays.

    PyTorch saves checkpoints as zip files containing:
    - data.pkl: pickled state_dict structure
    - data/*.storage: raw tensor data

    Args:
        file_path: Path to .pt or .bin file

    Returns:
        Dictionary mapping parameter names to MLX arrays
    """
    file_path = Path(file_path)

    with zipfile.ZipFile(file_path, 'r') as zf:
        # Load the pickle file (can be at archive/data.pkl or data.pkl)
        pickle_name = 'archive/data.pkl' if 'archive/data.pkl' in zf.namelist() else 'data.pkl'

        with zf.open(pickle_name) as f:
            # Need custom unpickler to handle persistent IDs
            unpickler = _TorchUnpickler(f, zf)
            state_dict = unpickler.load()

    return state_dict


class _TorchUnpickler(pickle.Unpickler):
    """Custom unpickler that handles PyTorch tensor storage references."""

    def __init__(self, file, zip_file):
        super().__init__(file, encoding='utf-8')
        self.zip_file = zip_file
        self.storage_cache = {}

    def find_class(self, module, name):
        """Override find_class to handle torch-specific functions."""
        # Handle tensor rebuild functions
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self._rebuild_tensor_v2
        elif module == 'torch._utils' and name == '_rebuild_tensor':
            return self._rebuild_tensor
        elif module == 'torch._utils' and name == '_rebuild_parameter':
            return self._rebuild_parameter
        else:
            # Default behavior for other classes
            return super().find_class(module, name)

    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        """Rebuild a tensor from storage (MLX version)."""
        # storage is already an MLX array from persistent_load
        # Just reshape and slice it
        total_elements = 1
        for s in size:
            total_elements *= s

        # Reshape the flat storage
        if storage_offset > 0 or total_elements < storage.size:
            # Take a slice of the storage
            storage = storage[storage_offset:storage_offset + total_elements]

        # Reshape to final size
        if total_elements > 0:
            tensor = storage.reshape(size)
        else:
            tensor = storage

        return tensor

    def _rebuild_tensor(self, storage, storage_offset, size, stride):
        """Rebuild a tensor (simpler version)."""
        return self._rebuild_tensor_v2(storage, storage_offset, size, stride, False, None)

    def _rebuild_parameter(self, data, requires_grad, backward_hooks):
        """Rebuild a parameter (just return the data for MLX)."""
        return data

    def persistent_load(self, pid):
        """
        Handle PyTorch persistent storage references.

        PyTorch uses persistent IDs like:
        ('storage', <storage_type>, '<key>', '<location>', <size>)
        """
        if not isinstance(pid, tuple):
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")

        typename = pid[0]
        if isinstance(typename, bytes):
            typename = typename.decode('ascii')

        if typename == 'storage':
            storage_type, key, location, size = pid[1:5]

            # Check cache
            if key in self.storage_cache:
                return self.storage_cache[key]

            # Get dtype from storage type
            dtype = _get_mlx_dtype(storage_type)

            # Load raw bytes from zip (can be archive/data/{key} or data/{key})
            storage_path = f'archive/data/{key}' if f'archive/data/{key}' in self.zip_file.namelist() else f'data/{key}'
            with self.zip_file.open(storage_path) as f:
                raw_bytes = f.read()

            # Convert bytes to MLX array via memoryview
            import numpy as np
            np_dtype = _mlx_to_numpy_dtype(dtype)
            np_array = np.frombuffer(raw_bytes, dtype=np_dtype)
            mlx_array = mx.array(np_array)

            # Cache it
            self.storage_cache[key] = mlx_array
            return mlx_array

        elif typename == 'module':
            # Module reference - return the module object
            return pid[1]

        else:
            raise pickle.UnpicklingError(f"Unknown typename: {typename}")


def _get_mlx_dtype(storage_type):
    """Get MLX dtype from PyTorch storage type."""
    # storage_type is like FloatStorage, HalfStorage, etc.
    type_str = str(storage_type)
    if 'Float' in type_str:
        return mx.float32
    elif 'Half' in type_str:
        return mx.float16
    elif 'Double' in type_str:
        return mx.float64
    elif 'Long' in type_str:
        return mx.int64
    elif 'Int' in type_str:
        return mx.int32
    elif 'Short' in type_str:
        return mx.int16
    elif 'Char' in type_str or 'Byte' in type_str:
        return mx.int8
    elif 'Bool' in type_str:
        return mx.bool_
    else:
        return mx.float32  # Default


def _mlx_to_numpy_dtype(mlx_dtype):
    """Convert MLX dtype to numpy dtype for frombuffer."""
    import numpy as np
    mapping = {
        mx.float32: np.float32,
        mx.float16: np.float16,
        mx.float64: np.float64,
        mx.int8: np.int8,
        mx.int16: np.int16,
        mx.int32: np.int32,
        mx.int64: np.int64,
        mx.uint8: np.uint8,
        mx.bool_: np.bool_,
    }
    return mapping.get(mlx_dtype, np.float32)
