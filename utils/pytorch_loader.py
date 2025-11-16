#!/usr/bin/env python
"""
PyTorch checkpoint loader for MLX with on-disk conversion to safetensors.
Converts .pt and .bin files to .safetensors format on first load to avoid
loading large pickle files into memory.
"""
import pickle
import zipfile
from pathlib import Path
from typing import Dict
import os

import mlx.core as mx


def load_pytorch_bin(file_path: Path) -> Dict[str, mx.array]:
    """
    Load PyTorch .pt/.bin checkpoint into MLX arrays.

    On first load, converts the checkpoint to .safetensors format and caches it.
    Subsequent loads use the cached .safetensors file directly via mx.load.

    Args:
        file_path: Path to .pt or .bin file

    Returns:
        Dictionary mapping parameter names to MLX arrays
    """
    file_path = Path(file_path)

    # Check for cached safetensors version
    cache_path = file_path.with_suffix('.safetensors')

    if cache_path.exists():
        # Use cached version - much faster and less memory
        print(f"Loading from cached safetensors: {cache_path}")
        return mx.load(str(cache_path))

    # Need to convert from PyTorch pickle format
    print(f"Converting PyTorch checkpoint to safetensors format...")
    print(f"Source: {file_path}")
    print(f"Cache: {cache_path}")

    # Load from pickle (unfortunately unavoidable for first conversion)
    state_dict = _load_pytorch_pickle(file_path)

    # Save to safetensors for future use
    print(f"Saving safetensors cache...")
    mx.save_safetensors(str(cache_path), state_dict)
    print(f"✓ Cached safetensors saved to {cache_path}")

    return state_dict


def _load_pytorch_pickle(file_path: Path) -> Dict[str, mx.array]:
    """
    Load PyTorch pickle checkpoint (internal use only for conversion).

    This is only called once per checkpoint to create the safetensors cache.
    """
    import gc

    with zipfile.ZipFile(file_path, 'r') as zf:
        # Find the pickle file - can be at various paths
        pickle_paths = ['pytorch_model/data.pkl', 'archive/data.pkl', 'data.pkl']
        pickle_name = None
        for path in pickle_paths:
            if path in zf.namelist():
                pickle_name = path
                break

        if pickle_name is None:
            raise ValueError(f"Could not find data.pkl in checkpoint. Available files: {zf.namelist()[:10]}")

        with zf.open(pickle_name) as f:
            # Need custom unpickler to handle persistent IDs
            unpickler = _TorchUnpickler(f, zf)
            state_dict = unpickler.load()

        # Clear storage cache to free memory
        unpickler.storage_cache.clear()

    gc.collect()
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
        # Calculate total elements needed
        total_elements = 1
        for s in size:
            total_elements *= s

        # Get storage size (flattened)
        storage_size = storage.size


        # Always slice to get exactly the elements we need
        end_idx = storage_offset + total_elements
        if end_idx > storage_size:
            raise ValueError(f"Storage slice out of bounds: need [{storage_offset}:{end_idx}] but storage has {storage_size} elements. Tensor shape: {size}")

        # Slice storage to get exactly what we need
        if storage_offset != 0 or end_idx != storage_size:
            storage = storage[storage_offset:end_idx]

        # Verify we have the right number of elements
        if storage.size != total_elements:
            raise ValueError(f"Storage size mismatch: have {storage.size} elements but need {total_elements} for shape {size}")

        # Reshape to final size
        if len(size) > 0 and total_elements > 0:
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

            # Load raw bytes from zip (can be at various paths)
            storage_paths = [
                f'pytorch_model/data/{key}',
                f'archive/data/{key}',
                f'data/{key}'
            ]
            storage_path = None
            for path in storage_paths:
                if path in self.zip_file.namelist():
                    storage_path = path
                    break

            if storage_path is None:
                raise ValueError(f"Could not find storage for key {key}")

            with self.zip_file.open(storage_path) as f:
                raw_bytes = f.read()

            # Convert raw bytes directly to MLX array
            # PyTorch stores tensors as raw binary data in native endianness
            mlx_array = _bytes_to_mlx_array(raw_bytes, dtype)


            # Clear raw bytes to free memory
            del raw_bytes

            # Cache the MLX array
            self.storage_cache[key] = mlx_array
            return mlx_array

        elif typename == 'module':
            # Module reference - return the module object
            return pid[1]

        else:
            raise pickle.UnpicklingError(f"Unknown typename: {typename}")


def _bytes_to_mlx_array(raw_bytes: bytes, dtype) -> mx.array:
    """
    Convert raw bytes from PyTorch storage to MLX array.
    Pure Python implementation using struct - NO numpy dependency.

    Args:
        raw_bytes: Raw binary data from PyTorch storage
        dtype: Target MLX dtype (or 'bfloat16' string for BFloat16)

    Returns:
        Flat MLX array with the specified dtype
    """
    import struct
    import array

    # Handle BFloat16 specially
    if dtype == 'bfloat16':
        # BFloat16 is 2 bytes per element
        # Convert to float32 by shifting bits left by 16
        num_elements = len(raw_bytes) // 2

        # Unpack as uint16
        uint16_values = struct.unpack(f'<{num_elements}H', raw_bytes)

        # Convert BFloat16 bits to Float32 by shifting left 16 bits
        float32_values = []
        for bf16_bits in uint16_values:
            # Shift BFloat16 bits to become the upper 16 bits of Float32
            f32_bits = bf16_bits << 16
            # Reinterpret as float32
            f32_bytes = struct.pack('I', f32_bits)
            f32_value = struct.unpack('f', f32_bytes)[0]
            float32_values.append(f32_value)

        return mx.array(float32_values, dtype=mx.float32)

    # Dtype info: (struct_format, element_size, mlx_dtype)
    dtype_info = {
        mx.float32: ('f', 4),
        mx.float16: ('e', 2),  # 'e' for half-precision float
        mx.float64: ('d', 8),
        mx.int8: ('b', 1),
        mx.int16: ('h', 2),
        mx.int32: ('i', 4),
        mx.int64: ('q', 8),
        mx.uint8: ('B', 1),
        mx.bool_: ('?', 1),
    }

    if dtype not in dtype_info:
        raise ValueError(f"Unsupported dtype: {dtype}")

    fmt_char, elem_size = dtype_info[dtype]
    num_elements = len(raw_bytes) // elem_size

    # Use memoryview for zero-copy access, then unpack
    # For large arrays, use array.array which is more efficient than struct
    if dtype == mx.float32:
        arr = array.array('f')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.float64:
        arr = array.array('d')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.int32:
        arr = array.array('i')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.int64:
        arr = array.array('q')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.int16:
        arr = array.array('h')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.int8:
        arr = array.array('b')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.uint8:
        arr = array.array('B')
        arr.frombytes(raw_bytes)
        return mx.array(list(arr), dtype=dtype)
    elif dtype == mx.float16:
        # float16 needs special handling - unpack with struct
        values = struct.unpack(f'<{num_elements}e', raw_bytes)
        return mx.array(values, dtype=dtype)
    elif dtype == mx.bool_:
        # bool needs special handling
        values = struct.unpack(f'{num_elements}?', raw_bytes)
        return mx.array(values, dtype=dtype)
    else:
        # Fallback to struct.unpack
        format_str = f'<{num_elements}{fmt_char}'
        values = struct.unpack(format_str, raw_bytes)
        return mx.array(values, dtype=dtype)


def _get_mlx_dtype(storage_type):
    """Get MLX dtype from PyTorch storage type."""
    # storage_type is like FloatStorage, HalfStorage, BFloat16Storage, etc.
    type_str = str(storage_type)
    if 'BFloat16' in type_str:
        # Return a special marker for BFloat16
        return 'bfloat16'
    elif 'Float' in type_str and 'BFloat' not in type_str:
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


