#!/usr/bin/env python
import mlx.core as mx

def f32(x):
    return mx.array(x, dtype=mx.float32)

def u32(x):
    return mx.array(x, dtype=mx.uint32)

def i32(x):
    return mx.array(x, dtype=mx.int32)

__all__ = ["f32", "u32", "i32"]

