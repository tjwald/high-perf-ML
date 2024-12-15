﻿using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ML.Infra;

public static class TensorExtensions
{
    public static Span<T> GetRowSpan<T>(this TensorSpan<T> tensor, int i)
    {
        if (tensor.Lengths.Length != 2) throw new InvalidOperationException("tensor must have 2 dimensions");
        
        TensorSpan<T> tensorRow = tensor.Slice([i..(i+1), 0..]);
        return MemoryMarshal.CreateSpan(ref tensorRow.GetPinnableReference(), (int)tensor.Lengths[1]);
    }

    public static Memory<T> AsMemory<T>(this Tensor<T> tensor)
    {
        return ExternalClassAccessor<T>.GetValues(tensor).AsMemory();
    }
}

internal static class ExternalClassAccessor<T>
{
    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_values")]
    public static extern ref T[] GetValues(Tensor<T> instance);
}
