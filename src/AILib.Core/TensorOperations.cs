using System.Numerics;

namespace AILib;

/// <summary>
/// Extension methods for basic tensor operations.
/// </summary>
public static class TensorOperations
{
    /// <summary>
    /// Reshapes the tensor to a new shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="newShape">The new shape.</param>
    /// <returns>A reshaped tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when new shape is incompatible.</exception>
    public static Tensor<T> Reshape<T>(this Tensor<T> tensor, ReadOnlySpan<nint> newShape)
        where T : unmanaged, INumber<T>
    {
        // Verify total element count matches
        nint newTotal = 1;
        foreach (var dim in newShape)
        {
            newTotal *= dim;
        }

        if (newTotal != tensor.ElementCount)
        {
            throw new ArgumentException(
                $"Cannot reshape tensor of size {tensor.ElementCount} to shape with {newTotal} elements");
        }

        var reshaped = System.Numerics.Tensors.Tensor.Reshape(tensor.Data, newShape);
        return new Tensor<T>(reshaped, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new object() : null // Placeholder for Phase 2
        };
    }

    /// <summary>
    /// Flattens the tensor into a 1D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A flattened 1D tensor.</returns>
    public static Tensor<T> Flatten<T>(this Tensor<T> tensor)
        where T : unmanaged, INumber<T>
    {
        var totalElements = tensor.ElementCount;
        return tensor.Reshape([totalElements]);
    }

    /// <summary>
    /// Transposes a 2D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A transposed tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when tensor is not 2D.</exception>
    public static Tensor<T> Transpose<T>(this Tensor<T> tensor)
        where T : unmanaged, INumber<T>
    {
        if (tensor.Rank != 2)
        {
            throw new InvalidOperationException("Transpose requires a 2D tensor");
        }

        var transposed = System.Numerics.Tensors.Tensor.Transpose(tensor.Data);
        return new Tensor<T>(transposed, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new object() : null // Placeholder for Phase 2
        };
    }

    /// <summary>
    /// Permutes the dimensions of the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="dimensions">The new order of dimensions.</param>
    /// <returns>A tensor with permuted dimensions.</returns>
    public static Tensor<T> Permute<T>(this Tensor<T> tensor, ReadOnlySpan<int> dimensions)
        where T : unmanaged, INumber<T>
    {
        var permuted = System.Numerics.Tensors.Tensor.PermuteDimensions(tensor.Data, dimensions);
        return new Tensor<T>(permuted, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new object() : null // Placeholder for Phase 2
        };
    }

    /// <summary>
    /// Creates a view of the tensor with a different shape (without copying data).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="newShape">The new shape for the view.</param>
    /// <returns>A tensor view with the new shape.</returns>
    public static Tensor<T> View<T>(this Tensor<T> tensor, ReadOnlySpan<nint> newShape)
        where T : unmanaged, INumber<T>
    {
        // View is essentially the same as Reshape in this implementation
        return tensor.Reshape(newShape);
    }

    /// <summary>
    /// Computes the sum of all elements in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A scalar tensor containing the sum.</returns>
    public static Tensor<T> Sum<T>(this Tensor<T> tensor)
        where T : unmanaged, INumber<T>
    {
        var sum = T.Zero;
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            sum += enumerator.Current;
        }

        var result = Tensor.FromArray([sum], [1], tensor.RequiresGrad);
        result.GradFn = tensor.RequiresGrad ? new object() : null; // Placeholder for Phase 2
        return result;
    }

    /// <summary>
    /// Computes the mean of all elements in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A scalar tensor containing the mean.</returns>
    public static Tensor<T> Mean<T>(this Tensor<T> tensor)
        where T : unmanaged, INumber<T>
    {
        var sum = T.Zero;
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            sum += enumerator.Current;
        }

        var count = T.CreateChecked(tensor.ElementCount);
        var mean = sum / count;

        var result = Tensor.FromArray([mean], [1], tensor.RequiresGrad);
        result.GradFn = tensor.RequiresGrad ? new object() : null; // Placeholder for Phase 2
        return result;
    }

    /// <summary>
    /// Matrix multiplication of two 2D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="left">The left tensor.</param>
    /// <param name="right">The right tensor.</param>
    /// <returns>The result of matrix multiplication.</returns>
    /// <exception cref="InvalidOperationException">Thrown when tensors are not compatible for matrix multiplication.</exception>
    public static Tensor<T> MatMul<T>(this Tensor<T> left, Tensor<T> right)
        where T : unmanaged, INumber<T>
    {
        if (left.Rank != 2 || right.Rank != 2)
        {
            throw new InvalidOperationException("MatMul requires 2D tensors");
        }

        var m = left.Shape[0];
        var k = left.Shape[1];
        var n = right.Shape[1];

        if (k != right.Shape[0])
        {
            throw new InvalidOperationException(
                $"Cannot multiply matrices with shapes [{left.Shape[0]}, {left.Shape[1]}] and [{right.Shape[0]}, {right.Shape[1]}]");
        }

        // Create result tensor
        var resultData = System.Numerics.Tensors.Tensor.CreateFromShape<T>([m, n], false);

        // Perform matrix multiplication
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var sum = T.Zero;
                for (int l = 0; l < k; l++)
                {
                    sum += left.Data[i, l] * right.Data[l, j];
                }
                resultData[i, j] = sum;
            }
        }

        var result = new Tensor<T>(resultData, left.Device, left.RequiresGrad || right.RequiresGrad);
        result.GradFn = result.RequiresGrad ? new object() : null; // Placeholder for Phase 2
        return result;
    }
}
