using System.Numerics;

namespace AILib;

/// <summary>
/// Tensor wrapper that tracks gradients and computation graph.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements (e.g., float, double).</typeparam>
public sealed partial class Tensor<T> : IDisposable where T : unmanaged, INumber<T>
{
    /// <summary>
    /// The underlying .NET tensor data.
    /// </summary>
    public System.Numerics.Tensors.Tensor<T> Data { get; }

    /// <summary>
    /// Gets the device where this tensor resides.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Gets the shape of the tensor as a read-only span.
    /// </summary>
    public ReadOnlySpan<nint> Shape => Data.Lengths;

    /// <summary>
    /// Gets the rank (number of dimensions) of the tensor.
    /// </summary>
    public int Rank => Data.Rank;

    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// </summary>
    public nint ElementCount => Data.FlattenedLength;

    /// <summary>
    /// Gets or sets whether this tensor requires gradient computation.
    /// </summary>
    public bool RequiresGrad { get; set; }

    /// <summary>
    /// Gets or sets the gradient of this tensor.
    /// </summary>
    public Tensor<T>? Grad { get; internal set; }

    /// <summary>
    /// Gets or sets the gradient function for backpropagation.
    /// </summary>
    internal Autograd.GradientFunction<T>? GradFn { get; set; }

    /// <summary>
    /// Initializes a new instance of the Tensor class.
    /// </summary>
    /// <param name="data">The underlying tensor data.</param>
    /// <param name="device">The device where the tensor resides.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    internal Tensor(System.Numerics.Tensors.Tensor<T> data, Device device, bool requiresGrad = false)
    {
        Data = data;
        Device = device;
        RequiresGrad = requiresGrad;
    }

    /// <summary>
    /// Gets the scalar value of a single-element tensor.
    /// </summary>
    /// <returns>The scalar value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when tensor has more than one element.</exception>
    public T Item()
    {
        if (ElementCount != 1)
        {
            throw new InvalidOperationException("Item() can only be called on single-element tensors");
        }
        return Data[0];
    }

    /// <summary>
    /// Zeros out the gradient of this tensor.
    /// </summary>
    public void ZeroGrad()
    {
        if (Grad != null)
        {
            Grad.Dispose();
            Grad = null;
        }
    }

    /// <summary>
    /// Performs backward pass from this tensor.
    /// </summary>
    public void Backward()
    {
        Autograd.GradientEngine.Backward(this);
    }

    /// <summary>
    /// Disposes the tensor and its gradient.
    /// </summary>
    public void Dispose()
    {
        Grad?.Dispose();
    }

    /// <summary>
    /// Returns a string representation of the tensor.
    /// </summary>
    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape.ToArray());
        var gradStr = RequiresGrad ? ", grad_fn=<...>" : "";
        return $"Tensor<{typeof(T).Name}>[{shapeStr}]{gradStr}";
    }
}
