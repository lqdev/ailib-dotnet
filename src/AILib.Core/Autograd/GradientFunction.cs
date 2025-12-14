using System.Numerics;

namespace AILib.Autograd;

/// <summary>
/// Base class for gradient computation functions in the computation graph.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
public abstract class GradientFunction<T> where T : unmanaged, INumber<T>
{
    /// <summary>
    /// Gets the list of tensors saved for backward computation.
    /// </summary>
    internal List<Tensor<T>> SavedTensors { get; } = new();

    /// <summary>
    /// Computes gradients for inputs given the output gradient.
    /// </summary>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <returns>Array of gradients for each input (null if input doesn't require grad).</returns>
    public abstract Tensor<T>?[] Backward(Tensor<T> gradOutput);

    /// <summary>
    /// Saves tensors for use in the backward pass.
    /// </summary>
    /// <param name="tensors">Tensors to save.</param>
    protected void Save(params Tensor<T>[] tensors)
    {
        SavedTensors.AddRange(tensors);
    }
}
