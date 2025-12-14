using System.Numerics;

namespace AILib.Autograd;

/// <summary>
/// Engine for computing gradients via reverse-mode automatic differentiation.
/// </summary>
public static class GradientEngine
{
    /// <summary>
    /// Performs backward pass from a scalar loss tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="loss">The scalar loss tensor to backpropagate from.</param>
    /// <exception cref="InvalidOperationException">Thrown when loss doesn't require gradients or isn't scalar.</exception>
    public static void Backward<T>(Tensor<T> loss) where T : unmanaged, INumber<T>
    {
        if (!loss.RequiresGrad)
        {
            throw new InvalidOperationException("Cannot backward on tensor that doesn't require gradients");
        }

        if (loss.ElementCount != 1)
        {
            throw new InvalidOperationException("Backward can only be called on scalar tensors");
        }

        // Initialize gradient of loss as 1
        loss.Grad = Tensor.Ones<T>(loss.Shape);

        // Build topological order of computation graph
        var topo = new List<Tensor<T>>();
        var visited = new HashSet<Tensor<T>>();
        TopologicalSort(loss, topo, visited);

        // Reverse-mode autodiff: traverse in reverse topological order
        foreach (var tensor in topo.AsEnumerable().Reverse())
        {
            if (tensor.GradFn == null || tensor.Grad == null)
                continue;

            // Compute gradients for this node's inputs
            var inputGrads = tensor.GradFn.Backward(tensor.Grad);

            // Accumulate gradients in the input tensors
            var inputs = tensor.GradFn.SavedTensors;
            for (int i = 0; i < inputs.Count && i < inputGrads.Length; i++)
            {
                if (inputGrads[i] != null && inputs[i].RequiresGrad)
                {
                    if (inputs[i].Grad == null)
                    {
                        inputs[i].Grad = inputGrads[i];
                    }
                    else
                    {
                        // Accumulate gradients (for nodes with multiple uses)
                        inputs[i].Grad = inputs[i].Grad + inputGrads[i]!;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Builds a topological ordering of the computation graph.
    /// </summary>
    private static void TopologicalSort<T>(
        Tensor<T> tensor,
        List<Tensor<T>> topo,
        HashSet<Tensor<T>> visited)
        where T : unmanaged, INumber<T>
    {
        if (visited.Contains(tensor))
            return;

        visited.Add(tensor);

        // Visit all inputs first (depth-first)
        if (tensor.GradFn != null)
        {
            foreach (var input in tensor.GradFn.SavedTensors)
            {
                TopologicalSort(input, topo, visited);
            }
        }

        // Add this tensor after its inputs
        topo.Add(tensor);
    }
}
