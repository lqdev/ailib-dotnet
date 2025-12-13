namespace AILib.NN.Optimizers;

/// <summary>
/// Stochastic Gradient Descent optimizer.
/// </summary>
public class SGD : Optimizer
{
    private readonly float _lr;
    private readonly float _momentum;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _velocities;

    /// <summary>
    /// Initializes SGD optimizer.
    /// </summary>
    /// <param name="parameters">Parameters to optimize.</param>
    /// <param name="lr">Learning rate.</param>
    /// <param name="momentum">Momentum factor.</param>
    /// <param name="weightDecay">Weight decay (L2 penalty).</param>
    public SGD(
        IEnumerable<Tensor<float>> parameters,
        float lr = 0.01f,
        float momentum = 0.0f,
        float weightDecay = 0.0f)
        : base(parameters)
    {
        _lr = lr;
        _momentum = momentum;
        _weightDecay = weightDecay;
        _velocities = new Dictionary<Tensor<float>, Tensor<float>>();

        if (momentum > 0)
        {
            foreach (var param in Parameters)
            {
                _velocities[param] = Tensor.Zeros<float>(param.Shape);
            }
        }
    }

    /// <summary>
    /// Updates parameters using SGD with optional momentum.
    /// </summary>
    protected override void UpdateParameters()
    {
        foreach (var param in Parameters)
        {
            if (param.Grad == null)
                continue;

            var grad = param.Grad;

            // Apply weight decay
            if (_weightDecay > 0)
            {
                grad = grad + _weightDecay * param;
            }

            // Apply momentum
            if (_momentum > 0)
            {
                var velocity = _velocities[param];
                velocity = _momentum * velocity + grad;
                _velocities[param] = velocity;
                grad = velocity;
            }

            // Update parameters: θ = θ - lr * grad
            var update = param - _lr * grad;
            
            // Copy data back using Fill operation
            UpdateTensorInPlace(param, update);
        }
    }

    private static void UpdateTensorInPlace(Tensor<float> dest, Tensor<float> source)
    {
        // Fill dest tensor with source values
        var srcEnum = source.Data.GetEnumerator();
        var destData = dest.Data;
        int idx = 0;
        
        while (srcEnum.MoveNext())
        {
            var positions = GetPositions(idx, dest.Shape);
            destData[positions] = srcEnum.Current;
            idx++;
        }
    }

    private static nint[] GetPositions(int linearIndex, ReadOnlySpan<nint> shape)
    {
        var positions = new nint[shape.Length];
        var remaining = linearIndex;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            positions[i] = remaining % shape[i];
            remaining /= (int)shape[i];
        }
        return positions;
    }
}
