namespace AILib.NN.Optimizers;

/// <summary>
/// Adam optimizer (Adaptive Moment Estimation).
/// </summary>
public class Adam : Optimizer
{
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _m; // First moment
    private readonly Dictionary<Tensor<float>, Tensor<float>> _v; // Second moment

    /// <summary>
    /// Initializes Adam optimizer.
    /// </summary>
    /// <param name="parameters">Parameters to optimize.</param>
    /// <param name="lr">Learning rate.</param>
    /// <param name="beta1">Exponential decay rate for first moment.</param>
    /// <param name="beta2">Exponential decay rate for second moment.</param>
    /// <param name="eps">Small constant for numerical stability.</param>
    /// <param name="weightDecay">Weight decay (L2 penalty).</param>
    public Adam(
        IEnumerable<Tensor<float>> parameters,
        float lr = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.0f)
        : base(parameters)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;

        _m = new Dictionary<Tensor<float>, Tensor<float>>();
        _v = new Dictionary<Tensor<float>, Tensor<float>>();

        foreach (var param in Parameters)
        {
            _m[param] = Tensor.Zeros<float>(param.Shape);
            _v[param] = Tensor.Zeros<float>(param.Shape);
        }
    }

    /// <summary>
    /// Updates parameters using Adam algorithm.
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

            // Update biased first moment estimate: m = β₁ * m + (1 - β₁) * g
            var m = _m[param];
            m = _beta1 * m + (1 - _beta1) * grad;
            _m[param] = m;

            // Update biased second moment estimate: v = β₂ * v + (1 - β₂) * g²
            var v = _v[param];
            var gradSquared = grad * grad;
            v = _beta2 * v + (1 - _beta2) * gradSquared;
            _v[param] = v;

            // Bias correction
            var beta1Power = MathF.Pow(_beta1, Step);
            var beta2Power = MathF.Pow(_beta2, Step);
            var mHat = m / (1 - beta1Power);
            var vHat = v / (1 - beta2Power);

            // Update parameters: θ = θ - lr * m̂ / (√v̂ + ε)
            var denominator = vHat.Sqrt() + _eps;
            var update = param - _lr * mHat / denominator;

            // Copy data back
            UpdateTensorInPlace(param, update);
        }
    }

    private static void UpdateTensorInPlace(Tensor<float> dest, Tensor<float> source)
    {
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
