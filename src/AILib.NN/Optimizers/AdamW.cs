namespace AILib.NN.Optimizers;

/// <summary>
/// AdamW optimizer (Adam with decoupled weight decay).
/// </summary>
public class AdamW : Optimizer
{
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _m;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _v;

    /// <summary>
    /// Initializes AdamW optimizer.
    /// </summary>
    /// <param name="parameters">Parameters to optimize.</param>
    /// <param name="lr">Learning rate.</param>
    /// <param name="beta1">Exponential decay rate for first moment.</param>
    /// <param name="beta2">Exponential decay rate for second moment.</param>
    /// <param name="eps">Small constant for numerical stability.</param>
    /// <param name="weightDecay">Weight decay (decoupled from gradient).</param>
    public AdamW(
        IEnumerable<Tensor<float>> parameters,
        float lr = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.01f)
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
    /// Updates parameters using AdamW algorithm (decoupled weight decay).
    /// </summary>
    protected override void UpdateParameters()
    {
        foreach (var param in Parameters)
        {
            if (param.Grad == null)
                continue;

            var grad = param.Grad;

            // Update biased first moment estimate
            var m = _m[param];
            m = _beta1 * m + (1 - _beta1) * grad;
            _m[param] = m;

            // Update biased second moment estimate
            var v = _v[param];
            var gradSquared = grad * grad;
            v = _beta2 * v + (1 - _beta2) * gradSquared;
            _v[param] = v;

            // Bias correction
            var beta1Power = MathF.Pow(_beta1, Step);
            var beta2Power = MathF.Pow(_beta2, Step);
            var mHat = m / (1 - beta1Power);
            var vHat = v / (1 - beta2Power);

            // Update with decoupled weight decay: θ = θ(1 - lr*λ) - lr * m̂ / (√v̂ + ε)
            var denominator = vHat.Sqrt() + _eps;
            var adamUpdate = _lr * mHat / denominator;
            var weightDecayUpdate = _lr * _weightDecay * param;
            var update = param - weightDecayUpdate - adamUpdate;

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
