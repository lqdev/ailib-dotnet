namespace AILib.NN.Layers;

/// <summary>
/// Dropout layer for regularization.
/// </summary>
public class Dropout : Module
{
    private readonly float _p;

    /// <summary>
    /// Initializes Dropout layer.
    /// </summary>
    /// <param name="p">Probability of dropping an element.</param>
    public Dropout(float p = 0.5f)
    {
        if (p < 0 || p > 1)
            throw new ArgumentException("Dropout probability must be in [0, 1]", nameof(p));

        _p = p;
    }

    /// <summary>
    /// Forward pass with dropout.
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        if (!Training || _p == 0)
            return input;

        // Generate random mask
        var mask = Tensor.Rand<float>(input.Shape);

        // Create result tensor
        var result = System.Numerics.Tensors.Tensor.CreateFromShape<float>(input.Shape, false);

        // Apply mask: keep elements where random > p, and scale by 1/(1-p)
        var keepProb = 1.0f - _p;
        var scale = 1.0f / keepProb;

        var inputEnum = input.Data.GetEnumerator();
        var maskEnum = mask.Data.GetEnumerator();
        int idx = 0;

        while (inputEnum.MoveNext() && maskEnum.MoveNext())
        {
            var positions = GetPositions(idx, input.Shape);
            if (maskEnum.Current > _p)
            {
                result[positions] = inputEnum.Current * scale;
            }
            else
            {
                result[positions] = 0f;
            }
            idx++;
        }

        return Tensor.FromTensor(result, input.RequiresGrad);
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

/// <summary>
/// Layer Normalization.
/// </summary>
public class LayerNorm : Module
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float> _bias;
    private readonly int _normalizedShape;
    private readonly float _eps;

    /// <summary>
    /// Initializes LayerNorm layer.
    /// </summary>
    /// <param name="normalizedShape">Size of the normalized dimension.</param>
    /// <param name="eps">Small constant for numerical stability.</param>
    public LayerNorm(int normalizedShape, float eps = 1e-5f)
    {
        _normalizedShape = normalizedShape;
        _eps = eps;

        _weight = Tensor.Ones<float>([normalizedShape], requiresGrad: true);
        _bias = Tensor.Zeros<float>([normalizedShape], requiresGrad: true);

        RegisterParameter("weight", _weight);
        RegisterParameter("bias", _bias);
    }

    /// <summary>
    /// Forward pass: normalize over last dimension.
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        // For now, implement a simplified version that works on 2D tensors
        // A full implementation would need dimension-aware mean/variance computation

        // Compute mean
        var mean = input.Mean();
        var meanValue = mean.Item();

        // Center the data
        var centered = input - meanValue;

        // Compute variance
        var squared = centered * centered;
        var variance = squared.Mean();
        var varianceValue = variance.Item();

        // Normalize
        var stdValue = MathF.Sqrt(varianceValue + _eps);
        var normalized = centered / stdValue;

        // Apply affine transformation: y = γ * x + β
        var result = normalized * _weight + _bias;

        return result;
    }
}
