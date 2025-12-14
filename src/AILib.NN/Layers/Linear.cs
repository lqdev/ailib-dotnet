namespace AILib.NN.Layers;

/// <summary>
/// Fully connected linear layer: y = xW^T + b
/// </summary>
public class Linear : Module
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float>? _bias;
    private readonly int _inFeatures;
    private readonly int _outFeatures;

    /// <summary>
    /// Initializes a new Linear layer.
    /// </summary>
    /// <param name="inFeatures">Number of input features.</param>
    /// <param name="outFeatures">Number of output features.</param>
    /// <param name="bias">Whether to include bias term.</param>
    public Linear(int inFeatures, int outFeatures, bool bias = true)
    {
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;

        // Initialize weight with Kaiming uniform
        var k = 1.0f / MathF.Sqrt(inFeatures);
        var weightData = new float[outFeatures * inFeatures];
        var random = Random.Shared;
        for (int i = 0; i < weightData.Length; i++)
        {
            weightData[i] = (float)(random.NextDouble() * 2 * k - k);
        }
        _weight = Tensor.FromArray(weightData, [outFeatures, inFeatures], requiresGrad: true);
        RegisterParameter("weight", _weight);

        if (bias)
        {
            var biasData = new float[outFeatures];
            for (int i = 0; i < biasData.Length; i++)
            {
                biasData[i] = (float)(random.NextDouble() * 2 * k - k);
            }
            _bias = Tensor.FromArray(biasData, [outFeatures], requiresGrad: true);
            RegisterParameter("bias", _bias);
        }
    }

    /// <summary>
    /// Forward pass: y = xW^T + b
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, in_features] or [*, in_features].</param>
    /// <returns>Output tensor of shape [batch, out_features] or [*, out_features].</returns>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        // input: [batch, in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features]

        var output = input.MatMul(_weight.Transpose());

        if (_bias != null)
        {
            // Broadcast bias across batch dimension
            output = output + _bias;
        }

        return output;
    }
}
