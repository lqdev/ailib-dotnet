namespace AILib.NN.Activations;

/// <summary>
/// ReLU activation module.
/// </summary>
public class ReLU : Module
{
    /// <summary>
    /// Forward pass: max(0, x)
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input.ReLU();
    }
}

/// <summary>
/// Sigmoid activation module.
/// </summary>
public class Sigmoid : Module
{
    /// <summary>
    /// Forward pass: 1 / (1 + exp(-x))
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input.Sigmoid();
    }
}

/// <summary>
/// Tanh activation module.
/// </summary>
public class Tanh : Module
{
    /// <summary>
    /// Forward pass: tanh(x)
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input.Tanh();
    }
}

/// <summary>
/// GELU activation module.
/// </summary>
public class GELU : Module
{
    /// <summary>
    /// Forward pass: GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input.GELU();
    }
}

/// <summary>
/// Softmax activation module.
/// </summary>
public class Softmax : Module
{
    private readonly int _dim;

    /// <summary>
    /// Initializes Softmax activation.
    /// </summary>
    /// <param name="dim">Dimension along which to compute softmax.</param>
    public Softmax(int dim = -1)
    {
        _dim = dim;
    }

    /// <summary>
    /// Forward pass: softmax(x) = exp(x) / sum(exp(x))
    /// </summary>
    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input.Softmax(_dim);
    }
}
