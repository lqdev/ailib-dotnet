namespace AILib.NN;

/// <summary>
/// Base class for all neural network modules/layers.
/// </summary>
public abstract class Module : IDisposable
{
    private readonly Dictionary<string, Tensor<float>> _parameters = new();
    private readonly Dictionary<string, Module> _submodules = new();
    private readonly Dictionary<string, Tensor<float>> _buffers = new();

    /// <summary>
    /// Gets or sets whether the module is in training mode.
    /// </summary>
    public bool Training { get; set; } = true;

    /// <summary>
    /// Forward pass through the module.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    public abstract Tensor<float> Forward(Tensor<float> input);

    /// <summary>
    /// Registers a parameter (trainable tensor) with the module.
    /// </summary>
    /// <param name="name">Parameter name.</param>
    /// <param name="tensor">Parameter tensor.</param>
    protected void RegisterParameter(string name, Tensor<float> tensor)
    {
        tensor.RequiresGrad = true;
        _parameters[name] = tensor;
    }

    /// <summary>
    /// Registers a buffer (non-trainable tensor) with the module.
    /// </summary>
    /// <param name="name">Buffer name.</param>
    /// <param name="tensor">Buffer tensor.</param>
    protected void RegisterBuffer(string name, Tensor<float> tensor)
    {
        _buffers[name] = tensor;
    }

    /// <summary>
    /// Registers a submodule.
    /// </summary>
    /// <param name="name">Submodule name.</param>
    /// <param name="module">Submodule.</param>
    protected void RegisterModule(string name, Module module)
    {
        _submodules[name] = module;
    }

    /// <summary>
    /// Gets all parameters, optionally including submodule parameters.
    /// </summary>
    /// <param name="recurse">Whether to include submodule parameters.</param>
    /// <returns>Enumerable of all parameters.</returns>
    public IEnumerable<Tensor<float>> Parameters(bool recurse = true)
    {
        foreach (var param in _parameters.Values)
            yield return param;

        if (recurse)
        {
            foreach (var module in _submodules.Values)
            {
                foreach (var param in module.Parameters(recurse: true))
                    yield return param;
            }
        }
    }

    /// <summary>
    /// Sets the module to training mode.
    /// </summary>
    public virtual void Train()
    {
        Training = true;
        foreach (var module in _submodules.Values)
            module.Train();
    }

    /// <summary>
    /// Sets the module to evaluation mode.
    /// </summary>
    public virtual void Eval()
    {
        Training = false;
        foreach (var module in _submodules.Values)
            module.Eval();
    }

    /// <summary>
    /// Zeros out all parameter gradients.
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var param in Parameters())
        {
            param.ZeroGrad();
        }
    }

    /// <summary>
    /// Disposes the module and all its resources.
    /// </summary>
    public virtual void Dispose()
    {
        foreach (var param in _parameters.Values)
            param.Dispose();

        foreach (var buffer in _buffers.Values)
            buffer.Dispose();

        foreach (var module in _submodules.Values)
            module.Dispose();
    }
}
