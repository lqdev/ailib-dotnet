namespace AILib.NN.Optimizers;

/// <summary>
/// Base class for all optimizers.
/// </summary>
public abstract class Optimizer
{
    /// <summary>
    /// List of parameters to optimize.
    /// </summary>
    protected readonly List<Tensor<float>> Parameters;

    /// <summary>
    /// Current step count.
    /// </summary>
    protected int Step { get; set; }

    /// <summary>
    /// Initializes the optimizer with parameters.
    /// </summary>
    /// <param name="parameters">Parameters to optimize.</param>
    protected Optimizer(IEnumerable<Tensor<float>> parameters)
    {
        Parameters = parameters.ToList();
        Step = 0;
    }

    /// <summary>
    /// Zeros all parameter gradients.
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var param in Parameters)
        {
            param.ZeroGrad();
        }
    }

    /// <summary>
    /// Performs a single optimization step.
    /// </summary>
    public void StepOptimizer()
    {
        Step++;
        UpdateParameters();
    }

    /// <summary>
    /// Updates parameters based on gradients.
    /// </summary>
    protected abstract void UpdateParameters();
}
