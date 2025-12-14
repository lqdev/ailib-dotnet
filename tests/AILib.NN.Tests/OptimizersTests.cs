using Xunit;
using AILib;
using AILib.NN.Optimizers;
using AILib.NN.Layers;

namespace AILib.NN.Tests;

public class OptimizersTests
{
    [Fact]
    public void SGD_UpdatesParameters()
    {
        var layer = new Linear(2, 1);
        var optimizer = new SGD(layer.Parameters(), lr: 0.1f);

        var input = Tensor.FromArray(new float[] { 1, 2 }, [1, 2], requiresGrad: true);
        var output = layer.Forward(input);
        var loss = output.Sum();

        // Get initial parameter value
        var param = layer.Parameters().First();
        var initialValue = param.Data[0, 0];

        // Backward and step
        loss.Backward();
        optimizer.StepOptimizer();

        // Parameter should have changed
        Assert.NotEqual(initialValue, param.Data[0, 0]);
    }

    [Fact]
    public void Adam_UpdatesParameters()
    {
        var layer = new Linear(2, 1);
        var optimizer = new Adam(layer.Parameters(), lr: 0.01f);

        var input = Tensor.FromArray(new float[] { 1, 2 }, [1, 2], requiresGrad: true);
        var output = layer.Forward(input);
        var loss = output.Sum();

        var param = layer.Parameters().First();
        var initialValue = param.Data[0, 0];

        loss.Backward();
        optimizer.StepOptimizer();

        Assert.NotEqual(initialValue, param.Data[0, 0]);
    }

    [Fact]
    public void AdamW_UpdatesParameters()
    {
        var layer = new Linear(2, 1);
        var optimizer = new AdamW(layer.Parameters(), lr: 0.01f);

        var input = Tensor.FromArray(new float[] { 1, 2 }, [1, 2], requiresGrad: true);
        var output = layer.Forward(input);
        var loss = output.Sum();

        var param = layer.Parameters().First();
        var initialValue = param.Data[0, 0];

        loss.Backward();
        optimizer.StepOptimizer();

        Assert.NotEqual(initialValue, param.Data[0, 0]);
    }

    [Fact]
    public void Optimizer_ZeroGrad_ClearsGradients()
    {
        var layer = new Linear(2, 1);
        var optimizer = new SGD(layer.Parameters(), lr: 0.1f);

        var input = Tensor.FromArray(new float[] { 1, 2 }, [1, 2], requiresGrad: true);
        var output = layer.Forward(input);
        var loss = output.Sum();

        loss.Backward();

        // Parameters should have gradients
        Assert.All(layer.Parameters(), p => Assert.NotNull(p.Grad));

        // Zero gradients via optimizer
        optimizer.ZeroGrad();

        // Gradients should be cleared
        Assert.All(layer.Parameters(), p => Assert.Null(p.Grad));
    }
}
