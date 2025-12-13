using Xunit;
using AILib;
using AILib.NN.Loss;

namespace AILib.NN.Tests;

public class LossTests
{
    [Fact]
    public void MSELoss_ComputesCorrectly()
    {
        var loss = new MSELoss();
        var predictions = Tensor.FromArray(new float[] { 1, 2, 3 }, [3]);
        var targets = Tensor.FromArray(new float[] { 1, 2, 3 }, [3]);

        var result = loss.Forward(predictions, targets);

        // MSE of identical tensors should be 0
        Assert.Equal(0f, result.Item(), precision: 5);
    }

    [Fact]
    public void MSELoss_ComputesDifferences()
    {
        var loss = new MSELoss();
        var predictions = Tensor.FromArray(new float[] { 0, 0 }, [2]);
        var targets = Tensor.FromArray(new float[] { 1, 1 }, [2]);

        var result = loss.Forward(predictions, targets);

        // MSE = mean((0-1)² + (0-1)²) = mean(1 + 1) = 1
        Assert.Equal(1f, result.Item(), precision: 5);
    }

    [Fact]
    public void BCELoss_ComputesCorrectly()
    {
        var loss = new BCELoss();
        var predictions = Tensor.FromArray(new float[] { 0.5f, 0.5f }, [2]);
        var targets = Tensor.FromArray(new float[] { 1f, 0f }, [2]);

        var result = loss.Forward(predictions, targets);

        // BCE should be positive
        Assert.True(result.Item() > 0);
    }

    [Fact]
    public void CrossEntropyLoss_ForwardOneHot_Works()
    {
        var loss = new CrossEntropyLoss();
        var logits = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var targets = Tensor.FromArray(new float[] { 1, 0, 0, 0, 1, 0 }, [2, 3]);

        var result = loss.ForwardOneHot(logits, targets);

        // Loss should be positive
        Assert.True(result.Item() > 0);
    }
}
