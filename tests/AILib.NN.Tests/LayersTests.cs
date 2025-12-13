using Xunit;
using AILib;
using AILib.NN.Layers;

namespace AILib.NN.Tests;

public class LayersTests
{
    [Fact]
    public void Linear_CreatesCorrectOutputShape()
    {
        var layer = new Linear(10, 5);
        var input = Tensor.Randn<float>([3, 10]);  // batch=3, features=10

        var output = layer.Forward(input);

        Assert.Equal(2, output.Rank);
        Assert.Equal(3, output.Shape[0]);  // batch size preserved
        Assert.Equal(5, output.Shape[1]);  // output features
    }

    [Fact]
    public void Linear_HasCorrectParameters()
    {
        var layer = new Linear(10, 5, bias: true);
        var parameters = layer.Parameters().ToList();

        Assert.Equal(2, parameters.Count);  // weight and bias
        Assert.All(parameters, p => Assert.True(p.RequiresGrad));
    }

    [Fact]
    public void ReLU_AppliesActivation()
    {
        var relu = new AILib.NN.Activations.ReLU();
        var input = Tensor.FromArray(new float[] { -1, 0, 1, 2 }, [4]);

        var output = relu.Forward(input);

        Assert.Equal(0f, output.Data[0]);   // -1 -> 0
        Assert.Equal(0f, output.Data[1]);   // 0 -> 0
        Assert.Equal(1f, output.Data[2]);   // 1 -> 1
        Assert.Equal(2f, output.Data[3]);   // 2 -> 2
    }

    [Fact]
    public void Sigmoid_OutputsInRange()
    {
        var sigmoid = new AILib.NN.Activations.Sigmoid();
        var input = Tensor.Randn<float>([10]);

        var output = sigmoid.Forward(input);

        // All sigmoid outputs should be in (0, 1)
        var enumerator = output.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            Assert.InRange(enumerator.Current, 0f, 1f);
        }
    }

    [Fact]
    public void Dropout_InEvalModeDoesNotDropout()
    {
        var dropout = new Dropout(0.5f);
        dropout.Eval();  // Set to eval mode

        var input = Tensor.Ones<float>([100]);
        var output = dropout.Forward(input);

        // In eval mode, output should equal input
        var inputEnum = input.Data.GetEnumerator();
        var outputEnum = output.Data.GetEnumerator();
        while (inputEnum.MoveNext() && outputEnum.MoveNext())
        {
            Assert.Equal(inputEnum.Current, outputEnum.Current);
        }
    }

    [Fact]
    public void LayerNorm_NormalizesData()
    {
        var layerNorm = new LayerNorm(10);
        var input = Tensor.Randn<float>([5, 10]);

        var output = layerNorm.Forward(input);

        Assert.Equal(input.Shape[0], output.Shape[0]);
        Assert.Equal(input.Shape[1], output.Shape[1]);
    }

    [Fact]
    public void Module_ZeroGrad_ClearsGradients()
    {
        var layer = new Linear(5, 3);
        var input = Tensor.Randn<float>([2, 5], requiresGrad: true);

        var output = layer.Forward(input);
        var loss = output.Sum();
        loss.Backward();

        // Parameters should have gradients
        Assert.All(layer.Parameters(), p => Assert.NotNull(p.Grad));

        // Zero gradients
        layer.ZeroGrad();

        // Gradients should be cleared
        Assert.All(layer.Parameters(), p => Assert.Null(p.Grad));
    }

    [Fact]
    public void Module_TrainEvalModes()
    {
        var module = new Dropout(0.5f);

        Assert.True(module.Training);  // Default is training mode

        module.Eval();
        Assert.False(module.Training);

        module.Train();
        Assert.True(module.Training);
    }
}
