using Xunit;

namespace AILib.Tests;

public class TensorCreationTests
{
    [Fact]
    public void FromArray_CreatesCorrectShape()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6 };
        var tensor = Tensor.FromArray(data, [2, 3]);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(6, tensor.ElementCount);
    }

    [Fact]
    public void Zeros_CreatesZeroFilledTensor()
    {
        var tensor = Tensor.Zeros<float>([3, 3]);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(9, tensor.ElementCount);

        // Verify all elements are zero
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            Assert.Equal(0f, enumerator.Current);
        }
    }

    [Fact]
    public void Ones_CreatesOnesFilledTensor()
    {
        var tensor = Tensor.Ones<float>([2, 2]);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(4, tensor.ElementCount);

        // Verify all elements are one
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            Assert.Equal(1f, enumerator.Current);
        }
    }

    [Fact]
    public void Full_CreatesValueFilledTensor()
    {
        var tensor = Tensor.Full<float>([3, 2], 5.0f);

        Assert.Equal(6, tensor.ElementCount);

        // Verify all elements are 5.0
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            Assert.Equal(5.0f, enumerator.Current);
        }
    }

    [Fact]
    public void Randn_CreatesNormalDistribution()
    {
        var tensor = Tensor.Randn<float>([10000]);

        // Calculate mean and standard deviation
        var sum = 0.0;
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            sum += enumerator.Current;
        }
        var mean = sum / tensor.ElementCount;

        var variance = 0.0;
        enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            var diff = enumerator.Current - mean;
            variance += diff * diff;
        }
        variance /= tensor.ElementCount;
        var std = Math.Sqrt(variance);

        // Mean should be close to 0, std should be close to 1
        Assert.InRange(mean, -0.1, 0.1);
        Assert.InRange(std, 0.9, 1.1);
    }

    [Fact]
    public void Rand_CreatesUniformDistribution()
    {
        var tensor = Tensor.Rand<float>([1000]);

        // All values should be in [0, 1)
        var enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            Assert.InRange(enumerator.Current, 0.0f, 1.0f);
        }

        // Mean should be approximately 0.5
        var sum = 0.0;
        enumerator = tensor.Data.GetEnumerator();
        while (enumerator.MoveNext())
        {
            sum += enumerator.Current;
        }
        var mean = sum / tensor.ElementCount;
        Assert.InRange(mean, 0.4, 0.6);
    }

    [Fact]
    public void Tensor_WithRequiresGrad_SetsProperty()
    {
        var tensor = Tensor.Zeros<float>([2, 2], requiresGrad: true);

        Assert.True(tensor.RequiresGrad);
    }

    [Fact]
    public void Tensor_ToString_ReturnsCorrectFormat()
    {
        var tensor = Tensor.Zeros<float>([2, 3]);

        var str = tensor.ToString();

        Assert.Contains("Tensor<Single>", str);
        Assert.Contains("[2, 3]", str);
    }

    [Fact]
    public void Tensor_Item_ReturnsSingleValue()
    {
        var tensor = Tensor.FromArray([42.0f], [1]);

        var value = tensor.Item();

        Assert.Equal(42.0f, value);
    }

    [Fact]
    public void Tensor_Item_ThrowsOnMultipleElements()
    {
        var tensor = Tensor.FromArray([1.0f, 2.0f], [2]);

        Assert.Throws<InvalidOperationException>(() => tensor.Item());
    }
}
