using Xunit;

namespace AILib.Tests;

public class TensorOperationsTests
{
    [Fact]
    public void Reshape_ChangesShape()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var reshaped = tensor.Reshape([3, 2]);

        Assert.Equal(2, reshaped.Rank);
        Assert.Equal(3, reshaped.Shape[0]);
        Assert.Equal(2, reshaped.Shape[1]);
        Assert.Equal(6, reshaped.ElementCount);
    }

    [Fact]
    public void Reshape_ThrowsOnIncompatibleShape()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2]);

        Assert.Throws<ArgumentException>(() => tensor.Reshape([3, 3]));
    }

    [Fact]
    public void Flatten_Creates1DTensor()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var flattened = tensor.Flatten();

        Assert.Equal(1, flattened.Rank);
        Assert.Equal(4, flattened.ElementCount);
    }

    [Fact]
    public void Transpose_Swaps2DDimensions()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var transposed = tensor.Transpose();

        Assert.Equal(2, transposed.Rank);
        Assert.Equal(3, transposed.Shape[0]);
        Assert.Equal(2, transposed.Shape[1]);

        // Check values are correctly transposed
        Assert.Equal(1f, transposed.Data[0, 0]);
        Assert.Equal(4f, transposed.Data[0, 1]);
        Assert.Equal(2f, transposed.Data[1, 0]);
        Assert.Equal(5f, transposed.Data[1, 1]);
    }

    [Fact]
    public void Transpose_ThrowsOnNon2DTensor()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [4]);

        Assert.Throws<InvalidOperationException>(() => tensor.Transpose());
    }

    [Fact]
    public void Sum_ComputesTotalSum()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var sum = tensor.Sum();

        Assert.Equal(1, sum.ElementCount);
        Assert.Equal(10f, sum.Item());
    }

    [Fact]
    public void Mean_ComputesAverage()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var mean = tensor.Mean();

        Assert.Equal(1, mean.ElementCount);
        Assert.Equal(2.5f, mean.Item());
    }

    [Fact]
    public void MatMul_Multiplies2x3And3x2()
    {
        var a = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var b = Tensor.FromArray(new float[] { 7, 8, 9, 10, 11, 12 }, [3, 2]);
        var c = a.MatMul(b);

        Assert.Equal(2, c.Rank);
        Assert.Equal(2, c.Shape[0]);
        Assert.Equal(2, c.Shape[1]);

        // Verify computation: c[0,0] = 1*7 + 2*9 + 3*11 = 58
        Assert.Equal(58f, c.Data[0, 0]);
        // c[0,1] = 1*8 + 2*10 + 3*12 = 64
        Assert.Equal(64f, c.Data[0, 1]);
        // c[1,0] = 4*7 + 5*9 + 6*11 = 139
        Assert.Equal(139f, c.Data[1, 0]);
        // c[1,1] = 4*8 + 5*10 + 6*12 = 154
        Assert.Equal(154f, c.Data[1, 1]);
    }

    [Fact]
    public void MatMul_ThrowsOnIncompatibleShapes()
    {
        var a = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2]);
        var b = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, [3, 2]);

        Assert.Throws<InvalidOperationException>(() => a.MatMul(b));
    }

    [Fact]
    public void MatMul_ThrowsOnNon2DTensor()
    {
        var a = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [4]);
        var b = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [4]);

        Assert.Throws<InvalidOperationException>(() => a.MatMul(b));
    }

    [Fact]
    public void View_CreatesNewShapeView()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, [2, 3]);
        var viewed = tensor.View([3, 2]);

        Assert.Equal(2, viewed.Rank);
        Assert.Equal(3, viewed.Shape[0]);
        Assert.Equal(2, viewed.Shape[1]);
    }

    [Fact]
    public void Permute_ReordersDimensions()
    {
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, [2, 2, 2]);
        var permuted = tensor.Permute([2, 0, 1]);

        Assert.Equal(3, permuted.Rank);
        Assert.Equal(2, permuted.Shape[0]);
        Assert.Equal(2, permuted.Shape[1]);
        Assert.Equal(2, permuted.Shape[2]);
    }
}
