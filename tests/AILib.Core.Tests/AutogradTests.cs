using Xunit;

namespace AILib.Tests;

public class AutogradTests
{
    [Fact]
    public void Addition_Backward_ComputesCorrectGradients()
    {
        // Create tensors with gradient tracking
        var a = Tensor.FromArray([2.0f], [1], requiresGrad: true);
        var b = Tensor.FromArray([3.0f], [1], requiresGrad: true);

        // Forward pass: c = a + b
        var c = a + b;

        // Backward pass
        c.Backward();

        // Gradients: ∂c/∂a = 1, ∂c/∂b = 1
        Assert.NotNull(a.Grad);
        Assert.NotNull(b.Grad);
        Assert.Equal(1.0f, a.Grad.Item());
        Assert.Equal(1.0f, b.Grad.Item());
    }

    [Fact]
    public void Multiplication_Backward_ComputesCorrectGradients()
    {
        // Create tensors
        var a = Tensor.FromArray([2.0f], [1], requiresGrad: true);
        var b = Tensor.FromArray([3.0f], [1], requiresGrad: true);

        // Forward: c = a * b = 6
        var c = a * b;

        // Backward
        c.Backward();

        // Gradients: ∂c/∂a = b = 3, ∂c/∂b = a = 2
        Assert.NotNull(a.Grad);
        Assert.NotNull(b.Grad);
        Assert.Equal(3.0f, a.Grad.Item());
        Assert.Equal(2.0f, b.Grad.Item());
    }

    [Fact]
    public void ChainedOperations_Backward_ComputesCorrectGradients()
    {
        // Create tensors
        var x = Tensor.FromArray([2.0f], [1], requiresGrad: true);
        var y = Tensor.FromArray([3.0f], [1], requiresGrad: true);

        // Forward: z = (x + y) * x = 5 * 2 = 10
        var sum = x + y;
        var z = sum * x;

        // Backward
        z.Backward();

        // ∂z/∂x = ∂z/∂sum * ∂sum/∂x + ∂z/∂x (second term)
        //       = x * 1 + sum
        //       = 2 + 5 = 7
        // ∂z/∂y = ∂z/∂sum * ∂sum/∂y = x * 1 = 2
        Assert.NotNull(x.Grad);
        Assert.NotNull(y.Grad);
        Assert.Equal(7.0f, x.Grad.Item());
        Assert.Equal(2.0f, y.Grad.Item());
    }

    [Fact]
    public void MatMul_Backward_ComputesCorrectGradients()
    {
        // Create matrices
        var a = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2], requiresGrad: true);
        var b = Tensor.FromArray(new float[] { 5, 6, 7, 8 }, [2, 2], requiresGrad: true);

        // Forward: c = a @ b
        var c = a.MatMul(b);

        // Sum to get scalar for backward
        var loss = c.Sum();

        // Backward
        loss.Backward();

        // Check that gradients exist
        Assert.NotNull(a.Grad);
        Assert.NotNull(b.Grad);

        // Gradient shapes should match input shapes
        Assert.Equal(2, a.Grad.Rank);
        Assert.Equal(2, a.Grad.Shape[0]);
        Assert.Equal(2, a.Grad.Shape[1]);
    }

    [Fact]
    public void Sum_Backward_BroadcastsGradient()
    {
        // Create tensor
        var x = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2], requiresGrad: true);

        // Forward: sum all elements
        var sum = x.Sum();

        // Backward
        sum.Backward();

        // Gradient should be 1 for all elements
        Assert.NotNull(x.Grad);
        Assert.Equal(1.0f, x.Grad.Data[0, 0]);
        Assert.Equal(1.0f, x.Grad.Data[0, 1]);
        Assert.Equal(1.0f, x.Grad.Data[1, 0]);
        Assert.Equal(1.0f, x.Grad.Data[1, 1]);
    }

    [Fact]
    public void Mean_Backward_ScalesGradient()
    {
        // Create tensor
        var x = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2], requiresGrad: true);

        // Forward: mean of all elements
        var mean = x.Mean();

        // Backward
        mean.Backward();

        // Gradient should be 1/4 for all elements
        Assert.NotNull(x.Grad);
        Assert.Equal(0.25f, x.Grad.Data[0, 0], precision: 5);
        Assert.Equal(0.25f, x.Grad.Data[0, 1], precision: 5);
        Assert.Equal(0.25f, x.Grad.Data[1, 0], precision: 5);
        Assert.Equal(0.25f, x.Grad.Data[1, 1], precision: 5);
    }

    [Fact]
    public void ZeroGrad_ClearsGradients()
    {
        var x = Tensor.FromArray([2.0f], [1], requiresGrad: true);
        var y = x * x;

        y.Backward();

        Assert.NotNull(x.Grad);

        // Zero the gradient
        x.ZeroGrad();

        Assert.Null(x.Grad);
    }

    [Fact]
    public void Backward_ThrowsOnNonScalar()
    {
        var x = Tensor.FromArray([1.0f, 2.0f], [2], requiresGrad: true);

        Assert.Throws<InvalidOperationException>(() => x.Backward());
    }

    [Fact]
    public void Backward_ThrowsOnNoGrad()
    {
        var x = Tensor.FromArray([1.0f], [1], requiresGrad: false);

        Assert.Throws<InvalidOperationException>(() => x.Backward());
    }
}
