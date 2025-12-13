using System.Numerics;

namespace AILib.Autograd.Operations;

/// <summary>
/// Gradient function for element-wise addition.
/// </summary>
internal class AddBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;

    public AddBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
        Save(input1, input2);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For addition: ∂z/∂x = 1, ∂z/∂y = 1
        // Just pass the gradient through (with broadcasting handled if needed)

        Tensor<T>? grad1 = _input1.RequiresGrad ? gradOutput : null;
        Tensor<T>? grad2 = _input2.RequiresGrad ? gradOutput : null;

        return [grad1, grad2];
    }
}

/// <summary>
/// Gradient function for element-wise subtraction.
/// </summary>
internal class SubtractBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;

    public SubtractBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
        Save(input1, input2);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For subtraction: ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1
        Tensor<T>? grad1 = _input1.RequiresGrad ? gradOutput : null;
        Tensor<T>? grad2 = _input2.RequiresGrad ? -gradOutput : null;

        return [grad1, grad2];
    }
}

/// <summary>
/// Gradient function for element-wise multiplication.
/// </summary>
internal class MulBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;

    public MulBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
        Save(input1, input2);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For multiplication: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x

        Tensor<T>? grad1 = _input1.RequiresGrad ? gradOutput * _input2 : null;
        Tensor<T>? grad2 = _input2.RequiresGrad ? gradOutput * _input1 : null;

        return [grad1, grad2];
    }
}

/// <summary>
/// Gradient function for element-wise division.
/// </summary>
internal class DivideBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;

    public DivideBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
        Save(input1, input2);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For division: ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²
        Tensor<T>? grad1 = _input1.RequiresGrad ? gradOutput / _input2 : null;
        Tensor<T>? grad2 = _input2.RequiresGrad ? -gradOutput * _input1 / (_input2 * _input2) : null;

        return [grad1, grad2];
    }
}

/// <summary>
/// Gradient function for unary negation.
/// </summary>
internal class NegateBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input;

    public NegateBackward(Tensor<T> input)
    {
        _input = input;
        Save(input);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        if (!_input.RequiresGrad)
            return [null];

        // Negation gradient: just negate the gradient
        return [-gradOutput];
    }
}

/// <summary>
/// Gradient function for matrix multiplication.
/// </summary>
internal class MatMulBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;

    public MatMulBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
        Save(input1, input2);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For C = A @ B:
        // ∂L/∂A = ∂L/∂C @ B^T
        // ∂L/∂B = A^T @ ∂L/∂C

        Tensor<T>? grad1 = _input1.RequiresGrad ? gradOutput.MatMul(_input2.Transpose()) : null;
        Tensor<T>? grad2 = _input2.RequiresGrad ? _input1.Transpose().MatMul(gradOutput) : null;

        return [grad1, grad2];
    }
}

/// <summary>
/// Gradient function for sum reduction.
/// </summary>
internal class SumBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input;

    public SumBackward(Tensor<T> input)
    {
        _input = input;
        Save(input);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        if (!_input.RequiresGrad)
            return [null];

        // Sum gradient: broadcast the gradient to match input shape
        // Create a tensor full of the gradient value with same shape as input
        var grad = Tensor.Full(_input.Shape, gradOutput.Item(), _input.RequiresGrad);
        return [grad];
    }
}

/// <summary>
/// Gradient function for mean reduction.
/// </summary>
internal class MeanBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input;

    public MeanBackward(Tensor<T> input)
    {
        _input = input;
        Save(input);
    }

    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        if (!_input.RequiresGrad)
            return [null];

        // Mean gradient: divide by number of elements
        var count = T.CreateChecked(_input.ElementCount);
        var gradValue = gradOutput.Item() / count;
        var grad = Tensor.Full(_input.Shape, gradValue, _input.RequiresGrad);
        return [grad];
    }
}
