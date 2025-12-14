using System.Numerics;
using System.Numerics.Tensors;
using AILib.Autograd;
using AILib.Autograd.Operations;

namespace AILib;

public sealed partial class Tensor<T>
{
    /// <summary>
    /// Element-wise addition of two tensors.
    /// </summary>
    public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
    {
        var leftSpan = left.Data.AsReadOnlyTensorSpan();
        var rightSpan = right.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Add(leftSpan, rightSpan);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);

        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new AddBackward<T>(left, right);
        }

        return tensor;
    }

    /// <summary>
    /// Element-wise subtraction of two tensors.
    /// </summary>
    public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
    {
        var leftSpan = left.Data.AsReadOnlyTensorSpan();
        var rightSpan = right.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Subtract(leftSpan, rightSpan);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);

        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new SubtractBackward<T>(left, right);
        }

        return tensor;
    }

    /// <summary>
    /// Element-wise multiplication of two tensors.
    /// </summary>
    public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
    {
        var leftSpan = left.Data.AsReadOnlyTensorSpan();
        var rightSpan = right.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Multiply(leftSpan, rightSpan);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);

        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new MulBackward<T>(left, right);
        }

        return tensor;
    }

    /// <summary>
    /// Element-wise division of two tensors.
    /// </summary>
    public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
    {
        var leftSpan = left.Data.AsReadOnlyTensorSpan();
        var rightSpan = right.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Divide(leftSpan, rightSpan);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);

        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new DivideBackward<T>(left, right);
        }

        return tensor;
    }

    /// <summary>
    /// Scalar addition.
    /// </summary>
    public static Tensor<T> operator +(Tensor<T> tensor, T scalar)
    {
        var scalarTensor = Tensor.Full(tensor.Shape, scalar);
        return tensor + scalarTensor;
    }

    /// <summary>
    /// Scalar addition (commutative).
    /// </summary>
    public static Tensor<T> operator +(T scalar, Tensor<T> tensor)
    {
        return tensor + scalar;
    }

    /// <summary>
    /// Scalar subtraction.
    /// </summary>
    public static Tensor<T> operator -(Tensor<T> tensor, T scalar)
    {
        var scalarTensor = Tensor.Full(tensor.Shape, scalar);
        return tensor - scalarTensor;
    }

    /// <summary>
    /// Scalar multiplication.
    /// </summary>
    public static Tensor<T> operator *(Tensor<T> tensor, T scalar)
    {
        var scalarTensor = Tensor.Full(tensor.Shape, scalar);
        return tensor * scalarTensor;
    }

    /// <summary>
    /// Scalar multiplication (commutative).
    /// </summary>
    public static Tensor<T> operator *(T scalar, Tensor<T> tensor)
    {
        return tensor * scalar;
    }

    /// <summary>
    /// Scalar division.
    /// </summary>
    public static Tensor<T> operator /(Tensor<T> tensor, T scalar)
    {
        var scalarTensor = Tensor.Full(tensor.Shape, scalar);
        return tensor / scalarTensor;
    }

    /// <summary>
    /// Unary negation.
    /// </summary>
    public static Tensor<T> operator -(Tensor<T> tensor)
    {
        var span = tensor.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Negate(span);
        var newTensor = new Tensor<T>(result, tensor.Device, tensor.RequiresGrad);

        if (newTensor.RequiresGrad)
        {
            newTensor.GradFn = new NegateBackward<T>(tensor);
        }

        return newTensor;
    }
}
