using System.Numerics;
using System.Numerics.Tensors;

namespace AILib;

/// <summary>
/// Extension methods for activation functions.
/// </summary>
public static class ActivationExtensions
{
    /// <summary>
    /// Rectified Linear Unit: max(0, x)
    /// </summary>
    public static Tensor<T> ReLU<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IComparisonOperators<T, T, bool>
    {
        var zeros = Tensor.Zeros<T>(input.Shape);
        return input.Max(zeros);
    }

    /// <summary>
    /// Maximum of two tensors element-wise.
    /// </summary>
    public static Tensor<T> Max<T>(this Tensor<T> left, Tensor<T> right)
        where T : unmanaged, INumber<T>, IComparisonOperators<T, T, bool>
    {
        var leftSpan = left.Data.AsReadOnlyTensorSpan();
        var rightSpan = right.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Max(leftSpan, rightSpan);
        return new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);
    }

    /// <summary>
    /// Sigmoid activation: 1 / (1 + exp(-x))
    /// </summary>
    public static Tensor<T> Sigmoid<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        var span = input.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Sigmoid(span);
        return new Tensor<T>(result, input.Device, input.RequiresGrad);
    }

    /// <summary>
    /// Hyperbolic tangent activation.
    /// </summary>
    public static Tensor<T> Tanh<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IHyperbolicFunctions<T>
    {
        var span = input.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Tanh(span);
        return new Tensor<T>(result, input.Device, input.RequiresGrad);
    }

    /// <summary>
    /// GELU activation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// </summary>
    public static Tensor<T> GELU<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IHyperbolicFunctions<T>
    {
        var coefficient = T.CreateChecked(Math.Sqrt(2.0 / Math.PI));
        var c = T.CreateChecked(0.044715);
        var half = T.CreateChecked(0.5);
        var one = T.One;

        var x3 = input * input * input;
        var inner = input + c * x3;
        var scaled = coefficient * inner;
        var tanhResult = scaled.Tanh();
        var result = half * input * (one + tanhResult);

        return result;
    }

    /// <summary>
    /// Softmax activation: exp(x) / sum(exp(x))
    /// Note: Using manual implementation for cross-compatibility
    /// </summary>
    public static Tensor<T> Softmax<T>(this Tensor<T> input, int dim = -1)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        // Manual softmax implementation
        var expValues = input.Exp();
        var sumExp = expValues.Sum();
        return expValues / sumExp;
    }

    /// <summary>
    /// Exponential function.
    /// </summary>
    public static Tensor<T> Exp<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        var span = input.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Exp(span);
        return new Tensor<T>(result, input.Device, input.RequiresGrad);
    }

    /// <summary>
    /// Natural logarithm.
    /// </summary>
    public static Tensor<T> Log<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, ILogarithmicFunctions<T>
    {
        var span = input.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Log(span);
        return new Tensor<T>(result, input.Device, input.RequiresGrad);
    }

    /// <summary>
    /// Power function: x^exponent
    /// </summary>
    public static Tensor<T> Pow<T>(this Tensor<T> input, T exponent)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IPowerFunctions<T>
    {
        var span = input.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Pow(span, exponent);
        return new Tensor<T>(result, input.Device, input.RequiresGrad);
    }

    /// <summary>
    /// Square root.
    /// </summary>
    public static Tensor<T> Sqrt<T>(this Tensor<T> input)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IRootFunctions<T>
    {
        var span = input.Data.AsReadOnlyTensorSpan();
        var result = System.Numerics.Tensors.Tensor.Sqrt(span);
        return new Tensor<T>(result, input.Device, input.RequiresGrad);
    }

    /// <summary>
    /// Clamps tensor values to [min, max] range.
    /// </summary>
    public static Tensor<T> Clamp<T>(this Tensor<T> input, T min, T max)
        where T : unmanaged, INumber<T>, IComparisonOperators<T, T, bool>
    {
        var result = System.Numerics.Tensors.Tensor.CreateFromShape<T>(input.Shape, false);
        var inputEnum = input.Data.GetEnumerator();
        var idx = 0;

        while (inputEnum.MoveNext())
        {
            var value = inputEnum.Current;
            if (value < min) value = min;
            if (value > max) value = max;

            var positions = GetPositions(idx, input.Shape);
            result[positions] = value;
            idx++;
        }

        return Tensor.FromTensor(result, input.RequiresGrad);
    }

    /// <summary>
    /// Log-Softmax: log(softmax(x))
    /// </summary>
    public static Tensor<T> LogSoftmax<T>(this Tensor<T> input, int dim = -1)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        // LogSoftmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
        // For numerical stability: LogSoftmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        
        var softmax = input.Softmax(dim);
        return softmax.Log();
    }

    private static nint[] GetPositions(int linearIndex, ReadOnlySpan<nint> shape)
    {
        var positions = new nint[shape.Length];
        var remaining = linearIndex;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            positions[i] = remaining % shape[i];
            remaining /= (int)shape[i];
        }
        return positions;
    }
}
