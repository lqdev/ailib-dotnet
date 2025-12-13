using System.Numerics;
using System.Numerics.Tensors;

namespace AILib;

/// <summary>
/// Factory methods for creating tensors.
/// </summary>
public static class Tensor
{
    /// <summary>
    /// Creates a tensor from an array of data with the specified shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="data">The array of data.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new tensor.</returns>
    public static Tensor<T> FromArray<T>(T[] data, ReadOnlySpan<nint> shape, bool requiresGrad = false)
        where T : unmanaged, INumber<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.Create(data, shape);
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor filled with zeros.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new tensor filled with zeros.</returns>
    public static Tensor<T> Zeros<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false)
        where T : unmanaged, INumber<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.CreateFromShape<T>(shape, false);
        // CreateFromShape already initializes to zero
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new tensor filled with ones.</returns>
    public static Tensor<T> Ones<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false)
        where T : unmanaged, INumber<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.CreateFromShape<T>(shape, false);
        tensor.Fill(T.One);
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor filled with a specific value.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="value">The value to fill the tensor with.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new tensor filled with the specified value.</returns>
    public static Tensor<T> Full<T>(ReadOnlySpan<nint> shape, T value, bool requiresGrad = false)
        where T : unmanaged, INumber<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.CreateFromShape<T>(shape, false);
        tensor.Fill(value);
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor with random values from a normal distribution N(0, 1).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements (must support floating-point).</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new tensor with random normal values.</returns>
    public static Tensor<T> Randn<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.CreateFromShape<T>(shape, false);
        
        // Use the built-in Gaussian distribution fill
        var tensorSpan = tensor.AsTensorSpan();
        System.Numerics.Tensors.Tensor.FillGaussianNormalDistribution(tensorSpan, Random.Shared);

        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor with random values from a uniform distribution [0, 1).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements (must support floating-point).</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new tensor with random uniform values.</returns>
    public static Tensor<T> Rand<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.CreateFromShape<T>(shape, false);
        
        // Use the built-in uniform distribution fill
        var tensorSpan = tensor.AsTensorSpan();
        System.Numerics.Tensors.Tensor.FillUniformDistribution(tensorSpan, Random.Shared);

        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor from an existing .NET Tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The existing .NET tensor.</param>
    /// <param name="requiresGrad">Whether gradient computation is required.</param>
    /// <returns>A new AILib tensor wrapping the .NET tensor.</returns>
    public static Tensor<T> FromTensor<T>(System.Numerics.Tensors.Tensor<T> tensor, bool requiresGrad = false)
        where T : unmanaged, INumber<T>
    {
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
}
