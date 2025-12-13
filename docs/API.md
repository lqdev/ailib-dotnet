# AILib API Reference

## Namespaces

- `AILib` - Core tensor types and factory methods
- `AILib.Autograd` - Automatic differentiation engine
- `AILib.Autograd.Operations` - Gradient computation functions (internal)

## AILib Namespace

### Tensor<T> Class

The main tensor class that wraps `System.Numerics.Tensors.Tensor<T>` and provides gradient tracking.

#### Type Parameters

- `T` - The numeric type of tensor elements (must be `unmanaged` and implement `INumber<T>`)

#### Properties

```csharp
public System.Numerics.Tensors.Tensor<T> Data { get; }
```
The underlying .NET tensor data.

```csharp
public Device Device { get; }
```
The device where this tensor resides (currently always CPU).

```csharp
public ReadOnlySpan<nint> Shape { get; }
```
The shape of the tensor as a read-only span.

```csharp
public int Rank { get; }
```
The number of dimensions in the tensor.

```csharp
public nint ElementCount { get; }
```
The total number of elements in the tensor.

```csharp
public bool RequiresGrad { get; set; }
```
Whether this tensor requires gradient computation.

```csharp
public Tensor<T>? Grad { get; internal set; }
```
The accumulated gradient for this tensor (null if no backward pass has been performed).

#### Methods

```csharp
public T Item()
```
Gets the scalar value of a single-element tensor. Throws `InvalidOperationException` if the tensor has more than one element.

```csharp
public void ZeroGrad()
```
Clears the accumulated gradient.

```csharp
public void Backward()
```
Performs backpropagation from this tensor. The tensor must be scalar and require gradients.

```csharp
public void Dispose()
```
Disposes the tensor and its gradient.

#### Operators

```csharp
public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
public static Tensor<T> operator +(Tensor<T> tensor, T scalar)
public static Tensor<T> operator +(T scalar, Tensor<T> tensor)
```
Element-wise addition.

```csharp
public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
public static Tensor<T> operator -(Tensor<T> tensor, T scalar)
public static Tensor<T> operator -(Tensor<T> tensor)
```
Element-wise subtraction and unary negation.

```csharp
public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
public static Tensor<T> operator *(Tensor<T> tensor, T scalar)
public static Tensor<T> operator *(T scalar, Tensor<T> tensor)
```
Element-wise multiplication.

```csharp
public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
public static Tensor<T> operator /(Tensor<T> tensor, T scalar)
```
Element-wise division.

### Tensor Static Class

Factory methods for creating tensors.

#### Methods

```csharp
public static Tensor<T> FromArray<T>(
    T[] data, 
    ReadOnlySpan<nint> shape, 
    bool requiresGrad = false)
    where T : unmanaged, INumber<T>
```
Creates a tensor from an array with the specified shape.

**Parameters:**
- `data` - The array of data
- `shape` - The shape of the tensor
- `requiresGrad` - Whether gradient computation is required

**Returns:** A new tensor

**Example:**
```csharp
var data = new float[] { 1, 2, 3, 4, 5, 6 };
var tensor = Tensor.FromArray(data, [2, 3]);
```

---

```csharp
public static Tensor<T> Zeros<T>(
    ReadOnlySpan<nint> shape, 
    bool requiresGrad = false)
    where T : unmanaged, INumber<T>
```
Creates a tensor filled with zeros.

**Example:**
```csharp
var zeros = Tensor.Zeros<float>([3, 3]);
```

---

```csharp
public static Tensor<T> Ones<T>(
    ReadOnlySpan<nint> shape, 
    bool requiresGrad = false)
    where T : unmanaged, INumber<T>
```
Creates a tensor filled with ones.

**Example:**
```csharp
var ones = Tensor.Ones<float>([2, 4]);
```

---

```csharp
public static Tensor<T> Full<T>(
    ReadOnlySpan<nint> shape, 
    T value, 
    bool requiresGrad = false)
    where T : unmanaged, INumber<T>
```
Creates a tensor filled with a specific value.

**Example:**
```csharp
var fives = Tensor.Full<float>([2, 2], 5.0f);
```

---

```csharp
public static Tensor<T> Randn<T>(
    ReadOnlySpan<nint> shape, 
    bool requiresGrad = false)
    where T : unmanaged, INumber<T>, IFloatingPoint<T>
```
Creates a tensor with random values from a normal distribution N(0, 1).

**Example:**
```csharp
var normal = Tensor.Randn<float>([100, 50]);
```

---

```csharp
public static Tensor<T> Rand<T>(
    ReadOnlySpan<nint> shape, 
    bool requiresGrad = false)
    where T : unmanaged, INumber<T>, IFloatingPoint<T>
```
Creates a tensor with random values from a uniform distribution [0, 1).

**Example:**
```csharp
var uniform = Tensor.Rand<float>([10, 10]);
```

### TensorOperations Extension Methods

Extension methods for tensor operations.

```csharp
public static Tensor<T> Reshape<T>(this Tensor<T> tensor, ReadOnlySpan<nint> newShape)
```
Reshapes the tensor to a new shape. The total number of elements must remain the same.

---

```csharp
public static Tensor<T> Flatten<T>(this Tensor<T> tensor)
```
Flattens the tensor into a 1D tensor.

---

```csharp
public static Tensor<T> Transpose<T>(this Tensor<T> tensor)
```
Transposes a 2D tensor. Throws `InvalidOperationException` if the tensor is not 2D.

---

```csharp
public static Tensor<T> Permute<T>(this Tensor<T> tensor, ReadOnlySpan<int> dimensions)
```
Permutes the dimensions of the tensor.

---

```csharp
public static Tensor<T> View<T>(this Tensor<T> tensor, ReadOnlySpan<nint> newShape)
```
Creates a view of the tensor with a different shape (alias for Reshape).

---

```csharp
public static Tensor<T> Sum<T>(this Tensor<T> tensor)
```
Computes the sum of all elements in the tensor, returning a scalar tensor.

---

```csharp
public static Tensor<T> Mean<T>(this Tensor<T> tensor)
```
Computes the mean of all elements in the tensor, returning a scalar tensor.

---

```csharp
public static Tensor<T> MatMul<T>(this Tensor<T> left, Tensor<T> right)
```
Performs matrix multiplication of two 2D tensors. Both tensors must be 2D and have compatible shapes.

**Example:**
```csharp
var a = Tensor.Randn<float>([3, 4]);
var b = Tensor.Randn<float>([4, 2]);
var c = a.MatMul(b);  // Shape: [3, 2]
```

### Device Classes

```csharp
public abstract class Device
{
    public static CpuDevice CPU { get; }
    public abstract string Name { get; }
}
```

Currently only CPU device is supported:

```csharp
public sealed class CpuDevice : Device
{
    public override string Name => "cpu";
}
```

## AILib.Autograd Namespace

### GradientEngine Static Class

Engine for computing gradients via reverse-mode automatic differentiation.

```csharp
public static void Backward<T>(Tensor<T> loss) where T : unmanaged, INumber<T>
```
Performs backward pass from a scalar loss tensor.

**Parameters:**
- `loss` - The scalar loss tensor to backpropagate from

**Throws:**
- `InvalidOperationException` - If loss doesn't require gradients or isn't scalar

**Example:**
```csharp
var x = Tensor.Randn<float>([10], requiresGrad: true);
var loss = (x * x).Sum();
loss.Backward();  // or: GradientEngine.Backward(loss);
Console.WriteLine(x.Grad);  // Gradient: 2 * x
```

### GradientFunction<T> Abstract Class

Base class for gradient computation functions in the computation graph.

**Note:** This is an internal implementation detail. Users don't typically interact with this class directly.

## Type Constraints

Most tensor operations require the type parameter `T` to satisfy:
- `unmanaged` - The type must be an unmanaged type
- `INumber<T>` - The type must support numeric operations

Some operations (like `Randn` and `Rand`) additionally require:
- `IFloatingPoint<T>` - The type must be a floating-point number

Commonly used types:
- `float` (Single)
- `double` (Double)
- `Half` (for half-precision)

## Error Handling

Common exceptions thrown by AILib:

### InvalidOperationException

- Calling `Backward()` on a non-scalar tensor
- Calling `Backward()` on a tensor that doesn't require gradients
- Calling `Transpose()` on a non-2D tensor
- Calling `MatMul()` with incompatible tensor shapes

### ArgumentException

- Calling `Reshape()` with an incompatible shape (different total element count)

### Example Error Handling

```csharp
try
{
    var a = Tensor.Randn<float>([3, 4]);
    var b = Tensor.Randn<float>([5, 2]);
    var c = a.MatMul(b);  // Throws: incompatible shapes
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Operation failed: {ex.Message}");
}
```

## Performance Considerations

1. **Memory Management**: Always dispose tensors when done, especially in tight loops:
   ```csharp
   using (var temp = a * b)
   {
       // Use temp
   }  // Automatically disposed
   ```

2. **Gradient Accumulation**: Remember to zero gradients between training steps:
   ```csharp
   for (int i = 0; i < iterations; i++)
   {
       var loss = ComputeLoss();
       loss.Backward();
       UpdateParameters();
       parameters.ZeroGrad();  // Important!
   }
   ```

3. **In-place Operations**: Currently not supported; all operations create new tensors.

## Thread Safety

Tensors are **not** thread-safe. Do not share tensors across threads without proper synchronization.

## Version

Current version: 0.1.0 (Development)

This API is subject to change as the library evolves.
