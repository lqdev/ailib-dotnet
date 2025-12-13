# Product Requirements Document: AILib

**Version:** 1.0  
**Date:** December 2025  
**Status:** Draft  
**Owner:** Dr. Doom

---

## Executive Summary

AILib is a minimalist, PyTorch-like machine learning framework for .NET that leverages the native `System.Numerics.Tensors` API for CPU-based tensor operations. Inspired by HuggingFace's Candle (Rust) design philosophy, this library provides automatic differentiation, neural network primitives, and model training capabilities while maintaining .NET idioms and type safety.

**Key Value Propositions:**
- **Native .NET Integration**: First-class citizen in the .NET ecosystem using official tensor APIs
- **Type Safety**: Leverage .NET's strong typing and generic constraints
- **Developer Experience**: PyTorch-like ergonomics familiar to ML practitioners
- **Performance**: CPU-optimized using SIMD, span-based operations, and zero-copy techniques
- **Simplicity**: CPU-only scope eliminates CUDA complexity and installation friction

---

## Goals and Non-Goals

### Goals

1. **Core Tensor Operations**: Provide a rich set of tensor operations built on `System.Numerics.Tensors`
2. **Automatic Differentiation**: Implement reverse-mode automatic differentiation (backpropagation)
3. **Neural Network Primitives**: Deliver common layers (Linear, Conv2d, Embedding, LayerNorm, etc.)
4. **Training Infrastructure**: Include optimizers (SGD, Adam, AdamW) and loss functions
5. **Model Interoperability**: Support loading models from safetensors format
6. **Performance**: Achieve competitive CPU performance through SIMD and efficient memory management
7. **Ergonomics**: Maintain PyTorch-like API surface for ease of adoption

### Non-Goals

1. **GPU Support**: Explicitly out of scope for v1.0 (CPU-only)
2. **Distributed Training**: No multi-node or data parallelism support
3. **Production Serving**: Focus on training and experimentation, not inference optimization
4. **Pre-trained Models**: No bundled model zoo (users bring their own)
5. **Framework Interop**: No direct PyTorch/TensorFlow model conversion (safetensors only)
6. **Mobile/Edge**: Not optimized for resource-constrained environments

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     AILib.Transformers                     │
│  (Attention, Transformer Blocks, GPT/Llama Models)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        AILib.NN                            │
│  (Layers, Activations, Loss Functions, Optimizers)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       AILib.Core                           │
│  (Tensor, Autograd, Device Abstraction, Backend)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              System.Numerics.Tensors (.NET)                 │
│  (Primitive Tensor<T>, TensorPrimitives Operations)         │
└─────────────────────────────────────────────────────────────┘
```

### Core Abstractions

#### 1. Tensor Wrapper

```csharp
/// <summary>
/// Tensor wrapper that tracks gradients and computation graph
/// </summary>
public sealed class Tensor<T> : IDisposable where T : unmanaged, INumber<T>
{
    // Underlying .NET tensor
    internal readonly System.Numerics.Tensors.Tensor<T> Data;
    
    // Gradient tracking
    public Tensor<T>? Grad { get; internal set; }
    public bool RequiresGrad { get; set; }
    
    // Computation graph
    internal GradientFunction? GradFn { get; set; }
    
    // Shape and metadata
    public ReadOnlySpan<nint> Shape => Data.Lengths;
    public int Rank => Data.Rank;
    public nint ElementCount => Data.FlattenedLength;
}
```

#### 2. Automatic Differentiation

```csharp
/// <summary>
/// Base class for gradient computation functions
/// </summary>
public abstract class GradientFunction
{
    protected List<Tensor<T>> Inputs { get; }
    
    /// <summary>
    /// Compute gradients for inputs given output gradient
    /// </summary>
    public abstract Tensor<T>[] Backward(Tensor<T> gradOutput);
}

/// <summary>
/// Manages computation graph and gradient computation
/// </summary>
public class GradientEngine
{
    public static void Backward(Tensor<T> loss)
    {
        // Topological sort of computation graph
        // Reverse-mode automatic differentiation
        // Accumulate gradients in tensor.Grad
    }
}
```

#### 3. Neural Network Module

```csharp
/// <summary>
/// Base class for all neural network layers
/// </summary>
public abstract class Module : IDisposable
{
    private Dictionary<string, Tensor<float>> _parameters = new();
    private Dictionary<string, Module> _submodules = new();
    
    public bool Training { get; set; } = true;
    
    public abstract Tensor<float> Forward(Tensor<float> input);
    
    public IEnumerable<Tensor<float>> Parameters() 
        => _parameters.Values.Concat(_submodules.Values.SelectMany(m => m.Parameters()));
        
    public void Train() => Training = true;
    public void Eval() => Training = false;
}
```

#### 4. Device Abstraction

```csharp
/// <summary>
/// Represents compute device (CPU-only for v1.0)
/// </summary>
public abstract class Device
{
    public static CpuDevice CPU { get; } = new CpuDevice();
    
    public abstract string Name { get; }
}

public sealed class CpuDevice : Device
{
    public override string Name => "cpu";
}
```

---

## API Design

### Tensor Creation

```csharp
// From arrays
var t1 = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, shape: [2, 2]);

// Zeros/Ones
var t2 = Tensor.Zeros<float>([3, 3]);
var t3 = Tensor.Ones<float>([2, 4]);

// Random initialization
var t4 = Tensor.Randn<float>([100, 50]); // Normal distribution
var t5 = Tensor.Rand<float>([10, 10]);   // Uniform [0, 1)

// From existing tensor
var t6 = Tensor.FromTensor(existingDotNetTensor);

// With gradient tracking
var t7 = Tensor.Zeros<float>([5, 5], requiresGrad: true);
```

### Tensor Operations

```csharp
// Element-wise operations
var sum = a + b;
var product = a * b;
var scaled = a * 2.5f;

// Matrix operations
var matmul = a.MatMul(b);
var transposed = a.Transpose();

// Reshaping
var reshaped = a.Reshape([10, 5]);
var flattened = a.Flatten();
var viewed = a.View([2, 2, 5]);

// Slicing/Indexing
var slice = a[1..3, ..];
var element = a[0, 0];

// Reductions
var mean = a.Mean();
var sum = a.Sum(dim: 1);
var max = a.Max();

// Activations
var relu = a.ReLU();
var sigmoid = a.Sigmoid();
var softmax = a.Softmax(dim: -1);
```

### Autograd

```csharp
// Enable gradient tracking
var x = Tensor.Randn<float>([10, 5], requiresGrad: true);
var w = Tensor.Randn<float>([5, 3], requiresGrad: true);

// Forward pass
var y = x.MatMul(w).ReLU();
var loss = y.Sum();

// Backward pass
loss.Backward();

// Access gradients
var gradX = x.Grad;  // ∂loss/∂x
var gradW = w.Grad;  // ∂loss/∂w

// Zero gradients
x.ZeroGrad();
w.ZeroGrad();
```

### Neural Network Layers

```csharp
// Linear layer
var linear = new Linear(inputSize: 784, outputSize: 128);
var output = linear.Forward(input);

// Convolutional layer
var conv = new Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, stride: 1, padding: 1);
var features = conv.Forward(images);

// Embedding layer
var embedding = new Embedding(vocabSize: 50000, embeddingDim: 512);
var embeddings = embedding.Forward(tokens);

// Layer normalization
var layerNorm = new LayerNorm(normalizedShape: 512);
var normalized = layerNorm.Forward(hidden);

// Dropout
var dropout = new Dropout(p: 0.1);
var dropped = dropout.Forward(activations);
```

### Training Loop

```csharp
// Define model
var model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Dropout(0.2),
    new Linear(256, 10)
);

// Optimizer
var optimizer = new Adam(model.Parameters(), lr: 0.001);

// Loss function
var criterion = new CrossEntropyLoss();

// Training loop
for (int epoch = 0; epoch < 10; epoch++)
{
    model.Train();
    
    foreach (var (inputs, labels) in trainLoader)
    {
        // Zero gradients
        optimizer.ZeroGrad();
        
        // Forward pass
        var outputs = model.Forward(inputs);
        var loss = criterion.Forward(outputs, labels);
        
        // Backward pass
        loss.Backward();
        
        // Update weights
        optimizer.Step();
    }
}
```

### Model Loading (Safetensors)

```csharp
// Load from safetensors file
var varBuilder = VarBuilder.FromSafetensors("model.safetensors");

// Create model with loaded weights
var linear = new Linear(varBuilder, "layer.weight", "layer.bias");

// Or load entire model
var model = new GPT2(varBuilder);
```

---

## Implementation Plan

### Phase 0: Project Setup (Day 1)

**Deliverables:**
- Solution structure with three projects: AILib.Core, AILib.NN, AILib.Transformers
- NuGet package references
- Build configuration (C# 12, .NET 9.0)
- Basic CI/CD setup (optional)

**Tasks:**
1. Create solution: `dotnet new sln -n AILib`
2. Create projects:
   ```bash
   dotnet new classlib -n AILib.Core -f net9.0
   dotnet new classlib -n AILib.NN -f net9.0
   dotnet new classlib -n AILib.Transformers -f net9.0
   dotnet new xunit -n AILib.Core.Tests -f net9.0
   ```
3. Add package references:
   ```xml
   <PackageReference Include="System.Numerics.Tensors" Version="10.0.0" />
   ```
4. Set up project references (NN → Core, Transformers → NN)

---

### Phase 1: Core Tensor Infrastructure (Days 2-4)

**Milestone:** Basic tensor wrapper with shape operations, no autograd yet

#### 1.1 Device Abstraction

**File:** `src/AILib.Core/Device.cs`

```csharp
namespace AILib;

/// <summary>
/// Represents a compute device where tensors reside
/// </summary>
public abstract class Device
{
    public static CpuDevice CPU { get; } = CpuDevice.Instance;
    public abstract string Name { get; }
}

/// <summary>
/// CPU device implementation
/// </summary>
public sealed class CpuDevice : Device
{
    internal static readonly CpuDevice Instance = new();
    private CpuDevice() { }
    
    public override string Name => "cpu";
}
```

**Tests:**
- Verify CPU device singleton pattern
- Test device name property

#### 1.2 Tensor Wrapper

**File:** `src/AILib.Core/Tensor.cs`

```csharp
namespace AILib;

using System.Numerics;
using System.Numerics.Tensors;

/// <summary>
/// Tensor wrapper with gradient tracking support
/// </summary>
public sealed class Tensor<T> : IDisposable where T : unmanaged, INumber<T>
{
    internal readonly System.Numerics.Tensors.Tensor<T> Data;
    
    public Device Device { get; }
    public ReadOnlySpan<nint> Shape => Data.Lengths;
    public int Rank => Data.Rank;
    public nint ElementCount => Data.FlattenedLength;
    
    // Autograd support (implemented in Phase 2)
    public bool RequiresGrad { get; set; }
    public Tensor<T>? Grad { get; internal set; }
    internal GradientFunction<T>? GradFn { get; set; }
    
    internal Tensor(System.Numerics.Tensors.Tensor<T> data, Device device, bool requiresGrad = false)
    {
        Data = data;
        Device = device;
        RequiresGrad = requiresGrad;
    }
    
    public void Dispose()
    {
        // Tensor<T> doesn't implement IDisposable in .NET 9
        // But we provide this for future compatibility
        Grad?.Dispose();
    }
    
    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape.ToArray());
        return $"Tensor<{typeof(T).Name}>[{shapeStr}]";
    }
}
```

**Tests:**
- Tensor creation with various shapes
- Shape property access
- Device assignment
- ToString formatting

#### 1.3 Tensor Factory Methods

**File:** `src/AILib.Core/TensorFactory.cs`

```csharp
namespace AILib;

using System.Numerics;
using System.Numerics.Tensors;

public static class Tensor
{
    /// <summary>
    /// Create tensor from array
    /// </summary>
    public static Tensor<T> FromArray<T>(T[] data, ReadOnlySpan<nint> shape, bool requiresGrad = false) 
        where T : unmanaged, INumber<T>
    {
        var tensor = new System.Numerics.Tensors.Tensor<T>(data, shape);
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
    
    /// <summary>
    /// Create tensor filled with zeros
    /// </summary>
    public static Tensor<T> Zeros<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false) 
        where T : unmanaged, INumber<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.Create<T>(shape, false);
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
    
    /// <summary>
    /// Create tensor filled with ones
    /// </summary>
    public static Tensor<T> Ones<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false) 
        where T : unmanaged, INumber<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.Create<T>(shape, false);
        TensorPrimitives.Fill(tensor.FlattenedValues, T.One);
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
    
    /// <summary>
    /// Create tensor with random normal distribution N(0, 1)
    /// </summary>
    public static Tensor<T> Randn<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false) 
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.Create<T>(shape, false);
        var flat = tensor.FlattenedValues;
        
        var random = Random.Shared;
        for (int i = 0; i < flat.Length; i++)
        {
            // Box-Muller transform for normal distribution
            var u1 = random.NextDouble();
            var u2 = random.NextDouble();
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            flat[i] = T.CreateChecked(randStdNormal);
        }
        
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
    
    /// <summary>
    /// Create tensor with uniform random distribution [0, 1)
    /// </summary>
    public static Tensor<T> Rand<T>(ReadOnlySpan<nint> shape, bool requiresGrad = false) 
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        var tensor = System.Numerics.Tensors.Tensor.Create<T>(shape, false);
        var flat = tensor.FlattenedValues;
        
        var random = Random.Shared;
        for (int i = 0; i < flat.Length; i++)
        {
            flat[i] = T.CreateChecked(random.NextDouble());
        }
        
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
    
    /// <summary>
    /// Create tensor from existing .NET Tensor
    /// </summary>
    public static Tensor<T> FromTensor<T>(System.Numerics.Tensors.Tensor<T> tensor, bool requiresGrad = false)
        where T : unmanaged, INumber<T>
    {
        return new Tensor<T>(tensor, Device.CPU, requiresGrad);
    }
}
```

**Tests:**
- FromArray with various shapes
- Zeros/Ones initialization
- Randn statistical properties (mean ≈ 0, std ≈ 1)
- Rand range validation [0, 1)

#### 1.4 Basic Tensor Operations

**File:** `src/AILib.Core/TensorOperations.cs`

```csharp
namespace AILib;

using System.Numerics;
using System.Numerics.Tensors;

public static class TensorOperations
{
    /// <summary>
    /// Reshape tensor to new shape
    /// </summary>
    public static Tensor<T> Reshape<T>(this Tensor<T> tensor, ReadOnlySpan<nint> newShape) 
        where T : unmanaged, INumber<T>
    {
        var reshaped = System.Numerics.Tensors.Tensor.Reshape(tensor.Data, newShape);
        return new Tensor<T>(reshaped, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new ReshapeBackward<T>(tensor) : null
        };
    }
    
    /// <summary>
    /// Flatten tensor to 1D
    /// </summary>
    public static Tensor<T> Flatten<T>(this Tensor<T> tensor) 
        where T : unmanaged, INumber<T>
    {
        var totalElements = tensor.ElementCount;
        return tensor.Reshape([totalElements]);
    }
    
    /// <summary>
    /// Transpose 2D tensor
    /// </summary>
    public static Tensor<T> Transpose<T>(this Tensor<T> tensor) 
        where T : unmanaged, INumber<T>
    {
        if (tensor.Rank != 2)
            throw new InvalidOperationException("Transpose requires 2D tensor");
            
        var transposed = System.Numerics.Tensors.Tensor.Transpose(tensor.Data);
        return new Tensor<T>(transposed, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new TransposeBackward<T>(tensor) : null
        };
    }
    
    /// <summary>
    /// Permute dimensions
    /// </summary>
    public static Tensor<T> Permute<T>(this Tensor<T> tensor, ReadOnlySpan<int> dimensions) 
        where T : unmanaged, INumber<T>
    {
        var permuted = System.Numerics.Tensors.Tensor.PermuteDimensions(tensor.Data, dimensions);
        return new Tensor<T>(permuted, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new PermuteBackward<T>(tensor, dimensions.ToArray()) : null
        };
    }
    
    /// <summary>
    /// Get slice of tensor
    /// </summary>
    public static Tensor<T> Slice<T>(this Tensor<T> tensor, params Range[] ranges) 
        where T : unmanaged, INumber<T>
    {
        var sliced = System.Numerics.Tensors.Tensor.Slice(tensor.Data, ranges);
        return new Tensor<T>(sliced, tensor.Device, tensor.RequiresGrad)
        {
            GradFn = tensor.RequiresGrad ? new SliceBackward<T>(tensor, ranges) : null
        };
    }
}
```

**Tests:**
- Reshape with various compatible shapes
- Reshape error on incompatible sizes
- Flatten 2D/3D tensors
- Transpose 2D tensors
- Permute 3D/4D tensors
- Slice operations

---

### Phase 2: Automatic Differentiation (Days 5-8)

**Milestone:** Reverse-mode autograd with gradient accumulation

#### 2.1 Gradient Function Base

**File:** `src/AILib.Core/Autograd/GradientFunction.cs`

```csharp
namespace AILib.Autograd;

using System.Numerics;

/// <summary>
/// Base class for gradient computation in computation graph
/// </summary>
public abstract class GradientFunction<T> where T : unmanaged, INumber<T>
{
    protected List<Tensor<T>> SavedTensors { get; } = new();
    
    /// <summary>
    /// Compute gradients for inputs given output gradient
    /// Returns gradients in same order as inputs
    /// </summary>
    public abstract Tensor<T>?[] Backward(Tensor<T> gradOutput);
    
    protected void Save(params Tensor<T>[] tensors)
    {
        SavedTensors.AddRange(tensors);
    }
}
```

#### 2.2 Backward Operations

**File:** `src/AILib.Core/Autograd/Operations/AddBackward.cs`

```csharp
namespace AILib.Autograd.Operations;

using System.Numerics;

internal class AddBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;
    
    public AddBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
    }
    
    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For addition: ∂z/∂x = 1, ∂z/∂y = 1
        // Handle broadcasting by summing over broadcast dimensions
        
        Tensor<T>? grad1 = _input1.RequiresGrad 
            ? UnbroadcastGrad(gradOutput, _input1.Shape) 
            : null;
            
        Tensor<T>? grad2 = _input2.RequiresGrad 
            ? UnbroadcastGrad(gradOutput, _input2.Shape) 
            : null;
        
        return new[] { grad1, grad2 };
    }
    
    private Tensor<T> UnbroadcastGrad(Tensor<T> grad, ReadOnlySpan<nint> targetShape)
    {
        // Sum over dimensions that were broadcast
        var result = grad;
        
        // Handle shape differences
        int gradRank = grad.Rank;
        int targetRank = targetShape.Length;
        
        if (gradRank > targetRank)
        {
            // Sum over leading dimensions
            for (int i = 0; i < gradRank - targetRank; i++)
            {
                result = result.Sum(dim: 0, keepDim: false);
            }
        }
        
        // Sum over singleton dimensions
        for (int i = 0; i < targetRank; i++)
        {
            if (targetShape[i] == 1 && result.Shape[i] > 1)
            {
                result = result.Sum(dim: i, keepDim: true);
            }
        }
        
        return result;
    }
}
```

**File:** `src/AILib.Core/Autograd/Operations/MulBackward.cs`

```csharp
namespace AILib.Autograd.Operations;

using System.Numerics;

internal class MulBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;
    
    public MulBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
    }
    
    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For multiplication: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
        
        Tensor<T>? grad1 = _input1.RequiresGrad 
            ? (gradOutput * _input2).UnbroadcastTo(_input1.Shape) 
            : null;
            
        Tensor<T>? grad2 = _input2.RequiresGrad 
            ? (gradOutput * _input1).UnbroadcastTo(_input2.Shape) 
            : null;
        
        return new[] { grad1, grad2 };
    }
}
```

**File:** `src/AILib.Core/Autograd/Operations/MatMulBackward.cs`

```csharp
namespace AILib.Autograd.Operations;

using System.Numerics;

internal class MatMulBackward<T> : GradientFunction<T> where T : unmanaged, INumber<T>
{
    private readonly Tensor<T> _input1;
    private readonly Tensor<T> _input2;
    
    public MatMulBackward(Tensor<T> input1, Tensor<T> input2)
    {
        _input1 = input1;
        _input2 = input2;
    }
    
    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        // For C = A @ B:
        // ∂L/∂A = ∂L/∂C @ B^T
        // ∂L/∂B = A^T @ ∂L/∂C
        
        Tensor<T>? grad1 = _input1.RequiresGrad 
            ? gradOutput.MatMul(_input2.Transpose()) 
            : null;
            
        Tensor<T>? grad2 = _input2.RequiresGrad 
            ? _input1.Transpose().MatMul(gradOutput) 
            : null;
        
        return new[] { grad1, grad2 };
    }
}
```

**File:** `src/AILib.Core/Autograd/Operations/ReLUBackward.cs`

```csharp
namespace AILib.Autograd.Operations;

using System.Numerics;

internal class ReLUBackward<T> : GradientFunction<T> 
    where T : unmanaged, INumber<T>, IComparisonOperators<T, T, bool>
{
    private readonly Tensor<T> _input;
    
    public ReLUBackward(Tensor<T> input)
    {
        _input = input;
    }
    
    public override Tensor<T>?[] Backward(Tensor<T> gradOutput)
    {
        if (!_input.RequiresGrad)
            return new Tensor<T>?[] { null };
        
        // ReLU gradient: grad if input > 0 else 0
        var mask = _input > T.Zero;
        var grad = gradOutput * mask;
        
        return new[] { grad };
    }
}
```

**Additional backward operations to implement:**
- SubBackward
- DivBackward  
- PowBackward
- ExpBackward
- LogBackward
- SumBackward
- MeanBackward
- SoftmaxBackward
- ReshapeBackward
- TransposeBackward
- PermuteBackward
- SliceBackward

#### 2.3 Gradient Engine

**File:** `src/AILib.Core/Autograd/GradientEngine.cs`

```csharp
namespace AILib.Autograd;

using System.Numerics;

public static class GradientEngine
{
    /// <summary>
    /// Perform backward pass from loss tensor
    /// </summary>
    public static void Backward<T>(Tensor<T> loss) where T : unmanaged, INumber<T>
    {
        if (!loss.RequiresGrad)
            throw new InvalidOperationException("Cannot backward on tensor that doesn't require gradients");
            
        if (loss.ElementCount != 1)
            throw new InvalidOperationException("Backward can only be called on scalar tensors");
        
        // Initialize gradient of loss as 1
        loss.Grad = Tensor.Ones<T>(loss.Shape);
        
        // Topological sort
        var topo = new List<Tensor<T>>();
        var visited = new HashSet<Tensor<T>>();
        TopologicalSort(loss, topo, visited);
        
        // Reverse-mode autodiff
        foreach (var tensor in topo.AsEnumerable().Reverse())
        {
            if (tensor.GradFn == null || tensor.Grad == null)
                continue;
                
            var grads = tensor.GradFn.Backward(tensor.Grad);
            
            // Accumulate gradients in parent tensors
            // This is tensor-specific logic that needs access to saved inputs
        }
    }
    
    private static void TopologicalSort<T>(
        Tensor<T> tensor, 
        List<Tensor<T>> topo, 
        HashSet<Tensor<T>> visited) 
        where T : unmanaged, INumber<T>
    {
        if (visited.Contains(tensor))
            return;
            
        visited.Add(tensor);
        
        if (tensor.GradFn != null)
        {
            // Visit all inputs (saved in GradientFunction)
            foreach (var input in tensor.GradFn.SavedTensors)
            {
                TopologicalSort(input, topo, visited);
            }
        }
        
        topo.Add(tensor);
    }
}
```

#### 2.4 Operator Overloads with Autograd

**File:** `src/AILib.Core/TensorOperators.cs`

```csharp
namespace AILib;

using System.Numerics;
using System.Numerics.Tensors;
using AILib.Autograd.Operations;

public sealed partial class Tensor<T>
{
    public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
    {
        var result = System.Numerics.Tensors.Tensor.Add(left.Data, right.Data);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);
        
        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new AddBackward<T>(left, right);
        }
        
        return tensor;
    }
    
    public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
    {
        var result = System.Numerics.Tensors.Tensor.Multiply(left.Data, right.Data);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);
        
        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new MulBackward<T>(left, right);
        }
        
        return tensor;
    }
    
    public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
    {
        var result = System.Numerics.Tensors.Tensor.Subtract(left.Data, right.Data);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);
        
        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new SubBackward<T>(left, right);
        }
        
        return tensor;
    }
    
    public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
    {
        var result = System.Numerics.Tensors.Tensor.Divide(left.Data, right.Data);
        var tensor = new Tensor<T>(result, left.Device, left.RequiresGrad || right.RequiresGrad);
        
        if (tensor.RequiresGrad)
        {
            tensor.GradFn = new DivBackward<T>(left, right);
        }
        
        return tensor;
    }
    
    // Scalar operations
    public static Tensor<T> operator +(Tensor<T> tensor, T scalar) => tensor + Tensor.Full(tensor.Shape, scalar);
    public static Tensor<T> operator *(Tensor<T> tensor, T scalar) => tensor * Tensor.Full(tensor.Shape, scalar);
    public static Tensor<T> operator -(Tensor<T> tensor, T scalar) => tensor - Tensor.Full(tensor.Shape, scalar);
    public static Tensor<T> operator /(Tensor<T> tensor, T scalar) => tensor / Tensor.Full(tensor.Shape, scalar);
}
```

**Tests:**
- Operator overload correctness
- Gradient computation for each operation
- Gradient accumulation
- Broadcasting behavior
- Chained operations (e.g., `(a + b) * c`)

---

### Phase 3: Neural Network Layers (Days 9-12)

**Milestone:** Common NN layers with forward and backward passes

#### 3.1 Module Base Class

**File:** `src/AILib.NN/Module.cs`

```csharp
namespace AILib.NN;

public abstract class Module : IDisposable
{
    private readonly Dictionary<string, Tensor<float>> _parameters = new();
    private readonly Dictionary<string, Module> _submodules = new();
    private readonly Dictionary<string, Tensor<float>> _buffers = new();
    
    public bool Training { get; set; } = true;
    
    /// <summary>
    /// Forward pass through the module
    /// </summary>
    public abstract Tensor<float> Forward(Tensor<float> input);
    
    /// <summary>
    /// Register a parameter (trainable tensor)
    /// </summary>
    protected void RegisterParameter(string name, Tensor<float> tensor)
    {
        tensor.RequiresGrad = true;
        _parameters[name] = tensor;
    }
    
    /// <summary>
    /// Register a buffer (non-trainable tensor)
    /// </summary>
    protected void RegisterBuffer(string name, Tensor<float> tensor)
    {
        _buffers[name] = tensor;
    }
    
    /// <summary>
    /// Register a submodule
    /// </summary>
    protected void RegisterModule(string name, Module module)
    {
        _submodules[name] = module;
    }
    
    /// <summary>
    /// Get all parameters including from submodules
    /// </summary>
    public IEnumerable<Tensor<float>> Parameters(bool recurse = true)
    {
        foreach (var param in _parameters.Values)
            yield return param;
            
        if (recurse)
        {
            foreach (var module in _submodules.Values)
            {
                foreach (var param in module.Parameters(recurse: true))
                    yield return param;
            }
        }
    }
    
    /// <summary>
    /// Set module to training mode
    /// </summary>
    public virtual void Train()
    {
        Training = true;
        foreach (var module in _submodules.Values)
            module.Train();
    }
    
    /// <summary>
    /// Set module to evaluation mode
    /// </summary>
    public virtual void Eval()
    {
        Training = false;
        foreach (var module in _submodules.Values)
            module.Eval();
    }
    
    /// <summary>
    /// Zero gradients of all parameters
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var param in Parameters())
        {
            param.ZeroGrad();
        }
    }
    
    public virtual void Dispose()
    {
        foreach (var param in _parameters.Values)
            param.Dispose();
            
        foreach (var buffer in _buffers.Values)
            buffer.Dispose();
            
        foreach (var module in _submodules.Values)
            module.Dispose();
    }
}
```

#### 3.2 Linear Layer

**File:** `src/AILib.NN/Layers/Linear.cs`

```csharp
namespace AILib.NN.Layers;

using AILib.NN.Initialization;

/// <summary>
/// Fully connected linear layer: y = xW^T + b
/// </summary>
public class Linear : Module
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float>? _bias;
    private readonly int _inFeatures;
    private readonly int _outFeatures;
    
    public Linear(int inFeatures, int outFeatures, bool bias = true)
    {
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;
        
        // Initialize weight with Kaiming uniform
        _weight = Initializers.KaimingUniform([outFeatures, inFeatures], fanIn: inFeatures);
        RegisterParameter("weight", _weight);
        
        if (bias)
        {
            // Initialize bias with uniform [-bound, bound] where bound = 1/sqrt(fan_in)
            var bound = 1.0f / MathF.Sqrt(inFeatures);
            _bias = Tensor.Rand<float>([outFeatures]) * (2 * bound) - bound;
            RegisterParameter("bias", _bias);
        }
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        // input: [batch, in_features] or [*, in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features] or [*, out_features]
        
        var output = input.MatMul(_weight.Transpose());
        
        if (_bias != null)
        {
            output = output + _bias;
        }
        
        return output;
    }
}
```

#### 3.3 Conv2d Layer

**File:** `src/AILib.NN/Layers/Conv2d.cs`

```csharp
namespace AILib.NN.Layers;

using AILib.NN.Initialization;

/// <summary>
/// 2D convolutional layer
/// </summary>
public class Conv2d : Module
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float>? _bias;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    
    public Conv2d(
        int inChannels, 
        int outChannels, 
        int kernelSize, 
        int stride = 1, 
        int padding = 0,
        bool bias = true)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        
        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        var fanIn = inChannels * kernelSize * kernelSize;
        _weight = Initializers.KaimingUniform(
            [outChannels, inChannels, kernelSize, kernelSize], 
            fanIn: fanIn);
        RegisterParameter("weight", _weight);
        
        if (bias)
        {
            var bound = 1.0f / MathF.Sqrt(fanIn);
            _bias = Tensor.Rand<float>([outChannels]) * (2 * bound) - bound;
            RegisterParameter("bias", _bias);
        }
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        // input: [batch, in_channels, height, width]
        // weight: [out_channels, in_channels, kernel_h, kernel_w]
        // output: [batch, out_channels, out_height, out_width]
        
        var output = input.Conv2d(_weight, _stride, _padding);
        
        if (_bias != null)
        {
            // Reshape bias to [1, out_channels, 1, 1] for broadcasting
            var biasReshaped = _bias.Reshape([1, _outChannels, 1, 1]);
            output = output + biasReshaped;
        }
        
        return output;
    }
}
```

#### 3.4 Embedding Layer

**File:** `src/AILib.NN/Layers/Embedding.cs`

```csharp
namespace AILib.NN.Layers;

/// <summary>
/// Embedding layer for discrete tokens
/// </summary>
public class Embedding : Module
{
    private readonly Tensor<float> _weight;
    private readonly int _vocabSize;
    private readonly int _embeddingDim;
    
    public Embedding(int vocabSize, int embeddingDim)
    {
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;
        
        // Initialize embeddings with N(0, 1)
        _weight = Tensor.Randn<float>([vocabSize, embeddingDim]);
        RegisterParameter("weight", _weight);
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        // This is a placeholder - actual implementation needs index selection
        // input: [batch, seq_len] of token indices
        // output: [batch, seq_len, embedding_dim]
        
        throw new NotImplementedException("Embedding requires index selection operation");
    }
    
    public Tensor<float> Forward(Tensor<int> indices)
    {
        // Proper typed version that takes integer indices
        // Use gather operation to select embeddings
        return _weight.Gather(indices, dim: 0);
    }
}
```

#### 3.5 LayerNorm

**File:** `src/AILib.NN/Layers/LayerNorm.cs`

```csharp
namespace AILib.NN.Layers;

/// <summary>
/// Layer normalization
/// </summary>
public class LayerNorm : Module
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float> _bias;
    private readonly int _normalizedShape;
    private readonly float _eps;
    
    public LayerNorm(int normalizedShape, float eps = 1e-5f)
    {
        _normalizedShape = normalizedShape;
        _eps = eps;
        
        _weight = Tensor.Ones<float>([normalizedShape]);
        _bias = Tensor.Zeros<float>([normalizedShape]);
        
        RegisterParameter("weight", _weight);
        RegisterParameter("bias", _bias);
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        // Normalize over last dimension
        // input: [*, normalized_shape]
        
        var mean = input.Mean(dim: -1, keepDim: true);
        var variance = ((input - mean).Pow(2)).Mean(dim: -1, keepDim: true);
        var normalized = (input - mean) / (variance + _eps).Sqrt();
        
        return normalized * _weight + _bias;
    }
}
```

#### 3.6 Dropout

**File:** `src/AILib.NN/Layers/Dropout.cs`

```csharp
namespace AILib.NN.Layers;

/// <summary>
/// Dropout regularization
/// </summary>
public class Dropout : Module
{
    private readonly float _p;
    
    public Dropout(float p = 0.5f)
    {
        if (p < 0 || p > 1)
            throw new ArgumentException("Dropout probability must be in [0, 1]");
            
        _p = p;
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        if (!Training || _p == 0)
            return input;
            
        // Generate random mask
        var mask = Tensor.Rand<float>(input.Shape) > _p;
        
        // Scale by 1/(1-p) to maintain expected value
        var scale = 1.0f / (1.0f - _p);
        
        return input * mask * scale;
    }
}
```

**Additional layers to implement:**
- BatchNorm1d, BatchNorm2d
- MaxPool2d, AvgPool2d
- RNN, LSTM, GRU (optional for v1.0)

#### 3.7 Activation Functions

**File:** `src/AILib.NN/Activations/Activations.cs`

```csharp
namespace AILib.NN.Activations;

public static class Activations
{
    public static Tensor<float> ReLU(this Tensor<float> input)
    {
        return input.Max(Tensor.Zeros<float>(input.Shape));
    }
    
    public static Tensor<float> GELU(this Tensor<float> input)
    {
        // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        var coefficient = MathF.Sqrt(2.0f / MathF.PI);
        var inner = coefficient * (input + 0.044715f * input.Pow(3));
        return 0.5f * input * (1.0f + inner.Tanh());
    }
    
    public static Tensor<float> Sigmoid(this Tensor<float> input)
    {
        return 1.0f / (1.0f + (-input).Exp());
    }
    
    public static Tensor<float> Tanh(this Tensor<float> input)
    {
        return input.Tanh(); // Use built-in
    }
    
    public static Tensor<float> Softmax(this Tensor<float> input, int dim = -1)
    {
        // Numerically stable softmax
        var maxVal = input.Max(dim: dim, keepDim: true);
        var exp = (input - maxVal).Exp();
        var sum = exp.Sum(dim: dim, keepDim: true);
        return exp / sum;
    }
}
```

**Tests for Phase 3:**
- Module parameter registration
- Linear layer forward pass
- Conv2d output shapes
- LayerNorm normalization properties
- Dropout mask generation and scaling
- Activation function outputs
- Gradient flow through layers

---

### Phase 4: Training Infrastructure (Days 13-15)

**Milestone:** Optimizers and loss functions for training

#### 4.1 Optimizer Base

**File:** `src/AILib.NN/Optimizers/Optimizer.cs`

```csharp
namespace AILib.NN.Optimizers;

/// <summary>
/// Base class for all optimizers
/// </summary>
public abstract class Optimizer
{
    protected readonly List<Tensor<float>> Parameters;
    protected int Step { get; set; }
    
    protected Optimizer(IEnumerable<Tensor<float>> parameters)
    {
        Parameters = parameters.ToList();
        Step = 0;
    }
    
    /// <summary>
    /// Zero all parameter gradients
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var param in Parameters)
        {
            param.ZeroGrad();
        }
    }
    
    /// <summary>
    /// Perform single optimization step
    /// </summary>
    public void StepOptimizer()
    {
        Step++;
        UpdateParameters();
    }
    
    protected abstract void UpdateParameters();
}
```

#### 4.2 SGD Optimizer

**File:** `src/AILib.NN/Optimizers/SGD.cs`

```csharp
namespace AILib.NN.Optimizers;

/// <summary>
/// Stochastic Gradient Descent optimizer
/// </summary>
public class SGD : Optimizer
{
    private readonly float _lr;
    private readonly float _momentum;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _velocities;
    
    public SGD(
        IEnumerable<Tensor<float>> parameters, 
        float lr = 0.01f, 
        float momentum = 0.0f,
        float weightDecay = 0.0f)
        : base(parameters)
    {
        _lr = lr;
        _momentum = momentum;
        _weightDecay = weightDecay;
        _velocities = new Dictionary<Tensor<float>, Tensor<float>>();
        
        if (momentum > 0)
        {
            foreach (var param in Parameters)
            {
                _velocities[param] = Tensor.Zeros<float>(param.Shape);
            }
        }
    }
    
    protected override void UpdateParameters()
    {
        foreach (var param in Parameters)
        {
            if (param.Grad == null)
                continue;
                
            var grad = param.Grad;
            
            // Apply weight decay
            if (_weightDecay > 0)
            {
                grad = grad + _weightDecay * param;
            }
            
            // Apply momentum
            if (_momentum > 0)
            {
                var velocity = _velocities[param];
                velocity = _momentum * velocity + grad;
                _velocities[param] = velocity;
                grad = velocity;
            }
            
            // Update parameters: θ = θ - lr * grad
            param.Data = param - _lr * grad;
        }
    }
}
```

#### 4.3 Adam Optimizer

**File:** `src/AILib.NN/Optimizers/Adam.cs`

```csharp
namespace AILib.NN.Optimizers;

/// <summary>
/// Adam optimizer
/// </summary>
public class Adam : Optimizer
{
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _m; // First moment
    private readonly Dictionary<Tensor<float>, Tensor<float>> _v; // Second moment
    
    public Adam(
        IEnumerable<Tensor<float>> parameters,
        float lr = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.0f)
        : base(parameters)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
        
        _m = new Dictionary<Tensor<float>, Tensor<float>>();
        _v = new Dictionary<Tensor<float>, Tensor<float>>();
        
        foreach (var param in Parameters)
        {
            _m[param] = Tensor.Zeros<float>(param.Shape);
            _v[param] = Tensor.Zeros<float>(param.Shape);
        }
    }
    
    protected override void UpdateParameters()
    {
        foreach (var param in Parameters)
        {
            if (param.Grad == null)
                continue;
                
            var grad = param.Grad;
            
            // Apply weight decay (AdamW variant if needed)
            if (_weightDecay > 0)
            {
                grad = grad + _weightDecay * param;
            }
            
            // Update biased first moment estimate
            var m = _m[param];
            m = _beta1 * m + (1 - _beta1) * grad;
            _m[param] = m;
            
            // Update biased second moment estimate
            var v = _v[param];
            v = _beta2 * v + (1 - _beta2) * grad.Pow(2);
            _v[param] = v;
            
            // Bias correction
            var mHat = m / (1 - MathF.Pow(_beta1, Step));
            var vHat = v / (1 - MathF.Pow(_beta2, Step));
            
            // Update parameters
            param.Data = param - _lr * mHat / (vHat.Sqrt() + _eps);
        }
    }
}
```

#### 4.4 AdamW Optimizer

**File:** `src/AILib.NN/Optimizers/AdamW.cs`

```csharp
namespace AILib.NN.Optimizers;

/// <summary>
/// AdamW optimizer (Adam with decoupled weight decay)
/// </summary>
public class AdamW : Optimizer
{
    private readonly float _lr;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _m;
    private readonly Dictionary<Tensor<float>, Tensor<float>> _v;
    
    public AdamW(
        IEnumerable<Tensor<float>> parameters,
        float lr = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0.01f)
        : base(parameters)
    {
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
        
        _m = new Dictionary<Tensor<float>, Tensor<float>>();
        _v = new Dictionary<Tensor<float>, Tensor<float>>();
        
        foreach (var param in Parameters)
        {
            _m[param] = Tensor.Zeros<float>(param.Shape);
            _v[param] = Tensor.Zeros<float>(param.Shape);
        }
    }
    
    protected override void UpdateParameters()
    {
        foreach (var param in Parameters)
        {
            if (param.Grad == null)
                continue;
                
            var grad = param.Grad;
            
            // Update biased first moment estimate
            var m = _m[param];
            m = _beta1 * m + (1 - _beta1) * grad;
            _m[param] = m;
            
            // Update biased second moment estimate
            var v = _v[param];
            v = _beta2 * v + (1 - _beta2) * grad.Pow(2);
            _v[param] = v;
            
            // Bias correction
            var mHat = m / (1 - MathF.Pow(_beta1, Step));
            var vHat = v / (1 - MathF.Pow(_beta2, Step));
            
            // Update with decoupled weight decay
            param.Data = param * (1 - _lr * _weightDecay) - _lr * mHat / (vHat.Sqrt() + _eps);
        }
    }
}
```

#### 4.5 Loss Functions

**File:** `src/AILib.NN/Loss/MSELoss.cs`

```csharp
namespace AILib.NN.Loss;

/// <summary>
/// Mean Squared Error loss
/// </summary>
public class MSELoss
{
    public Tensor<float> Forward(Tensor<float> predictions, Tensor<float> targets)
    {
        var diff = predictions - targets;
        var squared = diff.Pow(2);
        return squared.Mean();
    }
}
```

**File:** `src/AILib.NN/Loss/CrossEntropyLoss.cs`

```csharp
namespace AILib.NN.Loss;

using AILib.NN.Activations;

/// <summary>
/// Cross-entropy loss for classification
/// </summary>
public class CrossEntropyLoss
{
    public Tensor<float> Forward(Tensor<float> logits, Tensor<int> targets)
    {
        // logits: [batch, num_classes]
        // targets: [batch] of class indices
        
        // Compute log probabilities (numerically stable)
        var logProbs = logits.LogSoftmax(dim: -1);
        
        // Gather log probabilities of target classes
        var targetLogProbs = logProbs.GatherByIndex(targets);
        
        // Return negative mean
        return -targetLogProbs.Mean();
    }
}
```

**File:** `src/AILib.NN/Loss/BCELoss.cs`

```csharp
namespace AILib.NN.Loss;

/// <summary>
/// Binary Cross-Entropy loss
/// </summary>
public class BCELoss
{
    private readonly float _eps;
    
    public BCELoss(float eps = 1e-7f)
    {
        _eps = eps;
    }
    
    public Tensor<float> Forward(Tensor<float> predictions, Tensor<float> targets)
    {
        // Clamp predictions to avoid log(0)
        var clampedPreds = predictions.Clamp(_eps, 1 - _eps);
        
        var loss = -(targets * clampedPreds.Log() + (1 - targets) * (1 - clampedPreds).Log());
        return loss.Mean();
    }
}
```

**Tests for Phase 4:**
- SGD parameter updates
- Adam/AdamW convergence
- Weight decay application
- MSE loss computation
- CrossEntropy gradient correctness
- Optimizer state management

---

### Phase 5: Model Loading (Safetensors) (Days 16-18)

**Milestone:** Load pre-trained models from safetensors format

#### 5.1 Safetensors Parser

**File:** `src/AILib.NN/VarBuilder/SafetensorsParser.cs`

```csharp
namespace AILib.NN.VarBuilder;

using System.Text.Json;

/// <summary>
/// Parser for safetensors format
/// Spec: https://github.com/huggingface/safetensors
/// </summary>
public class SafetensorsParser
{
    private readonly Dictionary<string, TensorInfo> _metadata;
    private readonly byte[] _data;
    
    private class TensorInfo
    {
        public string DType { get; set; } = "";
        public long[] Shape { get; set; } = Array.Empty<long>();
        public long[] DataOffsets { get; set; } = Array.Empty<long>();
    }
    
    public SafetensorsParser(string filepath)
    {
        using var file = File.OpenRead(filepath);
        
        // Read header size (8 bytes, little-endian)
        var headerSizeBytes = new byte[8];
        file.Read(headerSizeBytes, 0, 8);
        var headerSize = BitConverter.ToInt64(headerSizeBytes, 0);
        
        // Read header JSON
        var headerBytes = new byte[headerSize];
        file.Read(headerBytes, 0, (int)headerSize);
        var headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);
        
        _metadata = ParseHeader(headerJson);
        
        // Read tensor data
        var dataSize = file.Length - 8 - headerSize;
        _data = new byte[dataSize];
        file.Read(_data, 0, (int)dataSize);
    }
    
    private Dictionary<string, TensorInfo> ParseHeader(string json)
    {
        var doc = JsonDocument.Parse(json);
        var result = new Dictionary<string, TensorInfo>();
        
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
                continue;
                
            var info = new TensorInfo
            {
                DType = prop.Value.GetProperty("dtype").GetString() ?? "",
                Shape = prop.Value.GetProperty("shape").EnumerateArray()
                    .Select(x => x.GetInt64()).ToArray(),
                DataOffsets = prop.Value.GetProperty("data_offsets").EnumerateArray()
                    .Select(x => x.GetInt64()).ToArray()
            };
            
            result[prop.Name] = info;
        }
        
        return result;
    }
    
    public Tensor<float> GetTensor(string name)
    {
        if (!_metadata.TryGetValue(name, out var info))
            throw new KeyException($"Tensor '{name}' not found in safetensors file");
            
        var start = (int)info.DataOffsets[0];
        var end = (int)info.DataOffsets[1];
        var tensorData = _data[start..end];
        
        // Convert to float array based on dtype
        float[] floatData = info.DType switch
        {
            "F32" => ParseF32(tensorData),
            "F16" => ParseF16(tensorData),
            "BF16" => ParseBF16(tensorData),
            _ => throw new NotSupportedException($"DType '{info.DType}' not supported")
        };
        
        var shape = info.Shape.Select(x => (nint)x).ToArray();
        return Tensor.FromArray(floatData, shape);
    }
    
    private float[] ParseF32(byte[] data)
    {
        var floats = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, floats, 0, data.Length);
        return floats;
    }
    
    private float[] ParseF16(byte[] data)
    {
        // Convert FP16 to FP32
        var floats = new float[data.Length / 2];
        for (int i = 0; i < floats.Length; i++)
        {
            var bytes = new[] { data[i * 2], data[i * 2 + 1] };
            floats[i] = HalfToFloat(BitConverter.ToUInt16(bytes, 0));
        }
        return floats;
    }
    
    private float[] ParseBF16(byte[] data)
    {
        // Convert BF16 to FP32
        var floats = new float[data.Length / 2];
        for (int i = 0; i < floats.Length; i++)
        {
            var bf16 = BitConverter.ToUInt16(data, i * 2);
            floats[i] = BFloat16ToFloat(bf16);
        }
        return floats;
    }
    
    private static float HalfToFloat(ushort half)
    {
        // FP16 to FP32 conversion
        // Implementation details omitted for brevity
        return (float)BitConverter.UInt16BitsToHalf(half);
    }
    
    private static float BFloat16ToFloat(ushort bf16)
    {
        // BF16 to FP32: shift left 16 bits
        uint bits = (uint)bf16 << 16;
        return BitConverter.Int32BitsToSingle((int)bits);
    }
}
```

#### 5.2 VarBuilder

**File:** `src/AILib.NN/VarBuilder/VarBuilder.cs`

```csharp
namespace AILib.NN.VarBuilder;

/// <summary>
/// Builder for loading model weights from safetensors
/// </summary>
public class VarBuilder
{
    private readonly SafetensorsParser _parser;
    private readonly string _prefix;
    
    private VarBuilder(SafetensorsParser parser, string prefix = "")
    {
        _parser = parser;
        _prefix = prefix;
    }
    
    public static VarBuilder FromSafetensors(string filepath)
    {
        var parser = new SafetensorsParser(filepath);
        return new VarBuilder(parser);
    }
    
    /// <summary>
    /// Get tensor by name
    /// </summary>
    public Tensor<float> Get(string name)
    {
        var fullName = string.IsNullOrEmpty(_prefix) ? name : $"{_prefix}.{name}";
        return _parser.GetTensor(fullName);
    }
    
    /// <summary>
    /// Create a sub-builder with a prefix
    /// </summary>
    public VarBuilder Prefix(string prefix)
    {
        var newPrefix = string.IsNullOrEmpty(_prefix) ? prefix : $"{_prefix}.{prefix}";
        return new VarBuilder(_parser, newPrefix);
    }
    
    /// <summary>
    /// Load Linear layer weights
    /// </summary>
    public Linear LoadLinear(int inFeatures, int outFeatures, bool bias = true)
    {
        var linear = new Linear(inFeatures, outFeatures, bias);
        
        // Load weight
        var weight = Get("weight");
        linear.SetWeight(weight);
        
        // Load bias if present
        if (bias)
        {
            var biasParam = Get("bias");
            linear.SetBias(biasParam);
        }
        
        return linear;
    }
}
```

**Tests for Phase 5:**
- Parse safetensors header
- Extract tensor data
- FP16/BF16 conversion accuracy
- VarBuilder prefix handling
- Load simple model from safetensors

---

### Phase 6: Examples and Documentation (Days 19-21)

**Milestone:** Working examples and comprehensive docs

#### 6.1 MNIST Example

**File:** `examples/MNIST/Program.cs`

```csharp
using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Define simple CNN for MNIST
public class MNISTModel : Module
{
    private readonly Conv2d _conv1;
    private readonly Conv2d _conv2;
    private readonly Linear _fc1;
    private readonly Linear _fc2;
    private readonly Dropout _dropout;
    
    public MNISTModel()
    {
        _conv1 = new Conv2d(1, 32, 3, padding: 1);
        _conv2 = new Conv2d(32, 64, 3, padding: 1);
        _fc1 = new Linear(64 * 7 * 7, 128);
        _fc2 = new Linear(128, 10);
        _dropout = new Dropout(0.5f);
        
        RegisterModule("conv1", _conv1);
        RegisterModule("conv2", _conv2);
        RegisterModule("fc1", _fc1);
        RegisterModule("fc2", _fc2);
        RegisterModule("dropout", _dropout);
    }
    
    public override Tensor<float> Forward(Tensor<float> x)
    {
        // x: [batch, 1, 28, 28]
        
        x = _conv1.Forward(x).ReLU().MaxPool2d(2);  // [batch, 32, 14, 14]
        x = _conv2.Forward(x).ReLU().MaxPool2d(2);  // [batch, 64, 7, 7]
        x = x.Flatten(startDim: 1);                 // [batch, 64*7*7]
        x = _fc1.Forward(x).ReLU();
        x = _dropout.Forward(x);
        x = _fc2.Forward(x);
        
        return x;
    }
}

class Program
{
    static void Main()
    {
        // Load MNIST data (implementation omitted)
        var (trainImages, trainLabels) = LoadMNIST("train");
        var (testImages, testLabels) = LoadMNIST("test");
        
        // Create model
        var model = new MNISTModel();
        var optimizer = new Adam(model.Parameters(), lr: 0.001f);
        var criterion = new CrossEntropyLoss();
        
        // Training loop
        int epochs = 10;
        int batchSize = 64;
        
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            model.Train();
            float totalLoss = 0;
            
            for (int i = 0; i < trainImages.Length; i += batchSize)
            {
                var batch = GetBatch(trainImages, trainLabels, i, batchSize);
                
                // Zero gradients
                optimizer.ZeroGrad();
                
                // Forward pass
                var outputs = model.Forward(batch.Images);
                var loss = criterion.Forward(outputs, batch.Labels);
                
                // Backward pass
                loss.Backward();
                
                // Update weights
                optimizer.StepOptimizer();
                
                totalLoss += loss.Item();
            }
            
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / (trainImages.Length / batchSize)}");
            
            // Evaluation
            model.Eval();
            float accuracy = Evaluate(model, testImages, testLabels);
            Console.WriteLine($"Test Accuracy: {accuracy * 100:F2}%");
        }
    }
}
```

#### 6.2 Simple Neural Network Example

**File:** `examples/SimpleNN/Program.cs`

```csharp
using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.NN.Optimizers;

// XOR problem
class Program
{
    static void Main()
    {
        // Training data for XOR
        var inputs = Tensor.FromArray(new float[]
        {
            0, 0,
            0, 1,
            1, 0,
            1, 1
        }, [4, 2]);
        
        var targets = Tensor.FromArray(new float[] { 0, 1, 1, 0 }, [4, 1]);
        
        // Simple 2-layer network
        var model = new Sequential(
            new Linear(2, 4),
            new ReLU(),
            new Linear(4, 1),
            new Sigmoid()
        );
        
        var optimizer = new Adam(model.Parameters(), lr: 0.01f);
        var criterion = new MSELoss();
        
        // Training
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            optimizer.ZeroGrad();
            
            var outputs = model.Forward(inputs);
            var loss = criterion.Forward(outputs, targets);
            
            loss.Backward();
            optimizer.StepOptimizer();
            
            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item()}");
            }
        }
        
        // Test
        model.Eval();
        var predictions = model.Forward(inputs);
        Console.WriteLine("Predictions:");
        Console.WriteLine(predictions);
    }
}
```

#### 6.3 Documentation

**File:** `README.md`

```markdown
# AILib

A minimalist, PyTorch-like ML framework for .NET leveraging `System.Numerics.Tensors`.

Inspired by HuggingFace's Candle but built from the ground up as a .NET-native library.

## Features

- 🔢 Tensor operations with automatic differentiation
- 🧠 Neural network layers (Linear, Conv2d, Embedding, LayerNorm, etc.)
- 🎯 Optimizers (SGD, Adam, AdamW)
- 📊 Loss functions (MSE, CrossEntropy, BCE)
- 📦 Model loading from safetensors format
- ⚡ CPU-optimized with SIMD

## Installation

```bash
dotnet add package AILib
```

## Quick Start

```csharp
using AILib;
using AILib.NN;

// Create tensors
var x = Tensor.Randn<float>([100, 10], requiresGrad: true);
var y = Tensor.Randn<float>([100, 1]);

// Define model
var model = new Sequential(
    new Linear(10, 20),
    new ReLU(),
    new Linear(20, 1)
);

// Train
var optimizer = new Adam(model.Parameters(), lr: 0.001f);
var criterion = new MSELoss();

for (int i = 0; i < 100; i++)
{
    optimizer.ZeroGrad();
    var pred = model.Forward(x);
    var loss = criterion.Forward(pred, y);
    loss.Backward();
    optimizer.StepOptimizer();
}
```

## Documentation

See [docs/](docs/) for detailed documentation.

## License

MIT
```

**Additional documentation to create:**
- API reference for all public types
- Tutorial notebooks (if applicable)
- Architecture diagrams
- Performance benchmarks
- Migration guide from PyTorch

---

## Testing Strategy

### Unit Tests

**File:** `tests/AILib.Core.Tests/TensorTests.cs`

```csharp
using Xunit;
using AILib;

public class TensorTests
{
    [Fact]
    public void CreateTensor_FromArray_CorrectShape()
    {
        var data = new float[] { 1, 2, 3, 4 };
        var tensor = Tensor.FromArray(data, [2, 2]);
        
        Assert.Equal(2, tensor.Rank);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(2, tensor.Shape[1]);
    }
    
    [Fact]
    public void Randn_ProducesNormalDistribution()
    {
        var tensor = Tensor.Randn<float>([10000]);
        var mean = tensor.Mean().Item();
        var std = tensor.Std().Item();
        
        Assert.InRange(mean, -0.1f, 0.1f);
        Assert.InRange(std, 0.9f, 1.1f);
    }
    
    [Fact]
    public void MatMul_2DTensors_CorrectShape()
    {
        var a = Tensor.Randn<float>([3, 4]);
        var b = Tensor.Randn<float>([4, 5]);
        var c = a.MatMul(b);
        
        Assert.Equal([3, 5], c.Shape.ToArray());
    }
}
```

### Gradient Tests

**File:** `tests/AILib.Core.Tests/AutogradTests.cs`

```csharp
using Xunit;
using AILib;

public class AutogradTests
{
    [Fact]
    public void Add_Backward_CorrectGradients()
    {
        var a = Tensor.FromArray(new float[] { 2.0f }, [1], requiresGrad: true);
        var b = Tensor.FromArray(new float[] { 3.0f }, [1], requiresGrad: true);
        var c = a + b;
        
        c.Backward();
        
        Assert.Equal(1.0f, a.Grad.Item());
        Assert.Equal(1.0f, b.Grad.Item());
    }
    
    [Fact]
    public void MatMul_Backward_NumericalGradient()
    {
        var a = Tensor.Randn<float>([2, 3], requiresGrad: true);
        var b = Tensor.Randn<float>([3, 2], requiresGrad: true);
        
        var c = a.MatMul(b);
        var loss = c.Sum();
        loss.Backward();
        
        // Verify with numerical gradient
        var eps = 1e-4f;
        var numericalGrad = ComputeNumericalGradient(
            () => a.MatMul(b).Sum(), 
            a, 
            eps);
            
        AssertTensorsClose(a.Grad, numericalGrad, tolerance: 1e-3f);
    }
}
```

### Integration Tests

**File:** `tests/AILib.NN.Tests/TrainingTests.cs`

```csharp
using Xunit;
using AILib;
using AILib.NN;

public class TrainingTests
{
    [Fact]
    public void SimpleRegression_Converges()
    {
        // Linear regression: y = 2x + 1
        var x = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [4, 1]);
        var y = Tensor.FromArray(new float[] { 3, 5, 7, 9 }, [4, 1]);
        
        var model = new Linear(1, 1, bias: true);
        var optimizer = new SGD(model.Parameters(), lr: 0.01f);
        var criterion = new MSELoss();
        
        // Train
        for (int i = 0; i < 1000; i++)
        {
            optimizer.ZeroGrad();
            var pred = model.Forward(x);
            var loss = criterion.Forward(pred, y);
            loss.Backward();
            optimizer.StepOptimizer();
        }
        
        // Check if learned y = 2x + 1
        var finalPred = model.Forward(x);
        var finalLoss = criterion.Forward(finalPred, y).Item();
        
        Assert.True(finalLoss < 0.01f, "Model should converge to low loss");
    }
}
```

---

## Performance Optimization

### SIMD Optimization

Use `TensorPrimitives` for vectorized operations:

```csharp
// Use built-in TensorPrimitives for SIMD
TensorPrimitives.Add(source1, source2, destination);
TensorPrimitives.Multiply(source1, source2, destination);

// For custom operations
if (Vector.IsHardwareAccelerated && length >= Vector<T>.Count)
{
    // Vectorized path
    int simdLength = length - (length % Vector<T>.Count);
    for (int i = 0; i < simdLength; i += Vector<T>.Count)
    {
        var v1 = new Vector<T>(source1, i);
        var v2 = new Vector<T>(source2, i);
        var result = v1 + v2;
        result.CopyTo(destination, i);
    }
    
    // Handle remainder
    for (int i = simdLength; i < length; i++)
    {
        destination[i] = source1[i] + source2[i];
    }
}
```

### Memory Management

- Use `stackalloc` for small temporary allocations
- Leverage `Span<T>` and `Memory<T>` to avoid heap allocations
- Implement object pooling for frequently allocated tensors
- Dispose tensors explicitly in tight loops

### Parallelization

```csharp
// Parallelize batch operations
Parallel.For(0, batchSize, i =>
{
    // Process individual batch item
});
```

---

## Documentation and Resources

### Official .NET Documentation

1. **System.Numerics.Tensors**
   - API Reference: https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors
   - Tensor<T>: https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensor-1
   - TensorPrimitives: https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives

2. **Generic Math (.NET 7+)**
   - INumber<T>: https://learn.microsoft.com/en-us/dotnet/api/system.numerics.inumber-1
   - Generic Math: https://learn.microsoft.com/en-us/dotnet/standard/generics/math

3. **SIMD and Vectorization**
   - Vector<T>: https://learn.microsoft.com/en-us/dotnet/api/system.numerics.vector-1
   - Hardware Intrinsics: https://learn.microsoft.com/en-us/dotnet/api/system.runtime.intrinsics

### Candle (Rust) Resources

1. **Candle GitHub**: https://github.com/huggingface/candle
2. **Candle Documentation**: https://huggingface.github.io/candle/
3. **Candle Examples**: https://github.com/huggingface/candle/tree/main/candle-examples

### ML/DL Concepts

1. **Automatic Differentiation**
   - CS231n Backprop: https://cs231n.github.io/optimization-2/
   - Autograd Explained: https://arxiv.org/abs/1502.05767

2. **Neural Network Architectures**
   - Deep Learning Book: https://www.deeplearningbook.org/
   - PyTorch Tutorials: https://pytorch.org/tutorials/

3. **Safetensors Format**
   - Specification: https://github.com/huggingface/safetensors
   - Rust Implementation: https://github.com/huggingface/safetensors/tree/main/safetensors

### Similar Projects

1. **TorchSharp**: https://github.com/dotnet/TorchSharp (LibTorch bindings)
2. **ML.NET**: https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet
3. **DiffSharp**: https://diffsharp.github.io/ (F# automatic differentiation)

---

## Project Deliverables Checklist

### Phase 1: Core Tensor Infrastructure
- [ ] Device abstraction (CPU only)
- [ ] Tensor wrapper class
- [ ] Factory methods (Zeros, Ones, Randn, Rand, FromArray)
- [ ] Basic operations (Reshape, Transpose, Slice)
- [ ] Unit tests for tensor creation and manipulation

### Phase 2: Automatic Differentiation
- [ ] GradientFunction base class
- [ ] Backward operations (Add, Mul, MatMul, ReLU, etc.)
- [ ] GradientEngine with topological sort
- [ ] Operator overloads with autograd
- [ ] Gradient correctness tests

### Phase 3: Neural Network Layers
- [ ] Module base class
- [ ] Linear layer
- [ ] Conv2d layer
- [ ] Embedding layer
- [ ] LayerNorm layer
- [ ] Dropout layer
- [ ] Activation functions (ReLU, GELU, Sigmoid, Tanh, Softmax)
- [ ] Layer tests

### Phase 4: Training Infrastructure
- [ ] Optimizer base class
- [ ] SGD optimizer
- [ ] Adam optimizer
- [ ] AdamW optimizer
- [ ] MSE loss
- [ ] CrossEntropy loss
- [ ] BCE loss
- [ ] Training convergence tests

### Phase 5: Model Loading
- [ ] Safetensors parser
- [ ] VarBuilder
- [ ] FP16/BF16 conversion
- [ ] Model loading tests

### Phase 6: Examples and Documentation
- [ ] MNIST example
- [ ] Simple NN example
- [ ] README with quick start
- [ ] API documentation
- [ ] Performance benchmarks

---

## Success Metrics

1. **Functional Completeness**
   - All core operations implemented
   - Autograd working for all operations
   - Can train simple models (MNIST, simple regression)

2. **Performance**
   - Competitive with ML.NET for CPU operations
   - Efficient memory usage (minimal allocations)
   - SIMD utilization verified

3. **Code Quality**
   - >80% test coverage
   - No memory leaks
   - Clear API surface

4. **Documentation**
   - All public APIs documented
   - Working examples for common use cases
   - Migration guide from PyTorch

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0: Setup | 1 day | Project structure |
| 1: Core Tensor | 3 days | Tensor operations |
| 2: Autograd | 4 days | Automatic differentiation |
| 3: NN Layers | 4 days | Neural network primitives |
| 4: Training | 3 days | Optimizers and loss functions |
| 5: Model Loading | 3 days | Safetensors support |
| 6: Examples/Docs | 3 days | Examples and documentation |
| **Total** | **21 days** | **Production-ready framework** |

---

## Notes for AI Coding Assistant

### Implementation Order

Follow phases strictly in order - each builds on the previous:
1. Start with Phase 0 (setup)
2. Implement Phase 1 completely before moving to Phase 2
3. Write tests as you implement each feature
4. Don't skip error handling

### Code Style

- Use modern C# features (records, pattern matching, etc.)
- Prefer `readonly` and immutability where possible
- Use `Span<T>` over arrays for better performance
- Document all public APIs with XML comments
- Follow .NET naming conventions

### Common Pitfalls

1. **Broadcasting**: Implement proper gradient unbroadcasting in backward passes
2. **Memory**: Always dispose tensors in tight loops
3. **Numerical Stability**: Use log-sum-exp for softmax, clamp for log
4. **Shape Validation**: Validate tensor shapes before operations
5. **Device Consistency**: Ensure all tensors in an operation are on the same device

### Testing Requirements

- Write unit tests for every operation
- Verify gradients with numerical differentiation
- Test edge cases (empty tensors, single element, etc.)
- Include integration tests for training loops

### Performance Notes

- Profile before optimizing
- Use TensorPrimitives for all element-wise operations
- Consider parallelization only for large batches
- Benchmark against ML.NET as baseline

---

## Appendix: API Quick Reference

```csharp
// Tensor creation
Tensor.Zeros<float>([3, 3])
Tensor.Ones<float>([2, 4])
Tensor.Randn<float>([100, 50])
Tensor.FromArray(data, shape)

// Operations
tensor.Reshape([10, 5])
tensor.Transpose()
tensor.MatMul(other)
tensor + other
tensor * scalar

// Autograd
tensor.RequiresGrad = true
loss.Backward()
tensor.Grad

// Layers
new Linear(inFeatures, outFeatures)
new Conv2d(inChannels, outChannels, kernelSize)
new LayerNorm(normalizedShape)

// Training
var optimizer = new Adam(model.Parameters(), lr: 0.001f)
optimizer.ZeroGrad()
optimizer.StepOptimizer()

// Loss
var criterion = new CrossEntropyLoss()
var loss = criterion.Forward(outputs, targets)
```

---

**End of PRD**

This document provides a complete specification for implementing AILib, a .NET-native ML framework inspired by Candle's design. Follow the phases sequentially, implementing tests alongside each feature. Good luck!
