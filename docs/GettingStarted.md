# Getting Started with AILib

AILib is a minimalist, PyTorch-like machine learning framework for .NET that provides automatic differentiation and tensor operations using `System.Numerics.Tensors`.

## Installation

AILib is currently under development. To use it, clone the repository and build from source:

```bash
git clone https://github.com/lqdev/ailib-dotnet.git
cd ailib-dotnet
dotnet build
```

## Quick Start

### Creating Tensors

AILib provides several factory methods for creating tensors:

```csharp
using AILib;

// Create tensors from arrays
var data = new float[] { 1, 2, 3, 4, 5, 6 };
var tensor = Tensor.FromArray(data, [2, 3]);  // Shape: [2, 3]

// Create tensors filled with zeros or ones
var zeros = Tensor.Zeros<float>([3, 3]);      // 3x3 matrix of zeros
var ones = Tensor.Ones<float>([2, 4]);         // 2x4 matrix of ones

// Create tensors with specific values
var fives = Tensor.Full<float>([2, 2], 5.0f); // 2x2 matrix filled with 5.0

// Create tensors with random values
var normal = Tensor.Randn<float>([100, 50]);  // N(0,1) distribution
var uniform = Tensor.Rand<float>([10, 10]);   // Uniform [0, 1)
```

### Tensor Operations

AILib supports standard tensor operations:

```csharp
// Element-wise operations
var a = Tensor.Randn<float>([2, 3]);
var b = Tensor.Randn<float>([2, 3]);

var sum = a + b;           // Element-wise addition
var diff = a - b;          // Element-wise subtraction  
var product = a * b;       // Element-wise multiplication
var quotient = a / b;      // Element-wise division

// Scalar operations
var scaled = a * 2.5f;     // Multiply by scalar
var offset = a + 1.0f;     // Add scalar

// Matrix operations
var x = Tensor.Randn<float>([3, 4]);
var y = Tensor.Randn<float>([4, 2]);
var matmul = x.MatMul(y);  // Matrix multiplication -> [3, 2]

// Shape operations
var reshaped = a.Reshape([3, 2]);      // Reshape to [3, 2]
var transposed = matmul.Transpose();   // Transpose -> [2, 3]
var flattened = a.Flatten();           // Flatten to 1D

// Reductions
var total = a.Sum();       // Sum all elements
var average = a.Mean();    // Mean of all elements
```

### Automatic Differentiation

Enable gradient tracking to use automatic differentiation:

```csharp
// Create tensors with gradient tracking
var x = Tensor.Randn<float>([10, 5], requiresGrad: true);
var w = Tensor.Randn<float>([5, 3], requiresGrad: true);
var b = Tensor.Randn<float>([3], requiresGrad: true);

// Forward pass
var y = x.MatMul(w) + b;
var loss = y.Sum();

// Backward pass - compute gradients
loss.Backward();

// Access gradients
Console.WriteLine($"Gradient of w: {w.Grad}");
Console.WriteLine($"Gradient of b: {b.Grad}");

// Zero gradients for next iteration
w.ZeroGrad();
b.ZeroGrad();
```

### Example: Linear Regression

Here's a complete example of training a linear model:

```csharp
using AILib;

// Generate synthetic data: y = 2x + 1 + noise
var random = new Random(42);
var xData = Enumerable.Range(0, 100).Select(i => (float)i / 10).ToArray();
var yData = xData.Select(x => 2 * x + 1 + (float)(random.NextDouble() - 0.5) * 0.5).ToArray();

var x = Tensor.FromArray(xData, [100, 1]);
var yTrue = Tensor.FromArray(yData, [100, 1]);

// Initialize parameters
var w = Tensor.FromArray([0.0f], [1, 1], requiresGrad: true);
var b = Tensor.FromArray([0.0f], [1], requiresGrad: true);

// Training loop
var learningRate = 0.01f;
for (int epoch = 0; epoch < 100; epoch++)
{
    // Forward pass
    var yPred = x.MatMul(w) + b;
    var diff = yPred - yTrue;
    var loss = (diff * diff).Mean();  // MSE loss

    // Backward pass
    loss.Backward();

    // Manual gradient descent update (no optimizer yet)
    w = w - learningRate * w.Grad!;
    b = b - learningRate * b.Grad!;

    // Zero gradients
    w.ZeroGrad();
    b.ZeroGrad();

    if (epoch % 10 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item()}");
    }
}

Console.WriteLine($"Learned parameters: w = {w.Item()}, b = {b.Item()}");
// Should be close to w=2.0, b=1.0
```

## API Reference

### Tensor Creation

| Method | Description |
|--------|-------------|
| `Tensor.FromArray<T>(T[] data, ReadOnlySpan<nint> shape)` | Create tensor from array |
| `Tensor.Zeros<T>(ReadOnlySpan<nint> shape)` | Create tensor filled with zeros |
| `Tensor.Ones<T>(ReadOnlySpan<nint> shape)` | Create tensor filled with ones |
| `Tensor.Full<T>(ReadOnlySpan<nint> shape, T value)` | Create tensor filled with value |
| `Tensor.Randn<T>(ReadOnlySpan<nint> shape)` | Create tensor with N(0,1) values |
| `Tensor.Rand<T>(ReadOnlySpan<nint> shape)` | Create tensor with uniform [0,1) values |

All creation methods accept an optional `requiresGrad` parameter for gradient tracking.

### Tensor Properties

| Property | Description |
|----------|-------------|
| `Shape` | ReadOnlySpan containing the shape |
| `Rank` | Number of dimensions |
| `ElementCount` | Total number of elements |
| `Device` | Device where tensor resides (CPU only) |
| `RequiresGrad` | Whether gradient tracking is enabled |
| `Grad` | The accumulated gradient (if any) |

### Tensor Operations

| Operation | Description |
|-----------|-------------|
| `a + b` | Element-wise addition |
| `a - b` | Element-wise subtraction |
| `a * b` | Element-wise multiplication |
| `a / b` | Element-wise division |
| `-a` | Unary negation |
| `a.MatMul(b)` | Matrix multiplication |
| `a.Transpose()` | Transpose 2D tensor |
| `a.Reshape(shape)` | Reshape to new shape |
| `a.Flatten()` | Flatten to 1D |
| `a.Sum()` | Sum all elements |
| `a.Mean()` | Mean of all elements |

### Autograd

| Method | Description |
|--------|-------------|
| `tensor.Backward()` | Compute gradients via backpropagation |
| `tensor.ZeroGrad()` | Clear accumulated gradients |

## Current Limitations (v0.1)

- **CPU-only**: No GPU acceleration yet
- **Basic operations**: Limited set of tensor operations
- **No neural network layers**: Phase 3 is in progress
- **No optimizers**: Manual gradient descent only
- **No model loading**: Safetensors support coming in Phase 4

## Next Steps

1. **Implement Neural Network Layers** (Phase 3)
   - Module base class
   - Linear, Dropout, LayerNorm layers
   - Activation functions (ReLU, GELU, Sigmoid, Softmax)

2. **Add Training Infrastructure** (Phase 4)
   - Optimizers (SGD, Adam, AdamW)
   - Loss functions (MSE, CrossEntropy, BCE)

3. **Model Loading** (Phase 5)
   - Safetensors format support
   - Pre-trained model loading

## Contributing

Contributions are welcome! Please see the main README for contribution guidelines.

## License

MIT License - see LICENSE file for details.
