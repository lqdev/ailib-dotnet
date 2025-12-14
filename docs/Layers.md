# Neural Network Layers Guide

AILib provides a modular approach to building neural networks with the `Module` base class and various layer implementations.

## Table of Contents

- [Module Base Class](#module-base-class)
- [Linear Layer](#linear-layer)
- [Activation Layers](#activation-layers)
- [Regularization Layers](#regularization-layers)
- [Building Custom Layers](#building-custom-layers)

## Module Base Class

The `Module` class is the foundation for all neural network layers in AILib. It provides:

- **Parameter management**: Automatic tracking of trainable parameters
- **Training/evaluation modes**: Control behavior like dropout
- **Gradient zeroing**: Convenient gradient reset for optimization

### Basic Usage

```csharp
using AILib.NN;

public class MyCustomLayer : Module
{
    private readonly Tensor<float> _weight;
    
    public MyCustomLayer(int inputSize, int outputSize)
    {
        _weight = Tensor.Randn<float>([outputSize, inputSize]);
        RegisterParameter("weight", _weight);
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        return input.MatMul(_weight.Transpose());
    }
}
```

### Key Methods

```csharp
// Get all trainable parameters
IEnumerable<Tensor<float>> params = module.Parameters();

// Switch between training and evaluation modes
module.Train();  // Enable training mode (affects Dropout, etc.)
module.Eval();   // Enable evaluation mode

// Zero out all parameter gradients
module.ZeroGrad();
```

## Linear Layer

Fully connected (dense) layer that applies a linear transformation: `y = xW^T + b`

### Constructor

```csharp
var linear = new Linear(
    inFeatures: 10,    // Number of input features
    outFeatures: 5,    // Number of output features
    bias: true         // Whether to include bias term (default: true)
);
```

### Features

- **Kaiming Uniform Initialization**: Weights initialized using He initialization for better gradient flow
- **Optional bias**: Can disable bias if needed (e.g., before BatchNorm)
- **Automatic gradient tracking**: Parameters registered for optimization

### Example

```csharp
using AILib;
using AILib.NN.Layers;

// Create a linear layer: 784 inputs -> 128 outputs
var layer = new Linear(784, 128);

// Forward pass with batch of 32 samples
var input = Tensor.Randn<float>([32, 784]);
var output = layer.Forward(input);

Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape.ToArray())}]");
// Output: Output shape: [32, 128]

// Access parameters
var weight = layer.Parameters().First();  // Shape: [128, 784]
var bias = layer.Parameters().Last();     // Shape: [128]
```

## Activation Layers

Activation functions introduce non-linearity into neural networks.

### ReLU (Rectified Linear Unit)

```csharp
using AILib.NN.Activations;

var relu = new ReLU();
var input = Tensor.FromArray(new float[] { -2, -1, 0, 1, 2 }, [5]);
var output = relu.Forward(input);
// Output: [0, 0, 0, 1, 2]
```

**Formula**: `ReLU(x) = max(0, x)`

**Use case**: Default activation for most hidden layers in deep networks.

### Sigmoid

```csharp
var sigmoid = new Sigmoid();
var input = Tensor.FromArray(new float[] { -1, 0, 1 }, [3]);
var output = sigmoid.Forward(input);
// Output: [0.268, 0.5, 0.731]
```

**Formula**: `σ(x) = 1 / (1 + e^(-x))`

**Use case**: Binary classification output layers, gate mechanisms.

### Tanh (Hyperbolic Tangent)

```csharp
var tanh = new Tanh();
var input = Tensor.FromArray(new float[] { -1, 0, 1 }, [3]);
var output = tanh.Forward(input);
// Output: [-0.762, 0, 0.762]
```

**Formula**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Use case**: Alternative to sigmoid, outputs in range [-1, 1].

### GELU (Gaussian Error Linear Unit)

```csharp
var gelu = new GELU();
var input = Tensor.Randn<float>([10, 20]);
var output = gelu.Forward(input);
```

**Formula**: `GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

**Use case**: Modern transformer architectures (GPT, BERT).

### Softmax

```csharp
var softmax = new Softmax(dim: -1);  // Apply along last dimension
var logits = Tensor.FromArray(new float[] { 1, 2, 3 }, [3]);
var probs = softmax.Forward(logits);
// Output: [0.09, 0.24, 0.67] (sums to 1.0)
```

**Formula**: `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`

**Use case**: Multi-class classification output layers.

## Regularization Layers

### Dropout

Randomly sets elements to zero during training to prevent overfitting.

```csharp
using AILib.NN.Layers;

var dropout = new Dropout(p: 0.5f);  // Drop 50% of elements

// Training mode: randomly drops elements and scales by 1/(1-p)
dropout.Train();
var train_output = dropout.Forward(input);

// Evaluation mode: passes input through unchanged
dropout.Eval();
var eval_output = dropout.Forward(input);
```

**Parameters**:
- `p`: Probability of dropping an element (0 to 1)

**Behavior**:
- **Training**: Randomly zeros elements and scales remaining by `1/(1-p)`
- **Evaluation**: Identity function (no dropout)

### LayerNorm (Layer Normalization)

Normalizes activations across features.

```csharp
var layerNorm = new LayerNorm(
    normalizedShape: 128,  // Size of feature dimension
    eps: 1e-5f             // Small constant for numerical stability
);

var input = Tensor.Randn<float>([32, 128]);
var output = layerNorm.Forward(input);
```

**Formula**: `LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β`

Where:
- `μ`: mean across features
- `σ²`: variance across features
- `γ` (weight): learnable scale parameter
- `β` (bias): learnable shift parameter

**Use case**: Transformer architectures, stabilizing training.

## Building Custom Layers

### Example: Custom MLP Block

```csharp
using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.NN.Activations;

public class MLPBlock : Module
{
    private readonly Linear _linear1;
    private readonly ReLU _relu;
    private readonly Dropout _dropout;
    private readonly Linear _linear2;
    
    public MLPBlock(int inputSize, int hiddenSize, int outputSize, float dropoutRate = 0.1f)
    {
        _linear1 = new Linear(inputSize, hiddenSize);
        _relu = new ReLU();
        _dropout = new Dropout(dropoutRate);
        _linear2 = new Linear(hiddenSize, outputSize);
        
        // Register submodules for parameter tracking
        RegisterModule("linear1", _linear1);
        RegisterModule("relu", _relu);
        RegisterModule("dropout", _dropout);
        RegisterModule("linear2", _linear2);
    }
    
    public override Tensor<float> Forward(Tensor<float> input)
    {
        var x = _linear1.Forward(input);
        x = _relu.Forward(x);
        x = _dropout.Forward(x);
        x = _linear2.Forward(x);
        return x;
    }
}

// Usage
var mlp = new MLPBlock(784, 256, 10);
mlp.Train();  // Enable dropout

var input = Tensor.Randn<float>([32, 784]);
var output = mlp.Forward(input);

// All parameters from nested modules are accessible
var allParams = mlp.Parameters().ToList();
Console.WriteLine($"Total parameters: {allParams.Count}");  // 4 (2 weights + 2 biases)
```

### Example: Simple Classifier

```csharp
public class SimpleClassifier : Module
{
    private readonly Linear _fc1;
    private readonly ReLU _relu1;
    private readonly Linear _fc2;
    private readonly ReLU _relu2;
    private readonly Linear _fc3;
    
    public SimpleClassifier(int inputDim, int numClasses)
    {
        _fc1 = new Linear(inputDim, 128);
        _relu1 = new ReLU();
        _fc2 = new Linear(128, 64);
        _relu2 = new ReLU();
        _fc3 = new Linear(64, numClasses);
        
        RegisterModule("fc1", _fc1);
        RegisterModule("relu1", _relu1);
        RegisterModule("fc2", _fc2);
        RegisterModule("relu2", _relu2);
        RegisterModule("fc3", _fc3);
    }
    
    public override Tensor<float> Forward(Tensor<float> x)
    {
        x = _fc1.Forward(x);
        x = _relu1.Forward(x);
        x = _fc2.Forward(x);
        x = _relu2.Forward(x);
        x = _fc3.Forward(x);
        return x;
    }
}
```

## Best Practices

1. **Always register parameters and submodules**: Use `RegisterParameter()` and `RegisterModule()` so they're tracked for optimization.

2. **Use appropriate initialization**: Linear layer uses Kaiming initialization by default, which is suitable for ReLU activations.

3. **Handle train/eval modes**: Call `module.Train()` during training and `module.Eval()` before evaluation.

4. **Zero gradients**: Always call `module.ZeroGrad()` or `optimizer.ZeroGrad()` before each training step.

5. **Parameter sharing**: You can share parameters between layers by registering the same tensor multiple times.

## Next Steps

- Learn about [Optimizers](Optimizers.md) for training your networks
- Explore [Loss Functions](LossFunctions.md) for different tasks
- See [Training Guide](TrainingGuide.md) for complete training loops
