# AILib - .NET Machine Learning Framework

A minimalist, PyTorch-like machine learning framework for .NET that leverages the native `System.Numerics.Tensors` API for CPU-based tensor operations. Inspired by HuggingFace's Candle (Rust) design philosophy, this library provides automatic differentiation, neural network primitives, and model training capabilities while maintaining .NET idioms and type safety.

## Features

- 🔢 **Tensor Operations** - Rich set of tensor operations with automatic differentiation
- 🧠 **Neural Network Layers** - Common layers (Linear, Conv2d, Embedding, LayerNorm, Dropout)
- 🎯 **Optimizers** - SGD, Adam, and AdamW optimizers for training
- 📊 **Loss Functions** - MSE, CrossEntropy, and Binary CrossEntropy
- 📦 **Model Loading** - Support for loading models from safetensors format
- ⚡ **CPU-Optimized** - SIMD-accelerated operations using .NET's native APIs
- 🎓 **PyTorch-like API** - Familiar ergonomics for ML practitioners

## Quick Start

```csharp
using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Create tensors
var x = Tensor.Randn<float>([100, 10], requiresGrad: true);
var y = Tensor.Randn<float>([100, 1]);

// Define model
var model = new Sequential(
    new Linear(10, 20),
    new ReLU(),
    new Dropout(0.2),
    new Linear(20, 1)
);

// Setup training
var optimizer = new Adam(model.Parameters(), lr: 0.001f);
var criterion = new MSELoss();

// Training loop
for (int epoch = 0; epoch < 100; epoch++)
{
    optimizer.ZeroGrad();
    
    var predictions = model.Forward(x);
    var loss = criterion.Forward(predictions, y);
    
    loss.Backward();
    optimizer.Step();
    
    Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item()}");
}
```

## Project Structure

```
AILib/
├── src/
│   ├── AILib.Core/          # Core tensor operations and autograd
│   ├── AILib.NN/            # Neural network layers and training
│   └── AILib.Transformers/  # Transformer models and components
├── tests/
│   ├── AILib.Core.Tests/    # Core functionality tests
│   └── AILib.NN.Tests/      # Neural network tests
├── examples/                # Example applications
└── docs/                    # Documentation
```

## Requirements

- .NET 10.0 or later
- System.Numerics.Tensors 10.0.1+

## Building

```bash
dotnet build
```

## Running Tests

```bash
dotnet test
```

## Goals and Non-Goals

### Goals
- Core tensor operations built on `System.Numerics.Tensors`
- Reverse-mode automatic differentiation (backpropagation)
- Common neural network layers and primitives
- Training infrastructure with optimizers and loss functions
- Model interoperability via safetensors format
- Competitive CPU performance through SIMD optimization
- PyTorch-like API surface for ease of adoption

### Non-Goals (v1.0)
- GPU support (CPU-only)
- Distributed training
- Production inference optimization
- Pre-trained model zoo
- Mobile/edge optimization

## Documentation

See [docs/](docs/) for detailed documentation and examples.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by:
- [HuggingFace Candle](https://github.com/huggingface/candle) - Rust ML framework
- [PyTorch](https://pytorch.org/) - Python ML framework
- [System.Numerics.Tensors](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors) - .NET tensor primitives
