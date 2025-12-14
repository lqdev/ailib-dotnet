# Optimizers Guide

Optimizers update model parameters based on computed gradients to minimize the loss function.

## Table of Contents

- [Overview](#overview)
- [SGD (Stochastic Gradient Descent)](#sgd-stochastic-gradient-descent)
- [Adam](#adam)
- [AdamW](#adamw)
- [Usage Examples](#usage-examples)
- [Choosing an Optimizer](#choosing-an-optimizer)

## Overview

All optimizers in AILib inherit from the `Optimizer` base class and provide:

- **Automatic parameter tracking**: Pass parameters from your model
- **Gradient zeroing**: `ZeroGrad()` method to clear gradients
- **Parameter updates**: `StepOptimizer()` method to update parameters

### Basic Optimizer Pattern

```csharp
using AILib.NN.Optimizers;

// 1. Create optimizer with model parameters
var optimizer = new Adam(model.Parameters(), lr: 0.001f);

// 2. Training loop
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    // Forward pass
    var output = model.Forward(input);
    var loss = lossFunction.Forward(output, target);
    
    // Backward pass
    loss.Backward();
    
    // Update parameters
    optimizer.StepOptimizer();
    
    // Clear gradients for next iteration
    optimizer.ZeroGrad();
}
```

## SGD (Stochastic Gradient Descent)

The simplest optimizer that updates parameters in the direction of the negative gradient.

### Constructor

```csharp
var sgd = new SGD(
    parameters: model.Parameters(),
    lr: 0.01f,           // Learning rate
    momentum: 0.0f,      // Momentum factor (default: 0)
    weightDecay: 0.0f    // L2 penalty (default: 0)
);
```

### Parameters

- **lr** (float): Learning rate, controls step size
  - Typical range: 0.1 to 0.0001
  - Larger = faster learning but less stable
  - Smaller = slower but more stable

- **momentum** (float, optional): Momentum factor for accelerating SGD
  - Range: [0, 1], default: 0
  - Helps smooth updates and escape local minima
  - Typical values: 0.9 or 0.99

- **weightDecay** (float, optional): L2 regularization strength
  - Range: [0, ∞), default: 0
  - Helps prevent overfitting
  - Typical values: 1e-4 to 1e-2

### Update Rule

Without momentum:
```
θ = θ - lr * (∇L + weightDecay * θ)
```

With momentum:
```
v = momentum * v + ∇L
θ = θ - lr * v
```

### Example

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Create model
var model = new Linear(10, 1);

// Create SGD optimizer with momentum
var optimizer = new SGD(
    model.Parameters(),
    lr: 0.01f,
    momentum: 0.9f
);

// Training
var input = Tensor.Randn<float>([32, 10]);
var target = Tensor.Randn<float>([32, 1]);

for (int step = 0; step < 100; step++)
{
    var output = model.Forward(input);
    var loss = new MSELoss().Forward(output, target);
    
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    if (step % 10 == 0)
    {
        Console.WriteLine($"Step {step}, Loss: {loss.Item():F4}");
    }
}
```

### When to Use SGD

- **Simple problems**: Quick to implement, easy to tune
- **Well-behaved loss landscapes**: When gradients are relatively smooth
- **Limited memory**: Minimal overhead compared to adaptive methods
- **With momentum**: Often competitive with Adam on vision tasks

## Adam

Adaptive Moment Estimation - combines momentum with adaptive learning rates per parameter.

### Constructor

```csharp
var adam = new Adam(
    parameters: model.Parameters(),
    lr: 0.001f,          // Learning rate
    beta1: 0.9f,         // Exponential decay rate for first moment (default: 0.9)
    beta2: 0.999f,       // Exponential decay rate for second moment (default: 0.999)
    eps: 1e-8f,          // Small constant for numerical stability (default: 1e-8)
    weightDecay: 0.0f    // L2 penalty (default: 0)
);
```

### Parameters

- **lr** (float): Learning rate
  - Typical range: 0.001 to 0.0001
  - Less sensitive than SGD, good default: 0.001

- **beta1** (float): Exponential decay for first moment (momentum)
  - Range: [0, 1), default: 0.9
  - Controls how much history to retain for momentum

- **beta2** (float): Exponential decay for second moment (variance)
  - Range: [0, 1), default: 0.999
  - Controls adaptation of learning rate per parameter

- **eps** (float): Small constant to prevent division by zero
  - Default: 1e-8
  - Rarely needs tuning

- **weightDecay** (float): L2 regularization
  - Applied to gradients (coupled weight decay)
  - For decoupled weight decay, use AdamW instead

### Update Rule

```
m_t = beta1 * m_{t-1} + (1 - beta1) * ∇L        # First moment (momentum)
v_t = beta2 * v_{t-1} + (1 - beta2) * (∇L)²     # Second moment (variance)

m̂_t = m_t / (1 - beta1^t)                       # Bias correction
v̂_t = v_t / (1 - beta2^t)

θ = θ - lr * m̂_t / (√v̂_t + eps)
```

### Example

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Multi-layer network
var fc1 = new Linear(784, 128);
var fc2 = new Linear(128, 10);

// Combine parameters from both layers
var parameters = fc1.Parameters().Concat(fc2.Parameters());

// Create Adam optimizer
var optimizer = new Adam(parameters, lr: 0.001f);

// Training loop
var lossFunc = new MSELoss();

for (int epoch = 0; epoch < 10; epoch++)
{
    var input = Tensor.Randn<float>([64, 784]);
    var target = Tensor.Randn<float>([64, 10]);
    
    // Forward
    var hidden = fc1.Forward(input);
    hidden = hidden.ReLU();  // Activation
    var output = fc2.Forward(hidden);
    
    var loss = lossFunc.Forward(output, target);
    
    // Backward
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item():F4}");
}
```

### When to Use Adam

- **Default choice**: Works well out-of-the-box for most problems
- **NLP/Transformers**: Industry standard for language models
- **Sparse gradients**: Adaptive learning rates help
- **Fast prototyping**: Less sensitive to learning rate choice

## AdamW

Adam with decoupled weight decay - fixes weight decay implementation in Adam.

### Constructor

```csharp
var adamw = new AdamW(
    parameters: model.Parameters(),
    lr: 0.001f,          // Learning rate
    beta1: 0.9f,         // First moment decay (default: 0.9)
    beta2: 0.999f,       // Second moment decay (default: 0.999)
    eps: 1e-8f,          // Numerical stability constant (default: 1e-8)
    weightDecay: 0.01f   // Decoupled weight decay (default: 0.01)
);
```

### Parameters

Same as Adam, but:
- **weightDecay** is applied differently (decoupled from gradient)
- Typically use larger values: 0.01 to 0.1

### Update Rule

```
m_t = beta1 * m_{t-1} + (1 - beta1) * ∇L
v_t = beta2 * v_{t-1} + (1 - beta2) * (∇L)²

m̂_t = m_t / (1 - beta1^t)
v̂_t = v_t / (1 - beta2^t)

θ = θ * (1 - lr * weightDecay) - lr * m̂_t / (√v̂_t + eps)
```

The key difference: weight decay is applied directly to parameters, not gradients.

### Example

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;

// Create model
var model = new Linear(100, 10);

// AdamW with decoupled weight decay
var optimizer = new AdamW(
    model.Parameters(),
    lr: 0.001f,
    weightDecay: 0.01f  // Larger than typical Adam weight decay
);

// Training
for (int step = 0; step < 1000; step++)
{
    var input = Tensor.Randn<float>([32, 100]);
    var target = Tensor.Randn<float>([32, 10]);
    
    var output = model.Forward(input);
    var loss = new MSELoss().Forward(output, target);
    
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
}
```

### When to Use AdamW

- **Modern default**: Generally preferred over Adam
- **Transformers**: Better regularization for large models
- **Fine-tuning**: Better performance when fine-tuning pre-trained models
- **Long training**: Improves generalization over extended training

## Usage Examples

### Example 1: Simple Linear Regression

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Generate synthetic data: y = 2x + 1
var x = Tensor.Randn<float>([100, 1]);
var y = 2.0f * x + 1.0f;

// Create model
var model = new Linear(1, 1);

// Create optimizer
var optimizer = new SGD(model.Parameters(), lr: 0.01f);
var lossFunc = new MSELoss();

// Train
for (int epoch = 0; epoch < 100; epoch++)
{
    var pred = model.Forward(x);
    var loss = lossFunc.Forward(pred, y);
    
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    if (epoch % 10 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item():F4}");
    }
}

// Check learned parameters
var weight = model.Parameters().First();
var bias = model.Parameters().Last();
Console.WriteLine($"Learned: weight={weight.Item():F2}, bias={bias.Item():F2}");
// Should be close to: weight=2.0, bias=1.0
```

### Example 2: Multi-layer Network with Different Optimizers

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Activations;

// Build network
var layer1 = new Linear(784, 256);
var relu1 = new ReLU();
var layer2 = new Linear(256, 128);
var relu2 = new ReLU();
var layer3 = new Linear(128, 10);

// Collect all parameters
var allParams = layer1.Parameters()
    .Concat(layer2.Parameters())
    .Concat(layer3.Parameters());

// Try different optimizers
IOptimizer optimizer;

// Option 1: SGD with momentum
optimizer = new SGD(allParams, lr: 0.01f, momentum: 0.9f);

// Option 2: Adam
optimizer = new Adam(allParams, lr: 0.001f);

// Option 3: AdamW
optimizer = new AdamW(allParams, lr: 0.001f, weightDecay: 0.01f);

// Forward pass
Tensor<float> Forward(Tensor<float> x)
{
    x = layer1.Forward(x);
    x = relu1.Forward(x);
    x = layer2.Forward(x);
    x = relu2.Forward(x);
    x = layer3.Forward(x);
    return x;
}

// Training step
var input = Tensor.Randn<float>([32, 784]);
var target = Tensor.Randn<float>([32, 10]);

var output = Forward(input);
var loss = new MSELoss().Forward(output, target);

loss.Backward();
optimizer.StepOptimizer();
optimizer.ZeroGrad();
```

## Choosing an Optimizer

### Quick Guide

| Scenario | Recommended Optimizer | Settings |
|----------|----------------------|----------|
| **Quick prototype** | Adam | lr=0.001 |
| **Production model** | AdamW | lr=0.001, wd=0.01 |
| **Computer Vision** | SGD + momentum | lr=0.01, mom=0.9 |
| **NLP/Transformers** | AdamW | lr=1e-4, wd=0.01 |
| **Fine-tuning** | AdamW | lr=1e-5, wd=0.01 |
| **Limited memory** | SGD | lr=0.01 |

### Decision Tree

1. **Are you fine-tuning a pre-trained model?**
   - Yes → Use AdamW with small learning rate (1e-5 to 1e-4)
   
2. **Is training time critical?**
   - Yes → Use Adam for faster convergence
   - No → Consider SGD with momentum for better final performance
   
3. **Do you have lots of data and compute?**
   - Yes → SGD with momentum (may achieve better generalization)
   - No → Adam or AdamW for sample efficiency

4. **Is this a vision task with strong augmentation?**
   - Yes → SGD with momentum often works best
   - No → Adam/AdamW are safer defaults

## Tips and Best Practices

### 1. Learning Rate Scheduling

```csharp
// Manual learning rate decay
var initialLr = 0.001f;
for (int epoch = 0; epoch < 100; epoch++)
{
    var currentLr = initialLr * MathF.Pow(0.95f, epoch / 10);
    
    // Recreate optimizer with new learning rate
    optimizer = new Adam(model.Parameters(), lr: currentLr);
    
    // Train for one epoch...
}
```

### 2. Gradient Clipping (Manual)

```csharp
// Clip gradients to prevent exploding gradients
float maxNorm = 1.0f;
foreach (var param in model.Parameters())
{
    if (param.Grad != null)
    {
        // Compute gradient norm and scale if needed
        // (simplified - full implementation would compute L2 norm)
        var grad = param.Grad;
        // Apply clipping...
    }
}
optimizer.StepOptimizer();
```

### 3. Warm-up for Adam/AdamW

```csharp
// Linear warm-up for first few steps
int warmupSteps = 1000;
float targetLr = 0.001f;

for (int step = 0; step < totalSteps; step++)
{
    float currentLr;
    if (step < warmupSteps)
    {
        currentLr = targetLr * (step + 1) / warmupSteps;
    }
    else
    {
        currentLr = targetLr;
    }
    
    optimizer = new Adam(model.Parameters(), lr: currentLr);
    
    // Training step...
}
```

### 4. Always Zero Gradients

```csharp
// WRONG: Gradients accumulate!
loss.Backward();
optimizer.StepOptimizer();
// Next iteration will have stale gradients!

// CORRECT: Zero before next backward
loss.Backward();
optimizer.StepOptimizer();
optimizer.ZeroGrad();  // Clear for next iteration
```

### 5. Check for NaN/Inf

```csharp
if (float.IsNaN(loss.Item()) || float.IsInfinity(loss.Item()))
{
    Console.WriteLine("Training diverged! Try:");
    Console.WriteLine("- Reducing learning rate");
    Console.WriteLine("- Adding gradient clipping");
    Console.WriteLine("- Checking input data for NaN/Inf");
    break;
}
```

## Common Issues

### Issue: Loss not decreasing

**Solutions**:
- Reduce learning rate by 10x
- Check that `ZeroGrad()` is called after each step
- Verify gradients are flowing (check `param.Grad != null`)
- Try a different optimizer (Adam if using SGD, or vice versa)

### Issue: Training unstable

**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Use AdamW instead of Adam
- Check for NaN values in data

### Issue: Overfitting

**Solutions**:
- Increase weight decay
- Add dropout layers
- Use more data / data augmentation
- Reduce model size

## Next Steps

- Learn about [Loss Functions](LossFunctions.md)
- See complete [Training Examples](TrainingGuide.md)
- Understand [Neural Network Layers](Layers.md)
