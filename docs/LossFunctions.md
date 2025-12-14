# Loss Functions Guide

Loss functions (also called cost functions or objective functions) measure how well your model's predictions match the target values.

## Table of Contents

- [Overview](#overview)
- [MSE Loss](#mse-loss-mean-squared-error)
- [BCE Loss](#bce-loss-binary-cross-entropy)
- [Cross-Entropy Loss](#cross-entropy-loss)
- [Choosing a Loss Function](#choosing-a-loss-function)

## Overview

All loss functions in AILib follow a similar pattern:

```csharp
using AILib.NN.Loss;

// Create loss function
var lossFunction = new MSELoss();

// Compute loss
var predictions = model.Forward(input);
var loss = lossFunction.Forward(predictions, targets);

// Backward pass
loss.Backward();
```

## MSE Loss (Mean Squared Error)

Measures the average squared difference between predictions and targets.

### Formula

```
MSE = (1/N) * Σ(prediction - target)²
```

### When to Use

- **Regression tasks**: Predicting continuous values (house prices, temperatures, etc.)
- **When outliers should be penalized**: Squared term heavily penalizes large errors
- **Gaussian noise assumption**: Optimal when errors follow a normal distribution

### Constructor

```csharp
var mse = new MSELoss();
```

No parameters needed - MSE is parameter-free.

### Example: Linear Regression

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Generate synthetic data: y = 3x + 2
var x = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5 }, [5, 1]);
var y = Tensor.FromArray(new float[] { 5, 8, 11, 14, 17 }, [5, 1]);

// Create model and optimizer
var model = new Linear(1, 1);
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
    
    if (epoch % 20 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item():F4}");
    }
}

// Check results
var finalPred = model.Forward(x);
Console.WriteLine("Predictions vs Targets:");
for (int i = 0; i < 5; i++)
{
    Console.WriteLine($"  {finalPred.Data[i, 0]:F2} vs {y.Data[i, 0]:F2}");
}
```

### Example: Multi-output Regression

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Loss;

// Predict multiple outputs (e.g., x, y, z coordinates)
var model = new Linear(10, 3);  // 10 features -> 3 outputs

var input = Tensor.Randn<float>([32, 10]);    // Batch of 32
var target = Tensor.Randn<float>([32, 3]);    // 3 target values each

var output = model.Forward(input);
var loss = new MSELoss().Forward(output, target);

Console.WriteLine($"MSE Loss: {loss.Item():F4}");
```

### Properties

**Pros**:
- Simple to understand and implement
- Smooth gradients for optimization
- Differentiable everywhere

**Cons**:
- Sensitive to outliers (squared term)
- Not suitable for classification
- Can be dominated by large errors

## BCE Loss (Binary Cross-Entropy)

Measures the difference between predicted probabilities and binary targets (0 or 1).

### Formula

```
BCE = -(1/N) * Σ[y * log(p) + (1-y) * log(1-p)]
```

Where:
- `y`: target (0 or 1)
- `p`: predicted probability (must be in [0, 1])

### When to Use

- **Binary classification**: Two-class problems (spam/not spam, cat/dog)
- **Multi-label classification**: Multiple binary decisions per sample
- **Sigmoid outputs**: When using sigmoid activation on output layer

### Constructor

```csharp
var bce = new BCELoss(eps: 1e-7f);  // eps prevents log(0)
```

**Parameters**:
- `eps` (float): Small constant to avoid log(0), default: 1e-7

### Example: Binary Classification

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Activations;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Create binary classifier
var linear = new Linear(10, 1);
var sigmoid = new Sigmoid();

// Generate data
var input = Tensor.Randn<float>([100, 10]);
var target = Tensor.FromArray(
    Enumerable.Range(0, 100).Select(i => i % 2 == 0 ? 1.0f : 0.0f).ToArray(),
    [100, 1]
);

// Setup training
var optimizer = new Adam(linear.Parameters(), lr: 0.01f);
var lossFunc = new BCELoss();

// Train
for (int epoch = 0; epoch < 50; epoch++)
{
    // Forward: linear -> sigmoid -> probabilities
    var logits = linear.Forward(input);
    var probs = sigmoid.Forward(logits);
    
    // Compute BCE loss
    var loss = lossFunc.Forward(probs, target);
    
    // Backward and update
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    if (epoch % 10 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item():F4}");
    }
}

// Evaluate
var testLogits = linear.Forward(input);
var testProbs = sigmoid.Forward(testLogits);

Console.WriteLine("Sample predictions:");
for (int i = 0; i < 5; i++)
{
    var pred = testProbs.Data[i, 0] > 0.5f ? 1 : 0;
    Console.WriteLine($"  Prob: {testProbs.Data[i, 0]:F3}, Pred: {pred}, Target: {target.Data[i, 0]}");
}
```

### Example: Multi-label Classification

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Activations;
using AILib.NN.Loss;

// Predict multiple binary labels per sample
var model = new Linear(20, 5);  // 5 independent binary predictions
var sigmoid = new Sigmoid();

var input = Tensor.Randn<float>([32, 20]);
var targets = Tensor.Rand<float>([32, 5]);  // Each in [0, 1]
// Round to 0 or 1
for (int i = 0; i < 32; i++)
{
    for (int j = 0; j < 5; j++)
    {
        targets.Data[i, j] = targets.Data[i, j] > 0.5f ? 1.0f : 0.0f;
    }
}

var logits = model.Forward(input);
var probs = sigmoid.Forward(logits);
var loss = new BCELoss().Forward(probs, targets);

Console.WriteLine($"Multi-label BCE Loss: {loss.Item():F4}");
```

### Properties

**Pros**:
- Probabilistic interpretation
- Works well with sigmoid activation
- Suitable for imbalanced datasets (with weighting)

**Cons**:
- Requires probabilities ([0, 1] range)
- Can be numerically unstable without epsilon
- Not suitable for multi-class (use CrossEntropy instead)

### Important Notes

⚠️ **Input Requirements**:
- Predictions must be in [0, 1] (use Sigmoid activation)
- Targets must be 0 or 1
- The `eps` parameter prevents log(0) = -∞

⚠️ **Common Mistake**:
```csharp
// WRONG: Using raw logits
var logits = model.Forward(input);
var loss = new BCELoss().Forward(logits, targets);  // Will fail!

// CORRECT: Apply sigmoid first
var probs = new Sigmoid().Forward(logits);
var loss = new BCELoss().Forward(probs, targets);
```

## Cross-Entropy Loss

Measures the difference between predicted class probabilities and true class labels for multi-class classification.

### Formula

```
CrossEntropy = -(1/N) * Σ Σ y_i,c * log(p_i,c)
```

Where:
- `y_i,c`: 1 if sample i belongs to class c, 0 otherwise
- `p_i,c`: predicted probability for sample i, class c

### When to Use

- **Multi-class classification**: More than 2 classes (digit recognition, image classification)
- **Mutually exclusive classes**: Each sample belongs to exactly one class
- **Softmax outputs**: When using softmax activation on output layer

### Constructor

```csharp
var ce = new CrossEntropyLoss();
```

### Methods

AILib provides two methods:

#### 1. ForwardOneHot (Recommended)

Use when targets are one-hot encoded:

```csharp
var logits = model.Forward(input);  // Raw scores, shape: [batch, num_classes]
var targets = Tensor.FromArray(     // One-hot, shape: [batch, num_classes]
    new float[] { 1, 0, 0,  // Sample 0: class 0
                  0, 1, 0,  // Sample 1: class 1
                  0, 0, 1 },// Sample 2: class 2
    [3, 3]
);

var loss = new CrossEntropyLoss().ForwardOneHot(logits, targets);
```

#### 2. Forward (Not Yet Implemented)

For integer class indices:

```csharp
// Future implementation
var targets = Tensor.FromArray(new int[] { 0, 1, 2 }, [3]);  // Class indices
var loss = new CrossEntropyLoss().Forward(logits, targets);
```

Currently throws `NotImplementedException` - use `ForwardOneHot` instead.

### Example: Multi-class Classification

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Activations;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// 3-class classifier
var model = new Linear(20, 3);  // Input: 20 features, Output: 3 classes

// Generate one-hot encoded targets
var input = Tensor.Randn<float>([64, 20]);
var targets = Tensor.Zeros<float>([64, 3]);

// Assign each sample to a random class
var random = new Random();
for (int i = 0; i < 64; i++)
{
    int classIdx = random.Next(3);
    targets.Data[i, classIdx] = 1.0f;
}

// Setup training
var optimizer = new Adam(model.Parameters(), lr: 0.001f);
var lossFunc = new CrossEntropyLoss();

// Train
for (int epoch = 0; epoch < 100; epoch++)
{
    // Forward: model outputs logits (raw scores)
    var logits = model.Forward(input);
    
    // Compute cross-entropy loss
    // (ForwardOneHot applies log-softmax internally)
    var loss = lossFunc.ForwardOneHot(logits, targets);
    
    // Backward and update
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    if (epoch % 20 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item():F4}");
    }
}

// Predict
var testLogits = model.Forward(input);
var testProbs = testLogits.Softmax();  // Convert to probabilities

Console.WriteLine("Sample predictions:");
for (int i = 0; i < 3; i++)
{
    var p0 = testProbs.Data[i, 0];
    var p1 = testProbs.Data[i, 1];
    var p2 = testProbs.Data[i, 2];
    var predicted = p0 > p1 ? (p0 > p2 ? 0 : 2) : (p1 > p2 ? 1 : 2);
    
    Console.WriteLine($"  Probs: [{p0:F3}, {p1:F3}, {p2:F3}], Pred: {predicted}");
}
```

### Example: Image Classification (MNIST-style)

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Activations;
using AILib.NN.Loss;

// Simple classifier for 28x28 images, 10 classes
var flatten = 28 * 28;
var fc1 = new Linear(flatten, 128);
var relu = new ReLU();
var fc2 = new Linear(128, 10);

// Forward pass
Tensor<float> Forward(Tensor<float> images)
{
    var x = images.Reshape([images.Shape[0], flatten]);
    x = fc1.Forward(x);
    x = relu.Forward(x);
    x = fc2.Forward(x);
    return x;  // Return logits
}

// Training data (simulated)
var images = Tensor.Randn<float>([64, 1, 28, 28]);  // Batch of images
var labels = Tensor.Zeros<float>([64, 10]);         // One-hot labels

// Assign random labels
for (int i = 0; i < 64; i++)
{
    labels.Data[i, i % 10] = 1.0f;
}

// Train
var logits = Forward(images);
var loss = new CrossEntropyLoss().ForwardOneHot(logits, labels);

Console.WriteLine($"Classification Loss: {loss.Item():F4}");
```

### Properties

**Pros**:
- Standard for multi-class classification
- Probabilistic interpretation
- Numerically stable with log-softmax

**Cons**:
- Only for mutually exclusive classes
- Requires one-hot encoding currently
- Not suitable for regression or multi-label

### Important Notes

⚠️ **Model Output**:
- Model should output **raw logits** (unnormalized scores)
- Do NOT apply softmax before loss (it's applied internally)
- LogSoftmax is more numerically stable than Softmax + Log

⚠️ **Target Format**:
- Currently requires one-hot encoding
- Each row must sum to 1.0
- Each sample assigned to exactly one class

⚠️ **Common Mistakes**:
```csharp
// WRONG: Applying softmax before CrossEntropy
var probs = logits.Softmax();
var loss = lossFunc.ForwardOneHot(probs, targets);  // Double softmax!

// CORRECT: Use raw logits
var loss = lossFunc.ForwardOneHot(logits, targets);
```

## Choosing a Loss Function

### Decision Tree

```
Is this a classification or regression task?
│
├─ Regression (continuous values)
│  └─ Use MSELoss
│
└─ Classification (categories)
   │
   ├─ Binary (2 classes)
   │  └─ Use BCELoss with Sigmoid
   │
   ├─ Multi-label (multiple binary)
   │  └─ Use BCELoss with Sigmoid
   │
   └─ Multi-class (3+ exclusive classes)
      └─ Use CrossEntropyLoss
```

### Quick Reference Table

| Task | Loss Function | Output Activation | Example |
|------|---------------|-------------------|---------|
| Regression | MSELoss | None (linear) | House prices, temperature |
| Binary Classification | BCELoss | Sigmoid | Spam detection, sentiment |
| Multi-label | BCELoss | Sigmoid | Image tags, document topics |
| Multi-class | CrossEntropyLoss | None (logits) | Digit recognition, object classification |

## Complete Training Examples

### Example 1: Simple Regression

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Problem: Predict y = 2*x1 + 3*x2 - 1
var numSamples = 1000;
var x = Tensor.Randn<float>([numSamples, 2]);
var y = Tensor.CreateFromShape<float>([numSamples, 1], false);

// Generate targets
for (int i = 0; i < numSamples; i++)
{
    y.Data[i, 0] = 2 * x.Data[i, 0] + 3 * x.Data[i, 1] - 1;
}

// Model
var model = new Linear(2, 1);
var optimizer = new Adam(model.Parameters(), lr: 0.01f);
var lossFunc = new MSELoss();

// Train
for (int epoch = 0; epoch < 200; epoch++)
{
    var pred = model.Forward(x);
    var loss = lossFunc.Forward(pred, y);
    
    loss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    if (epoch % 50 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss.Item():F6}");
    }
}

// Check learned parameters
Console.WriteLine("Learned parameters (should be ~[2, 3] and bias ~-1):");
foreach (var param in model.Parameters())
{
    Console.WriteLine($"  Shape: [{string.Join(", ", param.Shape.ToArray())}]");
}
```

### Example 2: Binary Classification with Validation

```csharp
using AILib;
using AILib.NN.Layers;
using AILib.NN.Activations;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Generate training and validation data
var trainX = Tensor.Randn<float>([800, 10]);
var trainY = Tensor.Rand<float>([800, 1]);
var valX = Tensor.Randn<float>([200, 10]);
var valY = Tensor.Rand<float>([200, 1]);

// Round to 0 or 1
for (int i = 0; i < 800; i++) trainY.Data[i, 0] = trainY.Data[i, 0] > 0.5f ? 1.0f : 0.0f;
for (int i = 0; i < 200; i++) valY.Data[i, 0] = valY.Data[i, 0] > 0.5f ? 1.0f : 0.0f;

// Model
var linear = new Linear(10, 1);
var sigmoid = new Sigmoid();
var optimizer = new Adam(linear.Parameters(), lr: 0.001f);
var lossFunc = new BCELoss();

// Train with validation
for (int epoch = 0; epoch < 100; epoch++)
{
    // Training
    linear.Train();
    var trainLogits = linear.Forward(trainX);
    var trainProbs = sigmoid.Forward(trainLogits);
    var trainLoss = lossFunc.Forward(trainProbs, trainY);
    
    trainLoss.Backward();
    optimizer.StepOptimizer();
    optimizer.ZeroGrad();
    
    // Validation
    if (epoch % 10 == 0)
    {
        linear.Eval();
        var valLogits = linear.Forward(valX);
        var valProbs = sigmoid.Forward(valLogits);
        var valLoss = lossFunc.Forward(valProbs, valY);
        
        Console.WriteLine($"Epoch {epoch}: Train Loss={trainLoss.Item():F4}, Val Loss={valLoss.Item():F4}");
    }
}
```

## Tips and Best Practices

### 1. Monitor Loss Values

```csharp
var losses = new List<float>();

for (int epoch = 0; epoch < 100; epoch++)
{
    var loss = /* ... compute loss ... */;
    losses.Add(loss.Item());
    
    // Check for issues
    if (float.IsNaN(loss.Item()))
    {
        Console.WriteLine("Loss is NaN! Training failed.");
        break;
    }
    
    if (losses.Count > 10 && losses[^1] > losses[^10])
    {
        Console.WriteLine("Loss increasing - consider reducing learning rate");
    }
}
```

### 2. Use Appropriate Loss for Task

```csharp
// WRONG: Using MSE for classification
var logits = classifier.Forward(input);
var loss = new MSELoss().Forward(logits, oneHotTargets);  // Poor results!

// CORRECT: Use CrossEntropy for classification
var loss = new CrossEntropyLoss().ForwardOneHot(logits, oneHotTargets);
```

### 3. Normalize Targets for Regression

```csharp
// Normalize targets to [0, 1] or [-1, 1]
var targetMean = /* compute mean */;
var targetStd = /* compute std */;
var normalizedTargets = (targets - targetMean) / targetStd;

// Train with normalized targets
var loss = new MSELoss().Forward(predictions, normalizedTargets);

// Denormalize predictions for evaluation
var denormalizedPreds = predictions * targetStd + targetMean;
```

### 4. Balanced vs Imbalanced Classification

For imbalanced datasets, consider:
- Weighted loss functions (custom implementation)
- Oversampling minority class
- Adjusting prediction threshold

## Next Steps

- Learn about [Optimizers](Optimizers.md) for training
- Explore [Neural Network Layers](Layers.md)
- See complete [Training Guide](TrainingGuide.md)
