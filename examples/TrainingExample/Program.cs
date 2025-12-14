using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.NN.Activations;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

Console.WriteLine("=== AILib Training Examples ===\n");

// Example 1: Simple Linear Regression
Console.WriteLine("Example 1: Linear Regression");
Console.WriteLine("-----------------------------");
LinearRegressionExample();

Console.WriteLine("\n");

// Example 2: Binary Classification
Console.WriteLine("Example 2: Binary Classification");
Console.WriteLine("---------------------------------");
BinaryClassificationExample();

Console.WriteLine("\n");

// Example 3: Multi-layer Network
Console.WriteLine("Example 3: Multi-Layer Network");
Console.WriteLine("-------------------------------");
MultiLayerNetworkExample();

Console.WriteLine("\n=== All examples completed! ===");

static void LinearRegressionExample()
{
    // Generate synthetic data: y = 2.5*x + 1.3
    var numSamples = 100;
    var x = Tensor.Randn<float>([numSamples, 1]);
    var y = System.Numerics.Tensors.Tensor.CreateFromShape<float>([numSamples, 1], false);
    
    for (int i = 0; i < numSamples; i++)
    {
        y[i, 0] = 2.5f * x.Data[i, 0] + 1.3f + (float)(Random.Shared.NextDouble() - 0.5) * 0.1f;
    }
    var yTensor = Tensor.FromTensor(y);
    
    // Create model
    var model = new Linear(1, 1);
    var optimizer = new SGD(model.Parameters(), lr: 0.01f);
    var lossFunc = new MSELoss();
    
    Console.WriteLine("Training linear regression (y = 2.5*x + 1.3)...");
    
    // Train
    for (int epoch = 0; epoch < 100; epoch++)
    {
        var pred = model.Forward(x);
        var loss = lossFunc.Forward(pred, yTensor);
        
        loss.Backward();
        optimizer.StepOptimizer();
        optimizer.ZeroGrad();
        
        if (epoch % 20 == 0)
        {
            Console.WriteLine($"  Epoch {epoch,3}: Loss = {loss.Item():F6}");
        }
    }
    
    // Check learned parameters
    var weight = model.Parameters().First();
    var bias = model.Parameters().Last();
    Console.WriteLine($"\nLearned parameters:");
    Console.WriteLine($"  Weight: {weight.Data[0, 0]:F3} (expected: 2.5)");
    Console.WriteLine($"  Bias:   {bias.Data[0]:F3} (expected: 1.3)");
}

static void BinaryClassificationExample()
{
    // Generate synthetic binary classification data
    var numSamples = 200;
    var x = Tensor.Randn<float>([numSamples, 5]);
    var y = System.Numerics.Tensors.Tensor.CreateFromShape<float>([numSamples, 1], false);
    
    // Create labels based on a simple rule: sum > 0 -> class 1, else class 0
    for (int i = 0; i < numSamples; i++)
    {
        float sum = 0;
        for (int j = 0; j < 5; j++)
        {
            sum += x.Data[i, j];
        }
        y[i, 0] = sum > 0 ? 1.0f : 0.0f;
    }
    var yTensor = Tensor.FromTensor(y);
    
    // Create model: Linear -> Sigmoid
    var linear = new Linear(5, 1);
    var sigmoid = new Sigmoid();
    var optimizer = new Adam(linear.Parameters(), lr: 0.01f);
    var lossFunc = new BCELoss();
    
    Console.WriteLine("Training binary classifier...");
    
    // Train
    for (int epoch = 0; epoch < 50; epoch++)
    {
        var logits = linear.Forward(x);
        var probs = sigmoid.Forward(logits);
        var loss = lossFunc.Forward(probs, yTensor);
        
        loss.Backward();
        optimizer.StepOptimizer();
        optimizer.ZeroGrad();
        
        if (epoch % 10 == 0)
        {
            Console.WriteLine($"  Epoch {epoch,3}: Loss = {loss.Item():F6}");
        }
    }
    
    // Evaluate accuracy
    var testLogits = linear.Forward(x);
    var testProbs = sigmoid.Forward(testLogits);
    
    int correct = 0;
    for (int i = 0; i < numSamples; i++)
    {
        var predicted = testProbs.Data[i, 0] > 0.5f ? 1.0f : 0.0f;
        if (predicted == y[i, 0])
        {
            correct++;
        }
    }
    
    Console.WriteLine($"\nAccuracy: {correct}/{numSamples} ({100.0 * correct / numSamples:F1}%)");
}

static void MultiLayerNetworkExample()
{
    // Build a 3-layer network with activations
    var layer1 = new Linear(20, 64);
    var relu1 = new ReLU();
    var dropout1 = new Dropout(0.2f);
    var layer2 = new Linear(64, 32);
    var relu2 = new ReLU();
    var layer3 = new Linear(32, 10);
    
    // Collect all parameters
    var allParams = layer1.Parameters()
        .Concat(layer2.Parameters())
        .Concat(layer3.Parameters());
    
    var optimizer = new AdamW(allParams, lr: 0.001f, weightDecay: 0.01f);
    var lossFunc = new MSELoss();
    
    Console.WriteLine("Training 3-layer network (20->64->32->10)...");
    Console.WriteLine("Architecture: Linear(20,64)->ReLU->Dropout(0.2)->Linear(64,32)->ReLU->Linear(32,10)");
    
    // Training mode
    dropout1.Train();
    
    // Train for a few iterations
    for (int iter = 0; iter < 30; iter++)
    {
        // Generate random batch
        var input = Tensor.Randn<float>([16, 20]);
        var target = Tensor.Randn<float>([16, 10]);
        
        // Forward pass
        var h1 = layer1.Forward(input);
        h1 = relu1.Forward(h1);
        h1 = dropout1.Forward(h1);
        
        var h2 = layer2.Forward(h1);
        h2 = relu2.Forward(h2);
        
        var output = layer3.Forward(h2);
        
        // Compute loss
        var loss = lossFunc.Forward(output, target);
        
        // Backward pass
        loss.Backward();
        optimizer.StepOptimizer();
        optimizer.ZeroGrad();
        
        if (iter % 10 == 0)
        {
            Console.WriteLine($"  Iteration {iter,3}: Loss = {loss.Item():F6}");
        }
    }
    
    // Switch to evaluation mode
    dropout1.Eval();
    
    Console.WriteLine("\nNetwork summary:");
    var totalParams = allParams.Count();
    Console.WriteLine($"  Total parameter tensors: {totalParams}");
    Console.WriteLine($"  Optimizer: AdamW (lr=0.001, weight_decay=0.01)");
    Console.WriteLine($"  Status: Ready for inference (eval mode)");
}

