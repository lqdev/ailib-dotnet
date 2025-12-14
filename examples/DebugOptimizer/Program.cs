using AILib;
using AILib.NN.Layers;
using AILib.NN.Optimizers;

Console.WriteLine("=== Debugging Optimizer Parameter Updates ===\n");

var layer = new Linear(2, 1);
var params_list = layer.Parameters().ToList();

Console.WriteLine($"Number of parameters: {params_list.Count}");
foreach (var p in params_list)
{
    Console.WriteLine($"  Parameter shape: [{string.Join(", ", p.Shape.ToArray())}], RequiresGrad: {p.RequiresGrad}");
}

var param = params_list[0];  // weight
Console.WriteLine($"\nInitial weight[0,0] = {param.Data[0, 0]}");

// Create simple input
var input = AILib.Tensor.FromArray(new float[] { 1.0f, 2.0f }, [1, 2], requiresGrad: false);  // Don't need grad for input
Console.WriteLine($"Input: [{input.Data[0, 0]}, {input.Data[0, 1]}]");

// Forward pass
var output = layer.Forward(input);
Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape.ToArray())}]");
Console.WriteLine($"Output value: {output.Data[0, 0]}");
Console.WriteLine($"Output RequiresGrad: {output.RequiresGrad}");

// Create loss
var loss = output.Sum();
Console.WriteLine($"\nLoss = {loss.Item()}");
Console.WriteLine($"Loss RequiresGrad: {loss.RequiresGrad}");

// Check if parameters still have RequiresGrad
Console.WriteLine($"\nBefore backward:");
Console.WriteLine($"  param RequiresGrad: {param.RequiresGrad}");
Console.WriteLine($"  param has Grad: {param.Grad != null}");

// Backward
try
{
    loss.Backward();
    Console.WriteLine("\nBackward completed successfully");
}
catch (Exception ex)
{
    Console.WriteLine($"\nBackward failed: {ex.Message}");
}

Console.WriteLine($"\nAfter backward:");
Console.WriteLine($"  param has Grad: {param.Grad != null}");
if (param.Grad != null)
{
    Console.WriteLine($"  param Grad shape: [{string.Join(", ", param.Grad.Shape.ToArray())}]");
    Console.WriteLine($"  param Grad[0, 0] = {param.Grad.Data[0, 0]}");
}

