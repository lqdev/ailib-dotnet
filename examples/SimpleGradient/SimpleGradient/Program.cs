using AILib;

Console.WriteLine("=== AILib Simple Gradient Example ===\n");

// Example 1: Basic autograd with simple operations
Console.WriteLine("Example 1: Basic Gradient Computation");
Console.WriteLine("--------------------------------------");

var x = Tensor.FromArray([2.0f], [1], requiresGrad: true);
var y = Tensor.FromArray([3.0f], [1], requiresGrad: true);

// Forward pass: z = x² + xy + y²
var x2 = x * x;
var xy = x * y;
var y2 = y * y;
var z = x2 + xy + y2;

Console.WriteLine($"x = {x.Item()}, y = {y.Item()}");
Console.WriteLine($"z = x² + xy + y² = {z.Item()}");

// Backward pass
z.Backward();

Console.WriteLine($"dz/dx = 2x + y = {x.Grad!.Item()} (expected: {2 * x.Item() + y.Item()})");
Console.WriteLine($"dz/dy = x + 2y = {y.Grad!.Item()} (expected: {x.Item() + 2 * y.Item()})");
Console.WriteLine();

// Example 2: Matrix operations
Console.WriteLine("Example 2: Matrix Multiplication Gradients");
Console.WriteLine("------------------------------------------");

var A = Tensor.FromArray(new float[] { 1, 2, 3, 4 }, [2, 2], requiresGrad: true);
var B = Tensor.FromArray(new float[] { 5, 6, 7, 8 }, [2, 2], requiresGrad: true);

Console.WriteLine("Matrix A:");
Console.WriteLine($"  [{A.Data[0, 0]}, {A.Data[0, 1]}]");
Console.WriteLine($"  [{A.Data[1, 0]}, {A.Data[1, 1]}]");

Console.WriteLine("\nMatrix B:");
Console.WriteLine($"  [{B.Data[0, 0]}, {B.Data[0, 1]}]");
Console.WriteLine($"  [{B.Data[1, 0]}, {B.Data[1, 1]}]");

// C = A @ B
var C = A.MatMul(B);

Console.WriteLine("\nMatrix C = A @ B:");
Console.WriteLine($"  [{C.Data[0, 0]}, {C.Data[0, 1]}]");
Console.WriteLine($"  [{C.Data[1, 0]}, {C.Data[1, 1]}]");

// Take sum to get scalar for backward
var sum = C.Sum();
Console.WriteLine($"\nSum of C = {sum.Item()}");

sum.Backward();

Console.WriteLine("\nGradient of A:");
Console.WriteLine($"  [{A.Grad!.Data[0, 0]}, {A.Grad!.Data[0, 1]}]");
Console.WriteLine($"  [{A.Grad!.Data[1, 0]}, {A.Grad!.Data[1, 1]}]");

Console.WriteLine("\nGradient of B:");
Console.WriteLine($"  [{B.Grad!.Data[0, 0]}, {B.Grad!.Data[0, 1]}]");
Console.WriteLine($"  [{B.Grad!.Data[1, 0]}, {B.Grad!.Data[1, 1]}]");
Console.WriteLine();

// Example 3: Simple optimization (finding minimum)
Console.WriteLine("Example 3: Simple Optimization");
Console.WriteLine("-------------------------------");
Console.WriteLine("Finding minimum of f(x) = (x - 3)²");

var param = Tensor.FromArray([0.0f], [1], requiresGrad: true);
var learningRate = 0.1f;

for (int i = 0; i < 20; i++)
{
    // f(x) = (x - 3)²
    var diff = param - 3.0f;
    var loss = diff * diff;

    if (i % 5 == 0)
    {
        Console.WriteLine($"Iteration {i}: x = {param.Item():F4}, loss = {loss.Item():F4}");
    }

    // Backward
    loss.Backward();

    // Manual SGD update: x = x - lr * grad
    var newParam = param - learningRate * param.Grad!;
    param = Tensor.FromArray([newParam.Item()], [1], requiresGrad: true);
}

Console.WriteLine($"Final: x = {param.Item():F4} (expected: 3.0)");
Console.WriteLine();

// Example 4: Chained operations
Console.WriteLine("Example 4: Complex Computation Graph");
Console.WriteLine("------------------------------------");

var a = Tensor.FromArray([1.0f], [1], requiresGrad: true);
var b = Tensor.FromArray([2.0f], [1], requiresGrad: true);
var c = Tensor.FromArray([3.0f], [1], requiresGrad: true);

// Complex expression: result = (a + b) * (b + c)
var sum1 = a + b;        // 3
var sum2 = b + c;        // 5
var result = sum1 * sum2; // 15

Console.WriteLine($"a = {a.Item()}, b = {b.Item()}, c = {c.Item()}");
Console.WriteLine($"result = (a + b) * (b + c) = {result.Item()}");

result.Backward();

// ∂result/∂a = (b + c) = 5
// ∂result/∂b = (b + c) + (a + b) = 5 + 3 = 8
// ∂result/∂c = (a + b) = 3

Console.WriteLine($"∂result/∂a = {a.Grad!.Item()} (expected: 5)");
Console.WriteLine($"∂result/∂b = {b.Grad!.Item()} (expected: 8)");
Console.WriteLine($"∂result/∂c = {c.Grad!.Item()} (expected: 3)");
Console.WriteLine();

Console.WriteLine("=== All examples completed successfully! ===");

