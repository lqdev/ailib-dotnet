using System.Numerics.Tensors;
using AILib;

// Test 1: Can we modify a System.Numerics.Tensors.Tensor?
Console.WriteLine("=== Test 1: Modify System.Numerics.Tensors.Tensor ===");
var data = new float[] { 1, 2, 3, 4 };
var sysTensor = Tensor.Create(data, [2, 2]);

Console.WriteLine($"Before: sysTensor[0, 0] = {sysTensor[0, 0]}");
sysTensor[0, 0] = 99;
Console.WriteLine($"After:  sysTensor[0, 0] = {sysTensor[0, 0]}");
Console.WriteLine($"Original array[0] = {data[0]}");

// Test 2: Our AILib Tensor
Console.WriteLine("\n=== Test 2: Modify AILib.Tensor ===");
var aiTensor = AILib.Tensor.FromArray(new float[] { 5, 6, 7, 8 }, [2, 2]);
Console.WriteLine($"Before: aiTensor.Data[0, 0] = {aiTensor.Data[0, 0]}");
aiTensor.Data[0, 0] = 100;
Console.WriteLine($"After:  aiTensor.Data[0, 0] = {aiTensor.Data[0, 0]}");
