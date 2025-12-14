using AILib;
using AILib.NN;
using AILib.NN.Layers;
using System.Numerics;

namespace AILib.Transformers.Layers;

/// <summary>
/// Multi-Head Attention mechanism as described in "Attention is All You Need".
/// </summary>
/// <remarks>
/// Computes scaled dot-product attention across multiple heads in parallel.
/// Formula: Attention(Q,K,V) = softmax(QK^T / √d_k)V
/// </remarks>
public class MultiHeadAttention : Module
{
    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _scale;
    
    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _outProj;

    /// <summary>
    /// Creates a new multi-head attention layer.
    /// </summary>
    /// <param name="embedDim">Total dimension of the model</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="bias">Whether to include bias in linear projections</param>
    public MultiHeadAttention(int embedDim, int numHeads, bool bias = true)
    {
        if (embedDim % numHeads != 0)
        {
            throw new ArgumentException(
                $"embed_dim ({embedDim}) must be divisible by num_heads ({numHeads})");
        }

        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _scale = 1.0f / MathF.Sqrt(_headDim);

        // Query, Key, Value projections
        _qProj = new Linear(embedDim, embedDim, bias);
        _kProj = new Linear(embedDim, embedDim, bias);
        _vProj = new Linear(embedDim, embedDim, bias);
        
        // Output projection
        _outProj = new Linear(embedDim, embedDim, bias);

        RegisterModule("q_proj", _qProj);
        RegisterModule("k_proj", _kProj);
        RegisterModule("v_proj", _vProj);
        RegisterModule("out_proj", _outProj);
    }

    /// <summary>
    /// Forward pass through multi-head attention.
    /// </summary>
    /// <param name="query">Query tensor [batch, seqLen, embedDim]</param>
    /// <param name="key">Key tensor [batch, seqLen, embedDim] (if null, uses query)</param>
    /// <param name="value">Value tensor [batch, seqLen, embedDim] (if null, uses query)</param>
    /// <param name="attentionMask">Optional mask [batch, seqLen, seqLen] or [seqLen, seqLen]</param>
    /// <returns>Attention output [batch, seqLen, embedDim]</returns>
    public Tensor<float> Forward(
        Tensor<float> query,
        Tensor<float>? key = null,
        Tensor<float>? value = null,
        Tensor<float>? attentionMask = null)
    {
        // Self-attention: use query for key and value if not provided
        key ??= query;
        value ??= query;

        var shape = query.Shape.ToArray();
        if (shape.Length != 3)
        {
            throw new ArgumentException("Query must be 3D tensor [batch, seqLen, embedDim]");
        }

        var batchSize = shape[0];
        var seqLen = shape[1];

        // Project to Q, K, V
        var q = _qProj.Forward(query);  // [batch, seqLen, embedDim]
        var k = _kProj.Forward(key);     // [batch, seqLen, embedDim]
        var v = _vProj.Forward(value);   // [batch, seqLen, embedDim]

        // Reshape to split heads: [batch, seqLen, numHeads, headDim]
        q = q.Reshape([batchSize, seqLen, _numHeads, _headDim]);
        k = k.Reshape([batchSize, seqLen, _numHeads, _headDim]);
        v = v.Reshape([batchSize, seqLen, _numHeads, _headDim]);

        // Transpose to: [batch, numHeads, seqLen, headDim]
        q = q.Permute([0, 2, 1, 3]);
        k = k.Permute([0, 2, 1, 3]);
        v = v.Permute([0, 2, 1, 3]);

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // Shape: [batch, numHeads, seqLen, seqLen]
        var scores = ScaledDotProductAttention(q, k, v, attentionMask);

        // Transpose back: [batch, seqLen, numHeads, headDim]
        scores = scores.Permute([0, 2, 1, 3]);

        // Concatenate heads: [batch, seqLen, embedDim]
        var output = scores.Reshape([batchSize, seqLen, _embedDim]);

        // Output projection
        output = _outProj.Forward(output);

        return output;
    }

    /// <summary>
    /// Computes scaled dot-product attention.
    /// </summary>
    private Tensor<float> ScaledDotProductAttention(
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Tensor<float>? mask)
    {
        // q, k, v: [batch, numHeads, seqLen, headDim]
        
        // Compute attention scores: Q @ K^T
        // We need to transpose k from [..., seqLen, headDim] to [..., headDim, seqLen]
        var kT = TransposeLast2Dims(k);  // [batch, numHeads, headDim, seqLen]
        var scores = BatchedMatMul(q, kT);  // [batch, numHeads, seqLen, seqLen]

        // Scale by sqrt(d_k)
        scores = scores * _scale;

        // Apply mask if provided (e.g., for causal attention)
        if (mask != null)
        {
            // Add large negative value to masked positions
            scores = ApplyMask(scores, mask);
        }

        // Apply softmax along last dimension
        var attnWeights = scores.Softmax();  // [batch, numHeads, seqLen, seqLen]

        // Apply attention to values: attnWeights @ V
        var output = BatchedMatMul(attnWeights, v);  // [batch, numHeads, seqLen, headDim]

        return output;
    }

    /// <summary>
    /// Transposes the last two dimensions of a 4D tensor.
    /// </summary>
    private Tensor<float> TransposeLast2Dims(Tensor<float> tensor)
    {
        var shape = tensor.Shape.ToArray();
        if (shape.Length != 4)
        {
            throw new ArgumentException("Expected 4D tensor");
        }

        // Permute from [batch, numHeads, seqLen, headDim] to [batch, numHeads, headDim, seqLen]
        return tensor.Permute([0, 1, 3, 2]);
    }

    /// <summary>
    /// Batched matrix multiplication for 4D tensors.
    /// </summary>
    private Tensor<float> BatchedMatMul(Tensor<float> a, Tensor<float> b)
    {
        var shapeA = a.Shape.ToArray();
        var shapeB = b.Shape.ToArray();

        if (shapeA.Length != 4 || shapeB.Length != 4)
        {
            throw new ArgumentException("Both tensors must be 4D");
        }

        var batch = shapeA[0];
        var numHeads = shapeA[1];
        var m = shapeA[2];
        var k = shapeA[3];
        var n = shapeB[3];

        if (k != shapeB[2])
        {
            throw new ArgumentException($"Incompatible dimensions for matmul: {k} vs {shapeB[2]}");
        }

        var outputTensor = System.Numerics.Tensors.Tensor.CreateFromShape<float>([batch, numHeads, m, n], false);
        var output = Tensor.FromTensor(outputTensor, requiresGrad: a.RequiresGrad || b.RequiresGrad);

        // Perform batched matrix multiplication
        for (nint b_idx = 0; b_idx < batch; b_idx++)
        {
            for (nint h = 0; h < numHeads; h++)
            {
                for (nint i = 0; i < m; i++)
                {
                    for (nint j = 0; j < n; j++)
                    {
                        float sum = 0.0f;
                        for (nint p = 0; p < k; p++)
                        {
                            sum += a.Data[b_idx, h, i, p] * b.Data[b_idx, h, p, j];
                        }
                        output.Data[b_idx, h, i, j] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies attention mask to scores.
    /// </summary>
    private Tensor<float> ApplyMask(Tensor<float> scores, Tensor<float> mask)
    {
        var scoresShape = scores.Shape.ToArray();
        var maskShape = mask.Shape.ToArray();

        var outputTensor = System.Numerics.Tensors.Tensor.CreateFromShape<float>(scoresShape, false);
        var output = Tensor.FromTensor(outputTensor, requiresGrad: scores.RequiresGrad);

        // Copy scores to output and apply mask
        var largeNegative = -1e9f;
        
        if (maskShape.Length == 2)
        {
            // Broadcast mask [seqLen, seqLen] to [batch, numHeads, seqLen, seqLen]
            var seqLen = maskShape[0];
            for (nint b = 0; b < scoresShape[0]; b++)
            {
                for (nint h = 0; h < scoresShape[1]; h++)
                {
                    for (nint i = 0; i < seqLen; i++)
                    {
                        for (nint j = 0; j < seqLen; j++)
                        {
                            var val = scores.Data[b, h, i, j];
                            output.Data[b, h, i, j] = mask.Data[i, j] == 0.0f ? largeNegative : val;
                        }
                    }
                }
            }
        }
        else if (maskShape.Length == 4)
        {
            // Direct application
            for (nint b = 0; b < scoresShape[0]; b++)
            {
                for (nint h = 0; h < scoresShape[1]; h++)
                {
                    for (nint i = 0; i < scoresShape[2]; i++)
                    {
                        for (nint j = 0; j < scoresShape[3]; j++)
                        {
                            var val = scores.Data[b, h, i, j];
                            output.Data[b, h, i, j] = mask.Data[b, h, i, j] == 0.0f ? largeNegative : val;
                        }
                    }
                }
            }
        }

        return output;
    }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        // Default to self-attention
        return Forward(input, null, null, null);
    }
}
