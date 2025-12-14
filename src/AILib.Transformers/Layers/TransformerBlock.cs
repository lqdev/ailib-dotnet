using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.NN.Activations;

namespace AILib.Transformers.Layers;

/// <summary>
/// Transformer encoder block combining multi-head attention and feed-forward network.
/// </summary>
/// <remarks>
/// Architecture:
/// 1. Multi-Head Attention + Residual + LayerNorm
/// 2. Feed-Forward Network + Residual + LayerNorm
/// </remarks>
public class TransformerBlock : Module
{
    private readonly MultiHeadAttention _attention;
    private readonly LayerNorm _norm1;
    private readonly Linear _ffn1;
    private readonly Linear _ffn2;
    private readonly LayerNorm _norm2;
    private readonly Dropout? _dropout;
    private readonly int _embedDim;
    private readonly int _ffnDim;

    /// <summary>
    /// Creates a new transformer block.
    /// </summary>
    /// <param name="embedDim">Dimension of the model</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="ffnDim">Dimension of the feed-forward network (typically 4 * embedDim)</param>
    /// <param name="dropoutRate">Dropout probability (0 to disable)</param>
    public TransformerBlock(int embedDim, int numHeads, int ffnDim, float dropoutRate = 0.1f)
    {
        _embedDim = embedDim;
        _ffnDim = ffnDim;

        // Multi-head attention
        _attention = new MultiHeadAttention(embedDim, numHeads);
        _norm1 = new LayerNorm(embedDim);

        // Feed-forward network (2-layer MLP)
        _ffn1 = new Linear(embedDim, ffnDim);
        _ffn2 = new Linear(ffnDim, embedDim);
        _norm2 = new LayerNorm(embedDim);

        // Dropout (if enabled)
        if (dropoutRate > 0)
        {
            _dropout = new Dropout(dropoutRate);
        }

        RegisterModule("attention", _attention);
        RegisterModule("norm1", _norm1);
        RegisterModule("ffn1", _ffn1);
        RegisterModule("ffn2", _ffn2);
        RegisterModule("norm2", _norm2);
        if (_dropout != null)
        {
            RegisterModule("dropout", _dropout);
        }
    }

    /// <summary>
    /// Forward pass through the transformer block.
    /// </summary>
    /// <param name="x">Input tensor [batch, seqLen, embedDim]</param>
    /// <param name="attentionMask">Optional attention mask</param>
    /// <returns>Output tensor [batch, seqLen, embedDim]</returns>
    public Tensor<float> Forward(Tensor<float> x, Tensor<float>? attentionMask = null)
    {
        // 1. Multi-Head Attention with residual connection and layer norm
        var attnOutput = _attention.Forward(x, null, null, attentionMask);
        
        if (_dropout != null)
        {
            attnOutput = _dropout.Forward(attnOutput);
        }
        
        // Residual connection + Layer Norm
        x = _norm1.Forward(x + attnOutput);

        // 2. Feed-Forward Network with residual connection and layer norm
        var ffnOutput = _ffn1.Forward(x);
        ffnOutput = ffnOutput.GELU();  // Use GELU activation (modern transformers)
        ffnOutput = _ffn2.Forward(ffnOutput);
        
        if (_dropout != null)
        {
            ffnOutput = _dropout.Forward(ffnOutput);
        }
        
        // Residual connection + Layer Norm
        x = _norm2.Forward(x + ffnOutput);

        return x;
    }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        return Forward(input, null);
    }
}
