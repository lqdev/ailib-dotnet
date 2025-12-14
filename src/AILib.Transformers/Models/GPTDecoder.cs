using AILib;
using AILib.NN;
using AILib.NN.Layers;
using AILib.Transformers.Layers;

namespace AILib.Transformers.Models;

/// <summary>
/// GPT-style transformer decoder for causal language modeling.
/// </summary>
/// <remarks>
/// Architecture:
/// - Token Embedding + Positional Embedding
/// - Stack of Transformer Blocks with causal attention
/// - Final layer norm + output projection
/// </remarks>
public class GPTDecoder : Module
{
    private readonly Embedding _tokenEmbedding;
    private readonly Tensor<float> _positionalEmbedding;
    private readonly List<TransformerBlock> _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _outputProjection;
    private readonly Dropout? _dropout;

    private readonly int _vocabSize;
    private readonly int _embedDim;
    private readonly int _maxSeqLen;
    private readonly int _numLayers;

    /// <summary>
    /// Creates a new GPT decoder.
    /// </summary>
    /// <param name="vocabSize">Size of the vocabulary</param>
    /// <param name="embedDim">Dimension of embeddings</param>
    /// <param name="numHeads">Number of attention heads</param>
    /// <param name="numLayers">Number of transformer blocks</param>
    /// <param name="maxSeqLen">Maximum sequence length</param>
    /// <param name="ffnDim">Feed-forward network dimension (default: 4 * embedDim)</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1)</param>
    public GPTDecoder(
        int vocabSize,
        int embedDim,
        int numHeads,
        int numLayers,
        int maxSeqLen,
        int? ffnDim = null,
        float dropoutRate = 0.1f)
    {
        _vocabSize = vocabSize;
        _embedDim = embedDim;
        _maxSeqLen = maxSeqLen;
        _numLayers = numLayers;

        var actualFfnDim = ffnDim ?? (4 * embedDim);

        // Token embeddings
        _tokenEmbedding = new Embedding(vocabSize, embedDim);
        RegisterModule("token_embedding", _tokenEmbedding);

        // Learned positional embeddings
        _positionalEmbedding = Tensor.Randn<float>([maxSeqLen, embedDim], requiresGrad: true);
        RegisterParameter("positional_embedding", _positionalEmbedding);

        // Dropout
        if (dropoutRate > 0)
        {
            _dropout = new Dropout(dropoutRate);
            RegisterModule("dropout", _dropout);
        }

        // Transformer blocks
        _blocks = new List<TransformerBlock>();
        for (int i = 0; i < numLayers; i++)
        {
            var block = new TransformerBlock(embedDim, numHeads, actualFfnDim, dropoutRate);
            _blocks.Add(block);
            RegisterModule($"block_{i}", block);
        }

        // Final layer norm
        _finalNorm = new LayerNorm(embedDim);
        RegisterModule("final_norm", _finalNorm);

        // Output projection (embedDim -> vocabSize)
        _outputProjection = new Linear(embedDim, vocabSize, bias: false);
        RegisterModule("output_projection", _outputProjection);
    }

    /// <summary>
    /// Forward pass through the decoder.
    /// </summary>
    /// <param name="tokenIndices">Token indices [batch, seqLen]</param>
    /// <returns>Logits over vocabulary [batch, seqLen, vocabSize]</returns>
    public Tensor<float> Forward(Tensor<int> tokenIndices)
    {
        var shape = tokenIndices.Shape.ToArray();
        if (shape.Length != 2)
        {
            throw new ArgumentException("Token indices must be 2D [batch, seqLen]");
        }

        var batchSize = shape[0];
        var seqLen = shape[1];

        if (seqLen > _maxSeqLen)
        {
            throw new ArgumentException($"Sequence length {seqLen} exceeds maximum {_maxSeqLen}");
        }

        // Get token embeddings: [batch, seqLen, embedDim]
        var x = _tokenEmbedding.Forward(tokenIndices);

        // Add positional embeddings
        x = AddPositionalEmbeddings(x, seqLen);

        // Apply dropout
        if (_dropout != null)
        {
            x = _dropout.Forward(x);
        }

        // Create causal attention mask
        var causalMask = AttentionMasks.CreateCausalMask((int)seqLen);

        // Pass through transformer blocks
        foreach (var block in _blocks)
        {
            x = block.Forward(x, causalMask);
        }

        // Final layer norm
        x = _finalNorm.Forward(x);

        // Project to vocabulary
        var logits = _outputProjection.Forward(x);

        return logits;
    }

    /// <summary>
    /// Adds positional embeddings to token embeddings.
    /// </summary>
    private Tensor<float> AddPositionalEmbeddings(Tensor<float> tokenEmbeddings, nint seqLen)
    {
        var shape = tokenEmbeddings.Shape.ToArray();
        var batchSize = shape[0];

        var outputTensor = System.Numerics.Tensors.Tensor.CreateFromShape<float>(shape, false);
        var output = Tensor.FromTensor(outputTensor, requiresGrad: tokenEmbeddings.RequiresGrad);

        // Add position embeddings to each position
        for (nint b = 0; b < batchSize; b++)
        {
            for (nint pos = 0; pos < seqLen; pos++)
            {
                for (nint e = 0; e < _embedDim; e++)
                {
                    output.Data[b, pos, e] = tokenEmbeddings.Data[b, pos, e] + _positionalEmbedding.Data[pos, e];
                }
            }
        }

        return output;
    }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        throw new NotImplementedException(
            "GPTDecoder requires integer token indices. Use Forward(Tensor<int> tokenIndices) instead.");
    }
}
