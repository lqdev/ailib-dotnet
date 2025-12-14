using AILib;
using AILib.NN;
using System.Numerics;

namespace AILib.Transformers.Layers;

/// <summary>
/// Embedding layer that maps discrete token indices to continuous vectors.
/// </summary>
/// <remarks>
/// This layer maintains a lookup table of embeddings with shape [vocabSize, embeddingDim].
/// Given token indices, it returns the corresponding embedding vectors.
/// </remarks>
public class Embedding : Module
{
    private readonly Tensor<float> _weight;
    private readonly int _vocabSize;
    private readonly int _embeddingDim;

    /// <summary>
    /// Creates a new embedding layer.
    /// </summary>
    /// <param name="vocabSize">Size of the vocabulary (number of unique tokens)</param>
    /// <param name="embeddingDim">Dimension of the embedding vectors</param>
    public Embedding(int vocabSize, int embeddingDim)
    {
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;

        // Initialize embeddings with N(0, 1)
        _weight = Tensor.Randn<float>([vocabSize, embeddingDim], requiresGrad: true);
        RegisterParameter("weight", _weight);
    }

    /// <summary>
    /// Forward pass: looks up embeddings for the given indices.
    /// </summary>
    /// <param name="indices">Tensor of token indices with shape [batch, seqLen] or [seqLen]</param>
    /// <returns>Embeddings with shape [batch, seqLen, embeddingDim] or [seqLen, embeddingDim]</returns>
    public Tensor<float> Forward(Tensor<int> indices)
    {
        // Get the shape of indices
        var indicesShape = indices.Shape.ToArray();
        var isBatched = indicesShape.Length == 2;
        
        nint batchSize = isBatched ? indicesShape[0] : 1;
        nint seqLen = isBatched ? indicesShape[1] : indicesShape[0];

        // Create output tensor
        var outputShape = isBatched 
            ? new nint[] { batchSize, seqLen, _embeddingDim }
            : new nint[] { seqLen, _embeddingDim };
        
        var outputTensor = System.Numerics.Tensors.Tensor.CreateFromShape<float>(outputShape, false);
        var output = Tensor.FromTensor(outputTensor, requiresGrad: _weight.RequiresGrad);

        // Gather embeddings for each index
        if (isBatched)
        {
            for (nint b = 0; b < batchSize; b++)
            {
                for (nint s = 0; s < seqLen; s++)
                {
                    var idx = indices.Data[b, s];
                    if (idx < 0 || idx >= _vocabSize)
                    {
                        throw new ArgumentOutOfRangeException($"Index {idx} out of range [0, {_vocabSize})");
                    }
                    
                    // Copy the embedding vector
                    for (nint e = 0; e < _embeddingDim; e++)
                    {
                        output.Data[b, s, e] = _weight.Data[idx, e];
                    }
                }
            }
        }
        else
        {
            for (nint s = 0; s < seqLen; s++)
            {
                var idx = indices.Data[s];
                if (idx < 0 || idx >= _vocabSize)
                {
                    throw new ArgumentOutOfRangeException($"Index {idx} out of range [0, {_vocabSize})");
                }
                
                // Copy the embedding vector
                for (nint e = 0; e < _embeddingDim; e++)
                {
                    output.Data[s, e] = _weight.Data[idx, e];
                }
            }
        }

        // TODO: Add gradient function for embedding lookup
        // For now, embeddings can be updated through direct parameter optimization
        
        return output;
    }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        throw new NotImplementedException(
            "Embedding layer requires integer indices. Use Forward(Tensor<int> indices) instead.");
    }
}
