using AILib;

namespace AILib.Transformers;

/// <summary>
/// Utility functions for creating attention masks.
/// </summary>
public static class AttentionMasks
{
    /// <summary>
    /// Creates a causal (lower triangular) attention mask for autoregressive models.
    /// </summary>
    /// <param name="seqLen">Sequence length</param>
    /// <returns>Mask tensor [seqLen, seqLen] where mask[i,j] = 1 if j <= i, 0 otherwise</returns>
    /// <remarks>
    /// In causal attention, each position can only attend to previous positions and itself.
    /// This is used in autoregressive models like GPT.
    /// </remarks>
    public static Tensor<float> CreateCausalMask(int seqLen)
    {
        var mask = Tensor.Zeros<float>([seqLen, seqLen]);

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                mask.Data[i, j] = 1.0f;
            }
        }

        return mask;
    }

    /// <summary>
    /// Creates a padding mask to ignore padding tokens in attention.
    /// </summary>
    /// <param name="tokens">Token indices where padding tokens are marked</param>
    /// <param name="padTokenId">The token ID used for padding</param>
    /// <returns>Mask tensor where non-padding positions are 1, padding positions are 0</returns>
    public static Tensor<float> CreatePaddingMask(Tensor<int> tokens, int padTokenId)
    {
        var shape = tokens.Shape.ToArray();
        var mask = Tensor.Ones<float>(shape);

        if (shape.Length == 1)
        {
            // 1D: [seqLen]
            for (nint i = 0; i < shape[0]; i++)
            {
                if (tokens.Data[i] == padTokenId)
                {
                    mask.Data[i] = 0.0f;
                }
            }
        }
        else if (shape.Length == 2)
        {
            // 2D: [batch, seqLen]
            for (nint b = 0; b < shape[0]; b++)
            {
                for (nint s = 0; s < shape[1]; s++)
                {
                    if (tokens.Data[b, s] == padTokenId)
                    {
                        mask.Data[b, s] = 0.0f;
                    }
                }
            }
        }

        return mask;
    }

    /// <summary>
    /// Combines causal mask with padding mask for masked language modeling.
    /// </summary>
    /// <param name="seqLen">Sequence length</param>
    /// <param name="paddingMask">Padding mask [batch, seqLen]</param>
    /// <returns>Combined mask [batch, seqLen, seqLen]</returns>
    public static Tensor<float> CombineCausalAndPaddingMask(int seqLen, Tensor<float>? paddingMask = null)
    {
        var causalMask = CreateCausalMask(seqLen);

        if (paddingMask == null)
        {
            return causalMask;
        }

        var paddingShape = paddingMask.Shape.ToArray();
        if (paddingShape.Length != 2)
        {
            throw new ArgumentException("Padding mask must be 2D [batch, seqLen]");
        }

        var batchSize = paddingShape[0];
        var combinedTensor = System.Numerics.Tensors.Tensor.CreateFromShape<float>([batchSize, seqLen, seqLen], false);
        var combinedMask = Tensor.FromTensor(combinedTensor, false);

        // Broadcast causal mask and combine with padding mask
        for (nint b = 0; b < batchSize; b++)
        {
            for (nint i = 0; i < seqLen; i++)
            {
                for (nint j = 0; j < seqLen; j++)
                {
                    // Position is valid if both causal and padding allow it
                    var causalValid = causalMask.Data[i, j];
                    var paddingValid = paddingMask.Data[b, j];
                    combinedMask.Data[b, i, j] = causalValid * paddingValid;
                }
            }
        }

        return combinedMask;
    }
}
