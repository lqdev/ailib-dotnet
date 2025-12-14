using AILib;
using AILib.Transformers;
using AILib.Transformers.Layers;
using AILib.Transformers.Models;

namespace AILib.Transformers.Tests;

public class EmbeddingTests
{
    [Fact]
    public void Embedding_CreatesCorrectShape()
    {
        var embedding = new Embedding(vocabSize: 100, embeddingDim: 64);
        var indices = Tensor.FromArray(new int[] { 1, 5, 10 }, [3]);
        
        var output = embedding.Forward(indices);
        
        Assert.Equal(2, output.Rank);
        Assert.Equal(3, output.Shape[0]);  // seqLen
        Assert.Equal(64, output.Shape[1]); // embeddingDim
    }

    [Fact]
    public void Embedding_BatchedInput()
    {
        var embedding = new Embedding(vocabSize: 100, embeddingDim: 64);
        var indices = Tensor.FromArray(new int[] { 1, 5, 10, 2, 6, 11 }, [2, 3]);
        
        var output = embedding.Forward(indices);
        
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]);  // batch
        Assert.Equal(3, output.Shape[1]);  // seqLen
        Assert.Equal(64, output.Shape[2]); // embeddingDim
    }

    [Fact]
    public void Embedding_ThrowsOnInvalidIndex()
    {
        var embedding = new Embedding(vocabSize: 100, embeddingDim: 64);
        var indices = Tensor.FromArray(new int[] { 1, 150 }, [2]);  // 150 > vocabSize
        
        Assert.Throws<ArgumentOutOfRangeException>(() => embedding.Forward(indices));
    }
}

public class AttentionMasksTests
{
    [Fact]
    public void CreateCausalMask_ProducesLowerTriangular()
    {
        var mask = AttentionMasks.CreateCausalMask(4);
        
        Assert.Equal(2, mask.Rank);
        Assert.Equal(4, mask.Shape[0]);
        Assert.Equal(4, mask.Shape[1]);
        
        // Check lower triangular property
        Assert.Equal(1.0f, mask.Data[0, 0]);
        Assert.Equal(0.0f, mask.Data[0, 1]);
        Assert.Equal(0.0f, mask.Data[0, 2]);
        
        Assert.Equal(1.0f, mask.Data[1, 0]);
        Assert.Equal(1.0f, mask.Data[1, 1]);
        Assert.Equal(0.0f, mask.Data[1, 2]);
        
        Assert.Equal(1.0f, mask.Data[2, 0]);
        Assert.Equal(1.0f, mask.Data[2, 1]);
        Assert.Equal(1.0f, mask.Data[2, 2]);
        Assert.Equal(0.0f, mask.Data[2, 3]);
    }

    [Fact]
    public void CreatePaddingMask_MarksNonPaddingPositions()
    {
        var tokens = Tensor.FromArray(new int[] { 1, 2, 0, 0 }, [4]);  // 0 is pad token
        var mask = AttentionMasks.CreatePaddingMask(tokens, padTokenId: 0);
        
        Assert.Equal(1.0f, mask.Data[0]);  // Non-padding
        Assert.Equal(1.0f, mask.Data[1]);  // Non-padding
        Assert.Equal(0.0f, mask.Data[2]);  // Padding
        Assert.Equal(0.0f, mask.Data[3]);  // Padding
    }
}

public class MultiHeadAttentionTests
{
    [Fact]
    public void MultiHeadAttention_OutputShapeCorrect()
    {
        var mha = new MultiHeadAttention(embedDim: 64, numHeads: 4);
        var input = Tensor.Randn<float>([2, 10, 64]);  // [batch, seqLen, embedDim]
        
        var output = mha.Forward(input);
        
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]);   // batch
        Assert.Equal(10, output.Shape[1]);  // seqLen
        Assert.Equal(64, output.Shape[2]);  // embedDim
    }

    [Fact]
    public void MultiHeadAttention_ThrowsOnInvalidEmbedDim()
    {
        // embedDim must be divisible by numHeads
        Assert.Throws<ArgumentException>(() => 
            new MultiHeadAttention(embedDim: 65, numHeads: 4));
    }
}

public class TransformerBlockTests
{
    [Fact]
    public void TransformerBlock_OutputShapeCorrect()
    {
        var block = new TransformerBlock(
            embedDim: 64,
            numHeads: 4,
            ffnDim: 256,
            dropoutRate: 0.1f
        );
        
        var input = Tensor.Randn<float>([2, 10, 64]);
        var output = block.Forward(input);
        
        Assert.Equal(3, output.Rank);
        Assert.Equal(2, output.Shape[0]);   // batch
        Assert.Equal(10, output.Shape[1]);  // seqLen
        Assert.Equal(64, output.Shape[2]);  // embedDim
    }
}

public class GPTDecoderTests
{
    [Fact]
    public void GPTDecoder_OutputShapeCorrect()
    {
        var decoder = new GPTDecoder(
            vocabSize: 1000,
            embedDim: 64,
            numHeads: 4,
            numLayers: 2,
            maxSeqLen: 128
        );
        
        var tokens = Tensor.FromArray(new int[] { 1, 5, 10, 2, 6, 11 }, [2, 3]);
        var logits = decoder.Forward(tokens);
        
        Assert.Equal(3, logits.Rank);
        Assert.Equal(2, logits.Shape[0]);     // batch
        Assert.Equal(3, logits.Shape[1]);     // seqLen
        Assert.Equal(1000, logits.Shape[2]);  // vocabSize
    }

    [Fact]
    public void GPTDecoder_ThrowsOnTooLongSequence()
    {
        var decoder = new GPTDecoder(
            vocabSize: 1000,
            embedDim: 64,
            numHeads: 4,
            numLayers: 2,
            maxSeqLen: 10
        );
        
        var tokens = Tensor.FromArray(new int[20], [1, 20]);  // 20 > maxSeqLen
        
        Assert.Throws<ArgumentException>(() => decoder.Forward(tokens));
    }
}
