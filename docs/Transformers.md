# Transformer Layers Guide

AILib.Transformers provides transformer components for building modern language models and sequence-to-sequence architectures.

## Table of Contents

- [Overview](#overview)
- [Embedding Layer](#embedding-layer)
- [Multi-Head Attention](#multi-head-attention)
- [Transformer Block](#transformer-block)
- [GPT Decoder](#gpt-decoder)
- [Attention Masks](#attention-masks)
- [Known Limitations](#known-limitations)

## Overview

The transformer architecture, introduced in "Attention is All You Need", has become the foundation for modern NLP models like GPT, BERT, and LLaMA. AILib.Transformers implements the core components needed to build these models.

**Key Components**:
- `Embedding`: Maps token indices to continuous vectors
- `MultiHeadAttention`: Scaled dot-product attention across multiple heads
- `TransformerBlock`: Complete encoder/decoder block with attention + FFN
- `GPTDecoder`: Full GPT-style decoder for causal language modeling
- `AttentionMasks`: Utilities for creating causal and padding masks

## Embedding Layer

Maps discrete token indices to continuous embedding vectors.

### Usage

```csharp
using AILib.Transformers.Layers;

// Create embedding layer
var embedding = new Embedding(
    vocabSize: 50000,     // Vocabulary size
    embeddingDim: 512     // Embedding dimension
);

// Single sequence
var tokens = Tensor.FromArray(new int[] { 1, 5, 10, 3 }, [4]);
var embeddings = embedding.Forward(tokens);
// Output shape: [4, 512] - [seqLen, embeddingDim]

// Batched sequences
var batchedTokens = Tensor.FromArray(new int[] { 
    1, 5, 10, 3,
    2, 6, 11, 4 
}, [2, 4]);
var batchedEmbeddings = embedding.Forward(batchedTokens);
// Output shape: [2, 4, 512] - [batch, seqLen, embeddingDim]
```

### Properties

- **Initialization**: Embeddings initialized with N(0, 1)
- **Trainable**: Embedding weights are registered as parameters
- **Lookup**: Simple index-based lookup into embedding table

### Example: Word Embeddings

```csharp
// Create small vocabulary
var vocab = new Embedding(vocabSize: 1000, embeddingDim: 128);

// Token indices for "hello world"
var tokens = Tensor.FromArray(new int[] { 42, 100 }, [2]);

// Get embeddings
var wordVectors = vocab.Forward(tokens);
// Shape: [2, 128]

Console.WriteLine($"Embedding for 'hello': shape {wordVectors.Shape[0]}x{wordVectors.Shape[1]}");
```

## Multi-Head Attention

Implements scaled dot-product attention across multiple heads in parallel.

### Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Multi-head attention splits Q, K, V into multiple heads, computes attention independently, then concatenates results.

### Usage

```csharp
using AILib.Transformers.Layers;

// Create multi-head attention
var mha = new MultiHeadAttention(
    embedDim: 512,      // Total model dimension
    numHeads: 8,        // Number of attention heads
    bias: true          // Include bias in projections
);

// Self-attention: Q = K = V
var x = Tensor.Randn<float>([2, 10, 512]);  // [batch, seqLen, embedDim]
var output = mha.Forward(x);
// Output shape: [2, 10, 512]

// Cross-attention: different K, V
var query = Tensor.Randn<float>([2, 10, 512]);
var key = Tensor.Randn<float>([2, 20, 512]);
var value = Tensor.Randn<float>([2, 20, 512]);
var crossAttnOutput = mha.Forward(query, key, value);
// Output shape: [2, 10, 512]
```

### With Attention Mask

```csharp
using AILib.Transformers;

// Create causal mask for autoregressive generation
var causalMask = AttentionMasks.CreateCausalMask(seqLen: 10);

var mha = new MultiHeadAttention(embedDim: 512, numHeads: 8);
var x = Tensor.Randn<float>([2, 10, 512]);

// Apply causal attention (each position attends only to previous positions)
var output = mha.Forward(x, attentionMask: causalMask);
```

### Parameters

- `embedDim`: Total dimension of the model (must be divisible by `numHeads`)
- `numHeads`: Number of parallel attention heads
- `bias`: Whether to include bias terms in Q/K/V/O projections

### Architecture

Each head operates on dimension `d_k = embedDim / numHeads`:

```
Input [batch, seqLen, embedDim]
  ↓
Q, K, V projections [batch, seqLen, embedDim]
  ↓
Reshape to [batch, numHeads, seqLen, d_k]
  ↓
Attention scores: Q @ K^T / √d_k
  ↓
Softmax + apply to values
  ↓
Concat heads → output projection
  ↓
Output [batch, seqLen, embedDim]
```

## Transformer Block

Complete transformer encoder/decoder block combining multi-head attention with a feed-forward network.

### Architecture

```
Input
  ↓
MultiHeadAttention
  ↓
Add & Norm (residual connection + LayerNorm)
  ↓
Feed-Forward Network (Linear → GELU → Linear)
  ↓
Add & Norm
  ↓
Output
```

### Usage

```csharp
using AILib.Transformers.Layers;

// Create transformer block
var block = new TransformerBlock(
    embedDim: 512,       // Model dimension
    numHeads: 8,         // Attention heads
    ffnDim: 2048,        // FFN hidden dimension (typically 4 * embedDim)
    dropoutRate: 0.1f    // Dropout probability
);

// Forward pass
var x = Tensor.Randn<float>([2, 10, 512]);
var output = block.Forward(x);
// Output shape: [2, 10, 512]

// With causal mask
var causalMask = AttentionMasks.CreateCausalMask(10);
var maskedOutput = block.Forward(x, causalMask);
```

### Parameters

- `embedDim`: Dimension of the model
- `numHeads`: Number of attention heads
- `ffnDim`: Hidden dimension of feed-forward network (typically `4 * embedDim`)
- `dropoutRate`: Dropout probability (0 to disable)

### Components

1. **Multi-Head Self-Attention**: Computes attention over the input sequence
2. **Residual Connection + LayerNorm**: Stabilizes training
3. **Feed-Forward Network**: Two-layer MLP with GELU activation
4. **Residual Connection + LayerNorm**: Second normalization

## GPT Decoder

Full GPT-style transformer decoder for causal language modeling.

### Architecture

```
Token Indices [batch, seqLen]
  ↓
Token Embedding + Positional Embedding
  ↓
Dropout
  ↓
TransformerBlock 1
  ↓
TransformerBlock 2
  ↓
...
  ↓
TransformerBlock N
  ↓
Final LayerNorm
  ↓
Output Projection
  ↓
Logits [batch, seqLen, vocabSize]
```

### Usage

```csharp
using AILib.Transformers.Models;

// Create GPT decoder
var model = new GPTDecoder(
    vocabSize: 50000,    // Vocabulary size
    embedDim: 768,       // Model dimension
    numHeads: 12,        // Attention heads
    numLayers: 12,       // Number of transformer blocks
    maxSeqLen: 1024,     // Maximum sequence length
    ffnDim: 3072,        // FFN dimension (default: 4 * embedDim)
    dropoutRate: 0.1f    // Dropout rate
);

// Forward pass
var tokens = Tensor.FromArray(new int[] { 
    1, 5, 10, 3,
    2, 6, 11, 4 
}, [2, 4]);  // [batch, seqLen]

var logits = model.Forward(tokens);
// Output shape: [2, 4, 50000] - [batch, seqLen, vocabSize]

// Get predictions for next token
// Take logits at last position: logits[:, -1, :]
// Apply softmax to get probabilities
// Sample or argmax to get next token
```

### Configuration Examples

**Tiny GPT (for testing)**:
```csharp
var tinyGPT = new GPTDecoder(
    vocabSize: 1000,
    embedDim: 128,
    numHeads: 4,
    numLayers: 4,
    maxSeqLen: 256
);
```

**GPT-2 Small**:
```csharp
var gpt2Small = new GPTDecoder(
    vocabSize: 50257,
    embedDim: 768,
    numHeads: 12,
    numLayers: 12,
    maxSeqLen: 1024
);
```

**GPT-2 Medium**:
```csharp
var gpt2Medium = new GPTDecoder(
    vocabSize: 50257,
    embedDim: 1024,
    numHeads: 16,
    numLayers: 24,
    maxSeqLen: 1024
);
```

### Features

- **Learned Positional Embeddings**: Unlike sinusoidal, uses trainable position embeddings
- **Causal Attention**: Automatic masking prevents attending to future tokens
- **GELU Activation**: Modern activation function used in GPT models
- **Pre-Norm Architecture**: LayerNorm before attention/FFN (more stable training)

## Attention Masks

Utilities for creating attention masks.

### Causal Mask

Prevents positions from attending to future positions (autoregressive models).

```csharp
using AILib.Transformers;

// Create causal mask
var mask = AttentionMasks.CreateCausalMask(seqLen: 5);

// Result: lower triangular matrix
// [[1, 0, 0, 0, 0],
//  [1, 1, 0, 0, 0],
//  [1, 1, 1, 0, 0],
//  [1, 1, 1, 1, 0],
//  [1, 1, 1, 1, 1]]
```

### Padding Mask

Ignores padding tokens in attention.

```csharp
// Tokens with padding (0 = pad token)
var tokens = Tensor.FromArray(new int[] { 
    1, 5, 10, 0, 0 
}, [5]);

var paddingMask = AttentionMasks.CreatePaddingMask(tokens, padTokenId: 0);
// Result: [1, 1, 1, 0, 0]
```

### Combined Mask

Combines causal and padding masks.

```csharp
// Batch of sequences with padding
var tokens = Tensor.FromArray(new int[] { 
    1, 5, 10, 0,
    2, 6, 0, 0
}, [2, 4]);

var paddingMask = AttentionMasks.CreatePaddingMask(tokens, padTokenId: 0);
var combinedMask = AttentionMasks.CombineCausalAndPaddingMask(
    seqLen: 4,
    paddingMask: paddingMask
);
// Result: [batch, seqLen, seqLen] with both causal and padding constraints
```

## Complete Example: Simple Language Model

```csharp
using AILib;
using AILib.Transformers;
using AILib.Transformers.Models;
using AILib.NN.Optimizers;
using AILib.NN.Loss;

// Create model
var model = new GPTDecoder(
    vocabSize: 1000,
    embedDim: 256,
    numHeads: 4,
    numLayers: 4,
    maxSeqLen: 128,
    dropoutRate: 0.1f
);

// Training setup
var optimizer = new AdamW(model.Parameters(), lr: 3e-4f, weightDecay: 0.01f);
var lossFunc = new CrossEntropyLoss();

// Training loop
for (int step = 0; step < 1000; step++)
{
    // Get batch of token sequences
    // In practice, load from dataset
    var inputTokens = Tensor.FromArray(new int[16 * 32], [16, 32]);  // [batch, seqLen]
    
    // Forward pass
    var logits = model.Forward(inputTokens);  // [batch, seqLen, vocabSize]
    
    // Compute loss
    // Shift logits and targets for next-token prediction
    // loss = criterion(logits[:, :-1], targets[:, 1:])
    
    // Note: Full training loop implementation requires:
    // - Data loading and batching
    // - Learning rate scheduling
    // - Gradient clipping
    // - Checkpointing
    
    if (step % 100 == 0)
    {
        Console.WriteLine($"Step {step}: Training...");
    }
}
```

## Known Limitations

### Current Implementation

1. **3D Tensor Support**: Linear layers currently only support 2D tensors. For transformer operations:
   - Input: `[batch, seqLen, embedDim]`
   - Linear expects: `[batch * seqLen, embedDim]` (flattened)
   - Need to reshape before/after Linear operations

2. **Attention Implementation**: Uses manual batched matmul instead of leveraging optimized tensor operations.

3. **Gradient Flow**: Autograd support for embedding lookups and complex attention patterns is limited.

### Workarounds

For now, when using transformers:
- Start with small batch sizes and sequence lengths
- Test shape transformations carefully
- Use the provided test cases as reference

### Future Improvements

- Optimized batched matmul using `System.Numerics.Tensors`
- Full autograd support for all operations
- Flash Attention for memory efficiency
- KV caching for efficient generation
- Rotary positional embeddings (RoPE)

## Best Practices

1. **Start Small**: Begin with tiny models for testing (embedDim=128, numLayers=2)

2. **Check Shapes**: Always verify tensor shapes match expected dimensions

3. **Use Masks**: Apply causal masks for autoregressive generation

4. **Monitor Memory**: Large models with long sequences require significant memory

5. **Optimize Hyperparameters**:
   - `embedDim`: Typically powers of 2 (256, 512, 768, 1024)
   - `numHeads`: Should divide `embedDim` evenly (8, 12, 16)
   - `ffnDim`: Usually `4 * embedDim`
   - `dropoutRate`: 0.1 is a good default

## Next Steps

- Learn about [Neural Network Layers](Layers.md)
- Explore [Optimizers](Optimizers.md) for training
- See [Loss Functions](LossFunctions.md) for different tasks
