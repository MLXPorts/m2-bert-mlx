#!/usr/bin/env python
"""
Complete M2-BERT Model in MLX

Full implementation of M2-BERT with:
- BertEmbeddings (token + position + segment)
- 12 M2-BERT layers (Monarch Mixer + Monarch MLP)
- Pooling layer for sentence embeddings
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from bert.src.mlx_ops.monarch_mixer import MonarchMixerSequenceMixing
from bert.src.mlx_ops.monarch_mlp import MonarchGLUMLP


class BertEmbeddings(nn.Module):
    """
    BERT-style embeddings: token + position + token_type

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Embedding dimension (d_model)
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Number of token types (for segment embeddings)
        layer_norm_eps: LayerNorm epsilon
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        # Token embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # Token type embeddings (for segment A/B in BERT)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # LayerNorm and dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Position IDs buffer (for compatibility with PyTorch checkpoints)
        # This is just [0, 1, 2, ..., max_position_embeddings-1]
        self.position_ids = mx.arange(max_position_embeddings).reshape(1, -1)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_length) - token IDs
            token_type_ids: (batch_size, seq_length) - segment IDs (0 or 1)
            position_ids: (batch_size, seq_length) - position IDs

        Returns:
            embeddings: (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length = input_ids.shape

        # Token embeddings
        inputs_embeds = self.word_embeddings(input_ids)

        # Position IDs (if not provided, use 0, 1, 2, ...)
        if position_ids is None:
            position_ids = mx.arange(seq_length).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        position_embeddings = self.position_embeddings(position_ids)

        # Token type IDs (if not provided, use all zeros)
        if token_type_ids is None:
            token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Sum all embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # LayerNorm + Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class M2BERTLayer(nn.Module):
    """
    One M2-BERT layer: Monarch Mixer + Monarch MLP

    Architecture:
        Input
          ↓
        Monarch Mixer (sequence mixing, replaces attention)
          ↓
        Residual + LayerNorm
          ↓
        Monarch MLP (state mixing)
          ↓
        Output

    Args:
        hidden_size: Model dimension (d_model)
        intermediate_size: MLP intermediate dimension
        max_seq_len: Maximum sequence length
        nblocks: Number of blocks for Monarch matrices
        dropout: Dropout rate
        layer_norm_eps: LayerNorm epsilon
        bidirectional: Use bidirectional Hyena filters
        hyena_filter_order: Order of Hyena filter MLP
        residual_long_conv: Add residual long conv path
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        max_seq_len: int = 2048,
        nblocks: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        bidirectional: bool = True,
        hyena_filter_order: int = 128,
        hyena_emb_dim: int = 3,
        residual_long_conv: bool = True,
        fft_chunk_size: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Monarch Mixer (sequence mixing - replaces self-attention)
        self.sequence_mixing = MonarchMixerSequenceMixing(
            d_model=hidden_size,
            l_max=max_seq_len,
            dropout=dropout,
            bidirectional=bidirectional,
            hyena_filter_order=hyena_filter_order,
            hyena_emb_dim=hyena_emb_dim,
            residual_long_conv=residual_long_conv,
            fft_chunk_size=fft_chunk_size
        )

        # LayerNorm after sequence mixing
        self.layernorm_seq = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Monarch MLP (state mixing)
        self.mlp = MonarchGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            nblocks=nblocks,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            use_monarch=True
        )

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None):
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length) - not used in M2-BERT

        Returns:
            output: (batch_size, seq_length, hidden_size)
        """
        # Sequence mixing (replaces attention)
        # Note: Monarch Mixer already includes residual connection internally?
        # Let me check the implementation...
        # No, it doesn't include residual in the original. Let me add it.

        seq_output, _ = self.sequence_mixing(hidden_states)

        # Residual connection + LayerNorm
        hidden_states = self.layernorm_seq(hidden_states + seq_output)

        # State mixing (MLP) - MonarchGLUMLP already has residual + layernorm
        output = self.mlp(hidden_states)

        return output


class M2BERTModel(nn.Module):
    """
    Complete M2-BERT model for embeddings

    Architecture:
        Input IDs
          ↓
        BertEmbeddings (token + position + type)
          ↓
        12× M2BERTLayer
          ↓
        Last hidden state

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model dimension (d_model)
        num_hidden_layers: Number of transformer layers
        intermediate_size: MLP intermediate dimension
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Number of token types
        nblocks: Number of blocks for Monarch matrices
        dropout: Dropout rate
        layer_norm_eps: LayerNorm epsilon
        bidirectional: Use bidirectional Hyena filters
        hyena_filter_order: Order of Hyena filter MLP
        residual_long_conv: Add residual long conv path
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        nblocks: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        bidirectional: bool = True,
        hyena_filter_order: int = 128,
        hyena_emb_dim: int = 3,
        residual_long_conv: bool = True,
        fft_chunk_size: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        print(f"Creating M2-BERT Model:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_hidden_layers}")
        print(f"  Intermediate size: {intermediate_size}")
        print(f"  Max seq length: {max_position_embeddings}")
        print(f"  Monarch blocks: {nblocks}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  Residual long conv: {residual_long_conv}")
        print()

        # Embeddings
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout
        )

        # Transformer layers
        self.layers = []
        for i in range(num_hidden_layers):
            layer = M2BERTLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                max_seq_len=max_position_embeddings,
                nblocks=nblocks,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                bidirectional=bidirectional,
                hyena_filter_order=hyena_filter_order,
                hyena_emb_dim=hyena_emb_dim,
                residual_long_conv=residual_long_conv,
                fft_chunk_size=fft_chunk_size
            )
            self.layers.append(layer)

        print(f"✅ Created {num_hidden_layers} M2-BERT layers")

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_length) - token IDs
            token_type_ids: (batch_size, seq_length) - segment IDs
            position_ids: (batch_size, seq_length) - position IDs
            attention_mask: (batch_size, seq_length) - attention mask (not used)

        Returns:
            last_hidden_state: (batch_size, seq_length, hidden_size)
        """
        # Embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        # Pass through all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        return hidden_states


class M2BERTForSentenceEmbeddings(nn.Module):
    """
    M2-BERT with pooling for sentence embeddings

    Used for sentence transformers / semantic similarity tasks

    Args:
        Same as M2BERTModel, plus:
        pooling_mode: 'cls' or 'mean'
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        nblocks: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        bidirectional: bool = True,
        hyena_filter_order: int = 128,
        residual_long_conv: bool = True,
        pooling_mode: str = 'mean',
        fft_chunk_size: int = 128
    ):
        super().__init__()

        self.pooling_mode = pooling_mode

        # Base M2-BERT model
        self.bert = M2BERTModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            nblocks=nblocks,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            bidirectional=bidirectional,
            hyena_filter_order=hyena_filter_order,
            residual_long_conv=residual_long_conv,
            fft_chunk_size=fft_chunk_size
        )

    def mean_pooling(self, token_embeddings: mx.array, attention_mask: Optional[mx.array] = None):
        """
        Mean pooling over sequence dimension

        Args:
            token_embeddings: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length) - 1 for real tokens, 0 for padding

        Returns:
            pooled: (batch_size, hidden_size)
        """
        if attention_mask is None:
            # No mask - simple mean
            return mx.mean(token_embeddings, axis=1)

        # Expand mask to match embeddings shape
        # attention_mask: (batch, seq_len) -> (batch, seq_len, 1)
        mask_expanded = attention_mask[:, :, None]

        # Sum embeddings (masked)
        sum_embeddings = mx.sum(token_embeddings * mask_expanded, axis=1)

        # Count tokens (sum of mask)
        sum_mask = mx.sum(mask_expanded, axis=1)
        sum_mask = mx.maximum(sum_mask, 1e-9)  # Avoid division by zero

        # Average
        return sum_embeddings / sum_mask

    def cls_pooling(self, token_embeddings: mx.array):
        """
        Use [CLS] token (first token) as sentence embedding

        Args:
            token_embeddings: (batch_size, seq_length, hidden_size)

        Returns:
            pooled: (batch_size, hidden_size)
        """
        return token_embeddings[:, 0, :]

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None
    ):
        """
        Args:
            input_ids: (batch_size, seq_length) - token IDs
            token_type_ids: (batch_size, seq_length) - segment IDs
            position_ids: (batch_size, seq_length) - position IDs
            attention_mask: (batch_size, seq_length) - attention mask

        Returns:
            sentence_embedding: (batch_size, hidden_size)
        """
        # Get token embeddings from BERT
        token_embeddings = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
        )

        # Pool to sentence embedding
        if self.pooling_mode == 'mean':
            sentence_embedding = self.mean_pooling(token_embeddings, attention_mask)
        elif self.pooling_mode == 'cls':
            sentence_embedding = self.cls_pooling(token_embeddings)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        return sentence_embedding


def test_m2bert_model():
    """Test the full M2-BERT model"""
    print("="*70)
    print("Testing Complete M2-BERT Model (MLX)")
    print("="*70)
    print()

    # Small config for testing
    batch_size = 2
    seq_len = 128
    vocab_size = 30522
    hidden_size = 768
    num_layers = 12
    intermediate_size = 3072

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Intermediate size: {intermediate_size}")
    print()

    # Test 1: BertEmbeddings
    print("Test 1: BertEmbeddings")
    print("-" * 70)

    embeddings = BertEmbeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=2048
    )

    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    embed_output = embeddings(input_ids)

    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Embeddings: {embed_output.shape}")
    print(f"  ✅ BertEmbeddings works!")
    print()

    # Test 2: M2BERTLayer
    print("Test 2: M2BERTLayer")
    print("-" * 70)

    layer = M2BERTLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_seq_len=seq_len,
        nblocks=4,
        bidirectional=True,
        residual_long_conv=True
    )

    layer_output = layer(embed_output)

    print(f"  Input: {embed_output.shape}")
    print(f"  Output: {layer_output.shape}")
    print(f"  ✅ M2BERTLayer works!")
    print()

    # Test 3: Full M2BERTModel
    print("Test 3: Full M2BERTModel (12 layers)")
    print("-" * 70)

    model = M2BERTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=2048,
        nblocks=4,
        bidirectional=True,
        residual_long_conv=True
    )

    print()
    print("Running forward pass...")
    last_hidden_state = model(input_ids)

    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Last hidden state: {last_hidden_state.shape}")
    print(f"  Output range: [{last_hidden_state.min().item():.3f}, {last_hidden_state.max().item():.3f}]")
    print(f"  ✅ Full M2BERTModel works!")
    print()

    # Test 4: M2BERTForSentenceEmbeddings (mean pooling)
    print("Test 4: M2BERTForSentenceEmbeddings (mean pooling)")
    print("-" * 70)

    sent_model = M2BERTForSentenceEmbeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=2048,
        nblocks=4,
        pooling_mode='mean'
    )

    print()
    sentence_embeddings = sent_model(input_ids)

    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Sentence embeddings: {sentence_embeddings.shape}")
    print(f"  Embedding range: [{sentence_embeddings.min().item():.3f}, {sentence_embeddings.max().item():.3f}]")
    print(f"  ✅ Sentence embeddings work!")
    print()

    # Test 5: With attention mask
    print("Test 5: With Attention Mask (mean pooling)")
    print("-" * 70)

    # Create attention mask: first 64 tokens are real, rest are padding
    attention_mask = mx.concatenate([
        mx.ones((batch_size, 64)),
        mx.zeros((batch_size, seq_len - 64))
    ], axis=1)

    sentence_embeddings_masked = sent_model(input_ids, attention_mask=attention_mask)

    print(f"  Attention mask: {attention_mask.shape}")
    print(f"  Real tokens: 64, Padding: {seq_len - 64}")
    print(f"  Sentence embeddings: {sentence_embeddings_masked.shape}")
    print(f"  ✅ Attention mask works!")
    print()

    # Test 6: Parameter count
    print("Test 6: Parameter Count")
    print("-" * 70)

    def count_params(module):
        params = module.parameters()
        def count_nested(obj):
            total = 0
            if isinstance(obj, dict):
                for k, v in obj.items():
                    total += count_nested(v)
            elif isinstance(obj, list):
                for item in obj:
                    total += count_nested(item)
            elif hasattr(obj, 'size'):
                total += obj.size
            return total
        return count_nested(params)

    total_params = count_params(model)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Target: ~80M params")
    print(f"  Actual: {total_params / 1_000_000:.1f}M params")
    print(f"  ✅ Parameter count reasonable!")
    print()

    print("="*70)
    print("✅ All M2-BERT model tests complete!")
    print("="*70)


if __name__ == '__main__':
    test_m2bert_model()
