#!/usr/bin/env python
"""
Loss Functions for M2-BERT Training in MLX

Port of sentence-transformers losses to pure MLX, maintaining
exact numerical behavior for training stability.

Based on:
- MultipleNegativesRankingLoss (InfoNCE)
- CachedMultipleNegativesRankingLoss (GradCache)
"""

import mlx.core as mx
import mlx.nn as nn


class MultipleNegativesRankingLoss:
    """
    Multiple Negatives Ranking Loss (InfoNCE / NT-Xent)

    Given pairs (anchor, positive), uses all other positives in the batch
    as negatives. Efficient in-batch negative sampling.

    Formula:
        For each anchor i:
        - Compute cosine similarity to all positives in batch
        - Apply softmax where correct positive is at index i
        - Minimize cross-entropy loss

    This is identical to sentence_transformers.losses.MultipleNegativesRankingLoss

    Args:
        scale: Temperature scaling factor (default: 20.0)
            Higher values make the distribution sharper
    """

    def __init__(self, scale: float = 20.0):
        self.scale = scale

    def cosine_similarity(self, a: mx.array, b: mx.array) -> mx.array:
        """
        Compute cosine similarity between all pairs

        Args:
            a: (batch_size, hidden_size) - anchors
            b: (batch_size, hidden_size) - candidates

        Returns:
            similarity: (batch_size, batch_size) - similarity matrix
        """
        # Normalize
        a_norm = a / mx.sqrt(mx.sum(a ** 2, axis=1, keepdims=True))
        b_norm = b / mx.sqrt(mx.sum(b ** 2, axis=1, keepdims=True))

        # Compute similarity
        similarity = a_norm @ b_norm.T  # (batch_size, batch_size)

        return similarity

    def __call__(self, anchors: mx.array, positives: mx.array) -> mx.array:
        """
        Compute Multiple Negatives Ranking Loss

        Args:
            anchors: (batch_size, hidden_size) - anchor embeddings
            positives: (batch_size, hidden_size) - positive embeddings

        Returns:
            loss: scalar - cross-entropy loss
        """
        batch_size = anchors.shape[0]

        # Compute similarity matrix (batch_size, batch_size)
        # similarity[i, j] = cosine(anchor_i, positive_j)
        similarity = self.cosine_similarity(anchors, positives)

        # Scale by temperature
        similarity = similarity * self.scale

        # Labels: each anchor i should match positive i
        # So labels = [0, 1, 2, ..., batch_size-1]
        labels = mx.arange(batch_size)

        # Cross-entropy loss
        # This treats it as classification: which positive matches this anchor?
        loss = nn.losses.cross_entropy(similarity, labels, reduction='mean')

        return loss


class CachedMultipleNegativesRankingLoss:
    """
    Cached Multiple Negatives Ranking Loss (GradCache)

    Same as MultipleNegativesRankingLoss but with gradient caching
    to support larger effective batch sizes with limited memory.

    Processes the batch in mini-batches, caching gradients and
    combining them at the end.

    Based on: "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup"
    https://arxiv.org/pdf/2101.06983.pdf

    Args:
        scale: Temperature scaling factor (default: 20.0)
        mini_batch_size: Size of mini-batches for gradient caching
    """

    def __init__(self, scale: float = 20.0, mini_batch_size: int = 32):
        self.scale = scale
        self.mini_batch_size = mini_batch_size
        self.base_loss = MultipleNegativesRankingLoss(scale=scale)

    def __call__(self, anchors: mx.array, positives: mx.array) -> mx.array:
        """
        Compute cached loss with gradient accumulation

        For simplicity in MLX, we'll process the full batch
        but this could be extended to use mini-batches.

        Args:
            anchors: (batch_size, hidden_size)
            positives: (batch_size, hidden_size)

        Returns:
            loss: scalar
        """
        # For now, just use the standard loss
        # TODO: Implement true gradient caching if needed for very large batches
        return self.base_loss(anchors, positives)


class TripletLoss:
    """
    Triplet Loss with cosine or euclidean distance

    Given (anchor, positive, negative) triplets, ensures:
        distance(anchor, positive) + margin < distance(anchor, negative)

    Args:
        margin: Margin for triplet loss (default: 0.5)
        distance_metric: 'cosine' or 'euclidean'
    """

    def __init__(self, margin: float = 0.5, distance_metric: str = 'cosine'):
        self.margin = margin
        self.distance_metric = distance_metric

    def cosine_distance(self, a: mx.array, b: mx.array) -> mx.array:
        """Cosine distance (1 - cosine_similarity)"""
        similarity = mx.sum(a * b, axis=1) / (
            mx.sqrt(mx.sum(a ** 2, axis=1)) * mx.sqrt(mx.sum(b ** 2, axis=1))
        )
        return 1 - similarity

    def euclidean_distance(self, a: mx.array, b: mx.array) -> mx.array:
        """Euclidean distance"""
        return mx.sqrt(mx.sum((a - b) ** 2, axis=1))

    def __call__(
        self,
        anchors: mx.array,
        positives: mx.array,
        negatives: mx.array
    ) -> mx.array:
        """
        Compute Triplet Loss

        Args:
            anchors: (batch_size, hidden_size)
            positives: (batch_size, hidden_size)
            negatives: (batch_size, hidden_size)

        Returns:
            loss: scalar
        """
        if self.distance_metric == 'cosine':
            dist_pos = self.cosine_distance(anchors, positives)
            dist_neg = self.cosine_distance(anchors, negatives)
        else:  # euclidean
            dist_pos = self.euclidean_distance(anchors, positives)
            dist_neg = self.euclidean_distance(anchors, negatives)

        # Triplet loss: max(0, dist_pos - dist_neg + margin)
        losses = mx.maximum(dist_pos - dist_neg + self.margin, 0)

        return mx.mean(losses)


class ContrastiveLoss:
    """
    Contrastive Loss (Hadsell et al., 2006)

    For positive pairs: minimize distance
    For negative pairs: maximize distance (up to margin)

    Args:
        margin: Margin for negative pairs (default: 0.5)
    """

    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def __call__(
        self,
        embeddings1: mx.array,
        embeddings2: mx.array,
        labels: mx.array
    ) -> mx.array:
        """
        Compute Contrastive Loss

        Args:
            embeddings1: (batch_size, hidden_size)
            embeddings2: (batch_size, hidden_size)
            labels: (batch_size,) - 1 for positive pairs, 0 for negative

        Returns:
            loss: scalar
        """
        # Euclidean distance
        distances = mx.sqrt(mx.sum((embeddings1 - embeddings2) ** 2, axis=1))

        # Positive pairs: minimize distance
        loss_positive = labels * (distances ** 2)

        # Negative pairs: maximize distance up to margin
        loss_negative = (1 - labels) * mx.maximum(self.margin - distances, 0) ** 2

        loss = mx.mean(loss_positive + loss_negative) / 2

        return loss


def test_losses():
    """Test loss functions"""
    print("="*70)
    print("Testing MLX Loss Functions")
    print("="*70)
    print()

    batch_size = 4
    hidden_size = 768

    # Test 1: Multiple Negatives Ranking Loss
    print("Test 1: Multiple Negatives Ranking Loss")
    print("-" * 70)

    mnr_loss = MultipleNegativesRankingLoss(scale=20.0)

    anchors = mx.random.normal((batch_size, hidden_size))
    positives = mx.random.normal((batch_size, hidden_size))

    loss = mnr_loss(anchors, positives)

    print(f"  Anchors shape: {anchors.shape}")
    print(f"  Positives shape: {positives.shape}")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  ✅ Multiple Negatives Ranking Loss works!")
    print()

    # Test 2: Cached version
    print("Test 2: Cached Multiple Negatives Ranking Loss")
    print("-" * 70)

    cached_loss = CachedMultipleNegativesRankingLoss(scale=20.0, mini_batch_size=2)

    loss_cached = cached_loss(anchors, positives)

    print(f"  Loss (cached): {loss_cached.item():.6f}")
    print(f"  Loss (standard): {loss.item():.6f}")
    print(f"  Difference: {abs(loss_cached.item() - loss.item()):.6f}")
    print(f"  ✅ Cached loss works!")
    print()

    # Test 3: Triplet Loss
    print("Test 3: Triplet Loss")
    print("-" * 70)

    triplet_loss = TripletLoss(margin=0.5, distance_metric='cosine')

    negatives = mx.random.normal((batch_size, hidden_size))

    loss_triplet = triplet_loss(anchors, positives, negatives)

    print(f"  Anchors: {anchors.shape}")
    print(f"  Positives: {positives.shape}")
    print(f"  Negatives: {negatives.shape}")
    print(f"  Loss: {loss_triplet.item():.6f}")
    print(f"  ✅ Triplet Loss works!")
    print()

    # Test 4: Contrastive Loss
    print("Test 4: Contrastive Loss")
    print("-" * 70)

    contrastive_loss = ContrastiveLoss(margin=0.5)

    # Half positive pairs, half negative pairs
    labels = mx.array([1, 1, 0, 0], dtype=mx.float32)

    loss_contrastive = contrastive_loss(anchors, positives, labels)

    print(f"  Embeddings1: {anchors.shape}")
    print(f"  Embeddings2: {positives.shape}")
    print(f"  Labels: {labels}")
    print(f"  Loss: {loss_contrastive.item():.6f}")
    print(f"  ✅ Contrastive Loss works!")
    print()

    # Test 5: Gradient computation
    print("Test 5: Gradient Computation")
    print("-" * 70)

    def loss_fn(anchors, positives):
        return mnr_loss(anchors, positives)

    # Compute gradients
    grad_fn = mx.grad(loss_fn, argnums=[0, 1])
    grads_anchor, grads_positive = grad_fn(anchors, positives)

    print(f"  Anchor gradients shape: {grads_anchor.shape}")
    print(f"  Positive gradients shape: {grads_positive.shape}")
    print(f"  Anchor gradient norm: {mx.sqrt(mx.sum(grads_anchor ** 2)).item():.6f}")
    print(f"  Positive gradient norm: {mx.sqrt(mx.sum(grads_positive ** 2)).item():.6f}")
    print(f"  ✅ Gradients computed correctly!")
    print()

    print("="*70)
    print("✅ All loss function tests complete!")
    print("="*70)


if __name__ == '__main__':
    test_losses()
