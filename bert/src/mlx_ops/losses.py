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
        # Keep MLX-typed constant for all math
        self.scale = mx.array(scale, dtype=mx.float32)

    def cosine_similarity(self, a: mx.array, b: mx.array) -> mx.array:
        """
        Compute cosine similarity between all pairs

        Args:
            a: (batch_size, hidden_size) - anchors
            b: (batch_size, hidden_size) - candidates

        Returns:
            similarity: (batch_size, batch_size) - similarity matrix
        """
        # Normalize (pure MLX ops)
        a_sq = mx.power(a, mx.array(2.0, dtype=mx.float32))
        b_sq = mx.power(b, mx.array(2.0, dtype=mx.float32))
        a_den = mx.sqrt(mx.sum(a_sq, axis=1, keepdims=True))
        b_den = mx.sqrt(mx.sum(b_sq, axis=1, keepdims=True))
        a_norm = mx.divide(a, a_den)
        b_norm = mx.divide(b, b_den)

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

        # Scale by temperature (MLX scalar)
        similarity = mx.multiply(similarity, self.scale)

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
        self.margin = mx.array(margin, dtype=mx.float32)
        self.distance_metric = distance_metric

    def cosine_distance(self, a: mx.array, b: mx.array) -> mx.array:
        """Cosine distance (1 - cosine_similarity)"""
        num = mx.sum(mx.multiply(a, b), axis=1)
        den = mx.multiply(
            mx.sqrt(mx.sum(mx.power(a, mx.array(2.0, dtype=mx.float32)), axis=1)),
            mx.sqrt(mx.sum(mx.power(b, mx.array(2.0, dtype=mx.float32)), axis=1)),
        )
        similarity = mx.divide(num, den)
        return mx.subtract(mx.array(1.0, dtype=mx.float32), similarity)

    def euclidean_distance(self, a: mx.array, b: mx.array) -> mx.array:
        """Euclidean distance"""
        diff = mx.subtract(a, b)
        return mx.sqrt(mx.sum(mx.power(diff, mx.array(2.0, dtype=mx.float32)), axis=1))

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
        losses = mx.maximum(
            mx.add(mx.subtract(dist_pos, dist_neg), self.margin),
            mx.array(0.0, dtype=mx.float32),
        )

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
        self.margin = mx.array(margin, dtype=mx.float32)

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
        diff = mx.subtract(embeddings1, embeddings2)
        distances = mx.sqrt(mx.sum(mx.power(diff, mx.array(2.0, dtype=mx.float32)), axis=1))

        # Positive pairs: minimize distance
        loss_positive = mx.multiply(labels, mx.power(distances, mx.array(2.0, dtype=mx.float32)))

        # Negative pairs: maximize distance up to margin
        one = mx.array(1.0, dtype=mx.float32)
        loss_negative = mx.multiply(
            mx.subtract(one, labels),
            mx.power(mx.maximum(mx.subtract(self.margin, distances), mx.array(0.0, dtype=mx.float32)), mx.array(2.0, dtype=mx.float32)),
        )

        loss = mx.divide(mx.mean(mx.add(loss_positive, loss_negative)), mx.array(2.0, dtype=mx.float32))

        return loss


# Tests moved to bert/tests to keep this module compute-only and scalar-clean.
