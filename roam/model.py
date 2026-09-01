"""Preference learning, active pair selection, and group recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PreferenceProfile:
    """A learned linear utility model and an uncertainty estimate."""

    weights: np.ndarray
    covariance: np.ndarray
    comparisons: int

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features) @ self.weights

    @property
    def confidence(self) -> float:
        # Comparisons have diminishing returns; this is a UI calibration, not a
        # claim about statistical coverage.
        return float(1.0 - np.exp(-self.comparisons / 5.0))


def fit_preference_model(
    comparisons: Iterable[tuple[int, int]],
    features: np.ndarray,
    regularization: float = 1.5,
    max_iterations: int = 40,
) -> PreferenceProfile:
    """Fit Bradley-Terry-style pairwise utility with a Gaussian prior.

    Each tuple is ``(winner_index, loser_index)``. Newton updates make the tiny
    model deterministic and fast while the L2 prior prevents overconfidence
    when a user has only answered a few questions.
    """
    observations = list(comparisons)
    feature_count = features.shape[1]
    weights = np.zeros(feature_count, dtype=float)

    if not observations:
        return PreferenceProfile(weights, np.eye(feature_count) / regularization, 0)

    differences = np.asarray([features[winner] - features[loser] for winner, loser in observations])
    identity = np.eye(feature_count)

    for _ in range(max_iterations):
        logits = np.clip(differences @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = differences.T @ (probabilities - 1.0) + regularization * weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = differences.T @ (differences * curvature[:, None]) + regularization * identity
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step) < 1e-7:
            break

    covariance = np.linalg.inv(hessian)
    return PreferenceProfile(weights, covariance, len(observations))


def ranked_recommendations(
    profile: PreferenceProfile,
    features: np.ndarray,
    seen_indices: Iterable[int] = (),
    limit: int = 6,
) -> list[tuple[int, float]]:
    """Rank unseen destinations and map utility to an interpretable match score."""
    raw_scores = profile.scores(features)
    scale = max(float(np.std(raw_scores)), 0.35)
    matches = 100.0 / (1.0 + np.exp(-raw_scores / scale))
    seen = set(seen_indices)
    candidates = [index for index in range(len(raw_scores)) if index not in seen]
    candidates.sort(key=lambda index: (-matches[index], index))
    return [(index, float(matches[index])) for index in candidates[:limit]]


def select_next_pair(
    profile: PreferenceProfile,
    features: np.ndarray,
    shown_pairs: Iterable[tuple[int, int]] = (),
    seen_indices: Iterable[int] = (),
) -> tuple[int, int]:
    """Choose an informative, visually distinct comparison.

    Pairs are valuable when the current model is uncertain, predicts a close
    contest, and the two destinations differ meaningfully in feature space.
    Early on, a small exposure bonus also avoids showing the same places.
    """
    shown = {tuple(sorted(pair)) for pair in shown_pairs}
    seen_counts = {index: 0 for index in range(len(features))}
    for index in seen_indices:
        seen_counts[index] = seen_counts.get(index, 0) + 1

    utilities = profile.scores(features)
    best_pair: tuple[int, int] | None = None
    best_value = -np.inf
    for left, right in combinations(range(len(features)), 2):
        if (left, right) in shown:
            continue
        difference = features[left] - features[right]
        uncertainty = float(np.sqrt(max(difference @ profile.covariance @ difference, 0.0)))
        closeness = float(np.exp(-abs(utilities[left] - utilities[right])))
        diversity = float(np.linalg.norm(difference))
        exposure = 1.0 / (1.0 + seen_counts[left] + seen_counts[right])
        value = 0.48 * uncertainty + 0.28 * closeness + 0.16 * diversity + 0.08 * exposure
        if value > best_value:
            best_value = value
            best_pair = (left, right)

    # This can only happen after all pairs have been shown.
    return best_pair if best_pair is not None else (0, 1)


def contribution_explanation(
    profile: PreferenceProfile,
    destination_features: np.ndarray,
    feature_names: Sequence[str],
    limit: int = 2,
) -> list[str]:
    """Return the strongest positive reasons for one recommendation."""
    contributions = destination_features * profile.weights
    positive = [index for index in np.argsort(contributions)[::-1] if contributions[index] > 0]
    return [feature_names[index] for index in positive[:limit]]


def aggregate_group_scores(
    profiles: Sequence[PreferenceProfile],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average normalized member utilities and report disagreement.

    Per-person z-scoring prevents a confident profile from dominating merely
    because its weight magnitudes are larger.
    """
    if not profiles:
        raise ValueError("At least one profile is required")
    member_scores = np.vstack([profile.scores(features) for profile in profiles])
    means = member_scores.mean(axis=1, keepdims=True)
    standard_deviations = member_scores.std(axis=1, keepdims=True)
    normalized = (member_scores - means) / np.maximum(standard_deviations, 1e-6)
    group_scores = normalized.mean(axis=0)
    disagreement = normalized.std(axis=0)
    return group_scores, disagreement
