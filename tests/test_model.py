"""Unit tests for Roam's model, rankings, group consensus, and profiles."""

import unittest

import numpy as np

from roam.data import DESTINATIONS, FEATURES, feature_matrix
from roam.model import (
    PreferenceProfile,
    aggregate_group_scores,
    fit_preference_model,
    ranked_recommendations,
    select_next_pair,
)
from roam.profiles import profile_from_json, profile_to_json


class PreferenceModelTests(unittest.TestCase):
    def setUp(self):
        self.features = feature_matrix()

    def test_winners_score_above_losers_after_fit(self):
        comparisons = [(0, 1), (0, 3), (10, 15), (6, 7)]
        profile = fit_preference_model(comparisons, self.features)
        for winner, loser in comparisons:
            self.assertGreater(profile.scores(self.features)[winner], profile.scores(self.features)[loser])

    def test_empty_profile_is_neutral_and_finite(self):
        profile = fit_preference_model([], self.features)
        np.testing.assert_allclose(profile.weights, 0)
        self.assertTrue(np.isfinite(profile.covariance).all())
        self.assertEqual(profile.comparisons, 0)

    def test_rankings_exclude_seen_destinations(self):
        profile = fit_preference_model([(0, 1), (4, 2)], self.features)
        results = ranked_recommendations(profile, self.features, {0, 1, 2, 4}, limit=5)
        self.assertEqual(len(results), 5)
        self.assertFalse({index for index, _ in results} & {0, 1, 2, 4})
        self.assertGreaterEqual(results[0][1], results[-1][1])

    def test_active_pair_is_new_and_valid(self):
        profile = fit_preference_model([(0, 1)], self.features)
        pair = select_next_pair(profile, self.features, shown_pairs=[(0, 1)])
        self.assertNotEqual(tuple(sorted(pair)), (0, 1))
        self.assertNotEqual(pair[0], pair[1])
        self.assertTrue(all(0 <= index < len(DESTINATIONS) for index in pair))

    def test_group_consensus_averages_normalized_member_scores(self):
        features = np.array([[1.0, 0.0], [0.4, 0.8], [0.0, 1.0]])
        explorer = PreferenceProfile(np.array([2.0, -0.5]), np.eye(2), 10)
        relaxer = PreferenceProfile(np.array([-0.4, 1.0]), np.eye(2), 10)
        consensus, disagreement = aggregate_group_scores([explorer, relaxer], features)

        member_scores = np.vstack([explorer.scores(features), relaxer.scores(features)])
        normalized = (member_scores - member_scores.mean(axis=1, keepdims=True)) / member_scores.std(axis=1, keepdims=True)
        np.testing.assert_allclose(consensus, normalized.mean(axis=0))
        np.testing.assert_allclose(disagreement, normalized.std(axis=0))

    def test_profile_json_round_trip(self):
        profile = fit_preference_model([(0, 1), (2, 3)], self.features)
        name, restored = profile_from_json(profile_to_json("Avery", profile))
        self.assertEqual(name, "Avery")
        self.assertEqual(restored.comparisons, 2)
        np.testing.assert_allclose(restored.weights, profile.weights, atol=1e-7)
        self.assertEqual(restored.weights.shape, (len(FEATURES),))


if __name__ == "__main__":
    unittest.main()
