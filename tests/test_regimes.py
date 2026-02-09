"""
Tests for Kappa-LLM regime classification module.

Tests verify correct regime classification:
- NAGARE: High diversity, low rigidity
- KATASHI: High rigidity, high persistence, low diversity
- UTSUROI: Transitional states
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kappa_llm.regimes import RegimeClassifier, AttentionRegime
from kappa_llm.regimes.classifier import RegimeThresholds, RegimeClassification


class TestAttentionRegime:
    """Tests for AttentionRegime enum."""

    def test_regime_values(self):
        """Test regime enum values."""
        assert AttentionRegime.NAGARE.value == "nagare"
        assert AttentionRegime.UTSUROI.value == "utsuroi"
        assert AttentionRegime.KATASHI.value == "katashi"

    def test_is_healthy(self):
        """Test is_healthy method."""
        assert AttentionRegime.NAGARE.is_healthy() is True
        assert AttentionRegime.UTSUROI.is_healthy() is False
        assert AttentionRegime.KATASHI.is_healthy() is False

    def test_is_pathological(self):
        """Test is_pathological method."""
        assert AttentionRegime.NAGARE.is_pathological() is False
        assert AttentionRegime.UTSUROI.is_pathological() is False
        assert AttentionRegime.KATASHI.is_pathological() is True


class TestRegimeClassifier:
    """Tests for RegimeClassifier."""

    def test_default_thresholds(self):
        """Test default thresholds are set correctly."""
        classifier = RegimeClassifier()
        t = classifier.thresholds

        assert t.nagare_xi_min == 0.6
        assert t.nagare_eta_max == 0.4
        assert t.katashi_eta_min == 0.6

    def test_classify_nagare(self):
        """Test classification of clear NAGARE pattern."""
        classifier = RegimeClassifier()

        # High diversity, low rigidity, low divergence
        obs = {
            'omega': 0.85,
            'phi': 0.20,
            'eta': 0.15,
            'xi': 0.80,
            'delta': 0.15
        }

        result = classifier.classify(obs)

        assert result.regime == AttentionRegime.NAGARE
        assert result.confidence > 0.5
        assert result.nagare_score > result.katashi_score

    def test_classify_katashi(self):
        """Test classification of clear KATASHI pattern."""
        classifier = RegimeClassifier()

        # High rigidity, high persistence, low diversity
        obs = {
            'omega': 0.15,
            'phi': 0.75,
            'eta': 0.85,
            'xi': 0.15,
            'delta': 0.80
        }

        result = classifier.classify(obs)

        assert result.regime == AttentionRegime.KATASHI
        assert result.confidence > 0.5
        assert result.katashi_score > result.nagare_score

    def test_classify_utsuroi(self):
        """Test classification of transitional pattern."""
        classifier = RegimeClassifier()

        # Intermediate values
        obs = {
            'omega': 0.50,
            'phi': 0.45,
            'eta': 0.50,
            'xi': 0.50,
            'delta': 0.50
        }

        result = classifier.classify(obs)

        # Should be transitional when neither score is dominant
        assert result.nagare_score < 0.7
        assert result.katashi_score < 0.7

    def test_classification_has_distances(self):
        """Test that classification includes distances."""
        classifier = RegimeClassifier()

        obs = {'omega': 0.5, 'phi': 0.5, 'eta': 0.5, 'xi': 0.5, 'delta': 0.5}
        result = classifier.classify(obs)

        assert 'nagare' in result.distances
        assert 'utsuroi' in result.distances
        assert 'katashi' in result.distances

        for d in result.distances.values():
            assert d >= 0

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        thresholds = RegimeThresholds(
            nagare_xi_min=0.5,  # Lower than default
            nagare_eta_max=0.5,  # Higher than default
            nagare_delta_max=0.4
        )

        classifier = RegimeClassifier(thresholds=thresholds)

        # This should now classify as NAGARE with relaxed thresholds
        obs = {
            'omega': 0.60,
            'phi': 0.35,
            'eta': 0.40,
            'xi': 0.55,  # Would be UTSUROI with default xi_min=0.6
            'delta': 0.30
        }

        result = classifier.classify(obs)
        assert result.nagare_score > 0.5

    def test_hard_classification_mode(self):
        """Test hard classification mode."""
        classifier = RegimeClassifier(use_soft_classification=False)

        # Clear NAGARE pattern
        obs_nagare = {
            'omega': 0.85,
            'phi': 0.20,
            'eta': 0.30,
            'xi': 0.75,
            'delta': 0.20
        }

        result = classifier.classify(obs_nagare)
        assert result.regime == AttentionRegime.NAGARE

    def test_to_dict(self):
        """Test classification to_dict method."""
        classifier = RegimeClassifier()

        obs = {'omega': 0.5, 'phi': 0.5, 'eta': 0.5, 'xi': 0.5, 'delta': 0.5}
        result = classifier.classify(obs)

        d = result.to_dict()
        assert 'regime' in d
        assert 'confidence' in d
        assert 'nagare_score' in d
        assert 'katashi_score' in d


class TestClassifyTrajectory:
    """Tests for trajectory classification."""

    def test_classify_trajectory(self):
        """Test classification of trajectory."""
        classifier = RegimeClassifier()

        # Trajectory from NAGARE to KATASHI
        trajectory = [
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2},
            {'omega': 0.6, 'phi': 0.4, 'eta': 0.4, 'xi': 0.6, 'delta': 0.4},
            {'omega': 0.4, 'phi': 0.5, 'eta': 0.6, 'xi': 0.4, 'delta': 0.6},
            {'omega': 0.2, 'phi': 0.7, 'eta': 0.8, 'xi': 0.2, 'delta': 0.8},
        ]

        results = classifier.classify_trajectory(trajectory)

        assert len(results) == 4
        assert results[0].regime == AttentionRegime.NAGARE
        assert results[-1].regime == AttentionRegime.KATASHI

    def test_detect_transition(self):
        """Test transition detection in trajectory."""
        classifier = RegimeClassifier()

        trajectory = [
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2},
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2},
            {'omega': 0.2, 'phi': 0.7, 'eta': 0.8, 'xi': 0.2, 'delta': 0.8},
        ]

        transition_idx = classifier.detect_transition(trajectory)

        # Transition should happen at index 2 (or possibly 1->2)
        assert transition_idx is not None
        assert transition_idx > 0

    def test_no_transition(self):
        """Test when no transition occurs."""
        classifier = RegimeClassifier()

        # All NAGARE
        trajectory = [
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2},
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2},
        ]

        transition_idx = classifier.detect_transition(trajectory)
        assert transition_idx is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
