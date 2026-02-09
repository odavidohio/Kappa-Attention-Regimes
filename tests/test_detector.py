"""
Tests for Kappa-LLM detector module.

Tests verify correct hallucination detection using multi-observable Kappa score.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kappa_llm.detection import KappaDetector, DetectorConfig, DetectionResult
from kappa_llm.regimes import AttentionRegime


class TestDetectorConfig:
    """Tests for DetectorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DetectorConfig()

        assert config.w_rscore == 0.30
        assert config.w_eta == 0.25
        assert config.threshold == 0.5

    def test_balanced_config(self):
        """Test balanced configuration."""
        config = DetectorConfig.balanced()

        weights = config.get_weights_dict()
        total = sum(weights.values())

        # Should be normalized
        assert abs(total - 1.0) < 0.01

    def test_rscore_focused_config(self):
        """Test R-Score focused configuration."""
        config = DetectorConfig.rscore_focused()

        weights = config.get_weights_dict()
        assert weights['w_rscore'] > weights['w_eta']
        assert weights['w_rscore'] > weights['w_delta']

    def test_weights_normalization(self):
        """Test that weights are normalized when requested."""
        config = DetectorConfig(
            w_rscore=1.0,
            w_eta=1.0,
            w_xi_inv=1.0,
            w_delta=1.0,
            w_omega_inv=1.0,
            normalize_weights=True
        )

        weights = config.get_weights_dict()
        total = sum(weights.values())

        assert abs(total - 1.0) < 0.01


class TestKappaDetector:
    """Tests for KappaDetector."""

    def test_compute_kappa_score_high_hallucination(self):
        """Test Kappa score for hallucination-like pattern."""
        detector = KappaDetector()

        # High rigidity, low diversity = high Kappa score
        obs = {
            'omega': 0.2,
            'phi': 0.8,
            'eta': 0.85,
            'xi': 0.15,
            'delta': 0.75,
            'rscore': 0.9
        }

        kappa = detector.compute_kappa_score(obs)

        # Should be high (hallucination-like)
        assert kappa > 0.5

    def test_compute_kappa_score_low_hallucination(self):
        """Test Kappa score for factual-like pattern."""
        detector = KappaDetector()

        # Low rigidity, high diversity = low Kappa score
        obs = {
            'omega': 0.85,
            'phi': 0.2,
            'eta': 0.15,
            'xi': 0.85,
            'delta': 0.1,
            'rscore': 0.1
        }

        kappa = detector.compute_kappa_score(obs)

        # Should be low (factual-like)
        assert kappa < 0.4

    def test_detect_hallucination(self):
        """Test hallucination detection."""
        detector = KappaDetector(config=DetectorConfig(threshold=0.4))

        # Hallucination pattern
        obs = {
            'omega': 0.2,
            'phi': 0.8,
            'eta': 0.85,
            'xi': 0.15,
            'delta': 0.75,
            'rscore': 0.8
        }

        result = detector.detect(obs)

        assert result.is_hallucination is True
        assert result.kappa_score > 0.4
        assert result.confidence > 0

    def test_detect_factual(self):
        """Test factual detection."""
        detector = KappaDetector(config=DetectorConfig(threshold=0.5))

        # Factual pattern
        obs = {
            'omega': 0.85,
            'phi': 0.2,
            'eta': 0.15,
            'xi': 0.85,
            'delta': 0.1,
            'rscore': 0.1
        }

        result = detector.detect(obs)

        assert result.is_hallucination is False
        assert result.kappa_score < 0.5

    def test_detection_includes_regime(self):
        """Test that detection includes regime classification."""
        config = DetectorConfig(include_regime=True)
        detector = KappaDetector(config=config)

        obs = {
            'omega': 0.5, 'phi': 0.5, 'eta': 0.5,
            'xi': 0.5, 'delta': 0.5, 'rscore': 0.5
        }

        result = detector.detect(obs)

        assert result.regime is not None
        assert isinstance(result.regime, AttentionRegime)

    def test_detection_without_regime(self):
        """Test detection without regime classification."""
        config = DetectorConfig(include_regime=False)
        detector = KappaDetector(config=config)

        obs = {
            'omega': 0.5, 'phi': 0.5, 'eta': 0.5,
            'xi': 0.5, 'delta': 0.5, 'rscore': 0.5
        }

        result = detector.detect(obs)

        assert result.regime is None

    def test_component_scores(self):
        """Test that component scores are computed."""
        detector = KappaDetector()

        obs = {
            'omega': 0.5, 'phi': 0.5, 'eta': 0.5,
            'xi': 0.5, 'delta': 0.5, 'rscore': 0.5
        }

        components = detector.compute_component_scores(obs)

        assert 'rscore_contrib' in components
        assert 'eta_contrib' in components
        assert 'xi_inv_contrib' in components
        assert 'delta_contrib' in components
        assert 'omega_inv_contrib' in components

    def test_detect_batch(self):
        """Test batch detection."""
        detector = KappaDetector()

        obs_list = [
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2, 'rscore': 0.1},
            {'omega': 0.2, 'phi': 0.8, 'eta': 0.8, 'xi': 0.2, 'delta': 0.8, 'rscore': 0.8},
        ]

        results = detector.detect_batch(obs_list)

        assert len(results) == 2
        assert results[0].is_hallucination != results[1].is_hallucination

    def test_to_dict(self):
        """Test DetectionResult to_dict method."""
        detector = KappaDetector()

        obs = {'omega': 0.5, 'phi': 0.5, 'eta': 0.5, 'xi': 0.5, 'delta': 0.5, 'rscore': 0.5}
        result = detector.detect(obs)

        d = result.to_dict()

        assert 'is_hallucination' in d
        assert 'kappa_score' in d
        assert 'confidence' in d
        assert 'component_scores' in d


class TestCalibration:
    """Tests for detector calibration."""

    def test_calibrate_threshold(self):
        """Test threshold calibration."""
        detector = KappaDetector()

        # Create synthetic data
        factual_obs = [
            {'omega': 0.8, 'phi': 0.2, 'eta': 0.2, 'xi': 0.8, 'delta': 0.2, 'rscore': 0.1}
            for _ in range(50)
        ]
        hallu_obs = [
            {'omega': 0.2, 'phi': 0.8, 'eta': 0.8, 'xi': 0.2, 'delta': 0.8, 'rscore': 0.8}
            for _ in range(50)
        ]

        threshold = detector.calibrate_threshold(factual_obs, hallu_obs, target_fpr=0.05)

        # Threshold should be somewhere between factual and hallucination scores
        factual_mean = np.mean([detector.compute_kappa_score(o) for o in factual_obs])
        hallu_mean = np.mean([detector.compute_kappa_score(o) for o in hallu_obs])

        assert factual_mean < threshold < hallu_mean


class TestFeatureImportance:
    """Tests for feature importance analysis."""

    def test_get_feature_importance(self):
        """Test feature importance computation."""
        pytest.importorskip("sklearn")

        detector = KappaDetector()

        # Synthetic dataset
        obs_list = []
        labels = []

        # Factual samples
        for _ in range(50):
            obs_list.append({
                'omega': 0.8 + np.random.uniform(-0.1, 0.1),
                'phi': 0.2 + np.random.uniform(-0.1, 0.1),
                'eta': 0.2 + np.random.uniform(-0.1, 0.1),
                'xi': 0.8 + np.random.uniform(-0.1, 0.1),
                'delta': 0.2 + np.random.uniform(-0.1, 0.1),
                'rscore': 0.1 + np.random.uniform(-0.05, 0.05)
            })
            labels.append(0)

        # Hallucination samples
        for _ in range(50):
            obs_list.append({
                'omega': 0.2 + np.random.uniform(-0.1, 0.1),
                'phi': 0.8 + np.random.uniform(-0.1, 0.1),
                'eta': 0.8 + np.random.uniform(-0.1, 0.1),
                'xi': 0.2 + np.random.uniform(-0.1, 0.1),
                'delta': 0.8 + np.random.uniform(-0.1, 0.1),
                'rscore': 0.8 + np.random.uniform(-0.05, 0.05)
            })
            labels.append(1)

        importance = detector.get_feature_importance(obs_list, labels)

        assert 'rscore' in importance
        assert 'eta' in importance
        assert 'xi' in importance

        # At least some features should have positive importance
        assert any(v > 0 for v in importance.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
