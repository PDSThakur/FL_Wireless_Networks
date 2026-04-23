"""
server.py - Flower ServerApp with FedAvg plus advanced DifFense-style filtering.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from flwr.common import (
    Context,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from model import get_model
from utils import MetricsTracker, evaluate


def weighted_average(metrics: List[Tuple[float, Metrics]]) -> Metrics:
    """Aggregate client metrics using weighted average by sample count."""
    if not metrics:
        return {}

    total_samples = float(sum(n for n, _ in metrics))
    if total_samples <= 0.0:
        return {}

    aggregated: Metrics = {}
    all_keys = metrics[0][1].keys()
    for key in all_keys:
        weighted_sum = float(sum(float(n) * float(m[key]) for n, m in metrics if key in m))
        aggregated[key] = weighted_sum / total_samples
    return aggregated


class FedAvgWithEval(FedAvg):
    """
    FedAvg strategy with:
    1) server-side evaluation,
    2) differential testing + two-step MAD scoring,
    3) adaptive thresholding,
    4) temporal suspicion memory,
    5) ensemble outlier detection,
    6) Bayesian malicious-posterior estimation,
    7) soft trust weighting in aggregation.
    """

    def __init__(
        self,
        test_loader,
        tracker: MetricsTracker,
        dataset: str,
        device: torch.device,
        pixel_backdoor_loader=None,
        semantic_backdoor_loader=None,
        defense_enabled: bool = False,
        defense_max_samples: int = 64,
        defense_pca_components: int = 5,
        defense_grad_steps: int = 3,
        defense_grad_step_size: float = 0.01,
        defense_mad_threshold: float = 2.5,
        defense_temporal_alpha: float = 0.3,
        defense_expected_malicious_frac: float = 0.1,
        defense_bayes_prior: float = 0.1,
        defense_sigmoid_temperature: float = 1.0,
        defense_min_trust_weight: float = 0.01,
        defense_detector_momentum: float = 0.8,
        defense_bayes_blend: float = 0.6,
        defense_adaptive_threshold: bool = True,
        defense_cv_folds: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.pixel_backdoor_loader = pixel_backdoor_loader
        self.semantic_backdoor_loader = semantic_backdoor_loader
        self.tracker = tracker
        self.dataset = dataset
        self.device = device
        self.model = get_model(dataset).to(device)

        self.defense_enabled = defense_enabled
        self.defense_max_samples = max(8, int(defense_max_samples))
        self.defense_pca_components = max(1, int(defense_pca_components))
        self.defense_grad_steps = max(1, int(defense_grad_steps))
        self.defense_grad_step_size = float(defense_grad_step_size)
        self.defense_mad_threshold = float(defense_mad_threshold)

        self.defense_temporal_alpha = float(np.clip(defense_temporal_alpha, 0.0, 1.0))
        self.defense_expected_malicious_frac = float(np.clip(defense_expected_malicious_frac, 1e-4, 0.49))
        self.defense_bayes_prior = float(np.clip(defense_bayes_prior, 1e-4, 0.49))
        self.defense_sigmoid_temperature = float(max(1e-4, defense_sigmoid_temperature))
        self.defense_min_trust_weight = float(np.clip(defense_min_trust_weight, 0.0, 1.0))
        self.defense_detector_momentum = float(np.clip(defense_detector_momentum, 0.0, 0.999))
        self.defense_bayes_blend = float(np.clip(defense_bayes_blend, 0.0, 1.0))
        self.defense_adaptive_threshold = bool(defense_adaptive_threshold)
        self.defense_cv_folds = max(2, int(defense_cv_folds))

        self._defense_images: Optional[torch.Tensor] = None
        self._suspicion_ema: Dict[str, float] = {}
        self._detector_weights: Dict[str, float] = {
            "mad": 1.0,
            "iforest": 1.0,
            "lof": 1.0,
            "ocsvm": 1.0,
        }
        self._last_client_trust_weights: Dict[str, float] = {}
        self._last_client_malicious_probs: Dict[str, float] = {}

        # Store flagged client IDs per round for FPR/FNR calculation
        self.defense_flagged_clients_per_round: list[list[str]] = []
        self._last_defense_stats: Dict[str, Scalar] = {
            "defense_flagged_clients": 0,
            "defense_retained_clients": 0,
            "defense_applied": 0,
            "defense_mad1": 0.0,
            "defense_mad2": 0.0,
            "defense_threshold": float(self.defense_mad_threshold),
            "defense_adaptive_threshold_used": 0,
            "defense_avg_trust_weight": 1.0,
            "defense_avg_malicious_prob": 0.0,
            "defense_detector_weight_mad": 0.25,
            "defense_detector_weight_iforest": 0.25,
            "defense_detector_weight_lof": 0.25,
            "defense_detector_weight_ocsvm": 0.25,
        }

    def get_defense_flagged_clients_per_round(self):
        """Return the list of flagged client IDs per round."""
        return self.defense_flagged_clients_per_round

    # ------------------------
    # Defense helper methods
    # ------------------------
    def _load_model_from_parameters(self, parameters: Parameters) -> torch.nn.Module:
        """Instantiate model and load client parameters."""
        model = get_model(self.dataset).to(self.device)
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _get_defense_images(self) -> Optional[torch.Tensor]:
        """Cache a small server-side image set used for differential testing."""
        if self._defense_images is not None:
            return self._defense_images

        batches = []
        seen = 0
        for images, _ in self.test_loader:
            batches.append(images)
            seen += images.size(0)
            if seen >= self.defense_max_samples:
                break
        if not batches:
            return None

        images = torch.cat(batches, dim=0)[: self.defense_max_samples].to(self.device)
        self._defense_images = images
        return self._defense_images

    @staticmethod
    def _softmax_embeddings(models: List[torch.nn.Module], images: torch.Tensor) -> np.ndarray:
        """
        Build model embeddings from softmax outputs.
        Shape: [num_models, batch_size * num_classes]
        """
        embeddings = []
        with torch.no_grad():
            for model in models:
                probs = torch.softmax(model(images), dim=1)
                embeddings.append(probs.reshape(-1).detach().cpu().numpy())
        return np.stack(embeddings, axis=0)

    def _cluster_models(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
        """Apply PCA + k-means (k=2), then identify majority/minority labels."""
        num_models = embeddings.shape[0]
        max_components = min(self.defense_pca_components, embeddings.shape[1], max(1, num_models - 1))
        if max_components < 1:
            max_components = 1

        pca = PCA(n_components=max_components, random_state=42)
        reduced = pca.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(reduced)
        centers = kmeans.cluster_centers_

        counts = np.bincount(labels, minlength=2)
        majority_label = int(np.argmax(counts))
        minority_label = 1 - majority_label
        return labels, majority_label, minority_label, reduced, centers

    def _generate_differential_inputs(
        self,
        models: List[torch.nn.Module],
        base_images: torch.Tensor,
        majority_indices: np.ndarray,
        minority_indices: np.ndarray,
    ) -> torch.Tensor:
        """
        Gradient ascent on images to maximize majority/minority behavioral gap.
        """
        x = base_images.detach().clone().requires_grad_(True)
        for _ in range(self.defense_grad_steps):
            maj_probs = []
            min_probs = []
            for idx in majority_indices:
                maj_probs.append(torch.softmax(models[int(idx)](x), dim=1))
            for idx in minority_indices:
                min_probs.append(torch.softmax(models[int(idx)](x), dim=1))

            maj_center = torch.stack(maj_probs, dim=0).mean(dim=0)
            min_center = torch.stack(min_probs, dim=0).mean(dim=0)
            objective = torch.mean((maj_center - min_center) ** 2)
            grad = torch.autograd.grad(objective, x, only_inputs=True)[0]

            x = (x + self.defense_grad_step_size * torch.sign(grad)).detach()
            x = torch.clamp(x, -5.0, 5.0).requires_grad_(True)
        return x.detach()

    @staticmethod
    def _compute_model_scores(
        reduced: np.ndarray,
        labels: np.ndarray,
        majority_label: int,
        minority_label: int,
        centers: np.ndarray,
    ) -> np.ndarray:
        """
        Score rule from prompt:
        - majority models: 2 * std(majority-cluster distances)
        - minority models: distance between cluster centers
        """
        maj_idx = np.where(labels == majority_label)[0]
        min_idx = np.where(labels == minority_label)[0]

        if len(maj_idx) == 0 or len(min_idx) == 0:
            return np.zeros(len(labels), dtype=np.float64)

        maj_dists = np.linalg.norm(reduced[maj_idx] - centers[majority_label], axis=1)
        majority_score = 2.0 * float(np.std(maj_dists))
        minority_score = float(np.linalg.norm(centers[majority_label] - centers[minority_label]))

        scores = np.full(len(labels), majority_score, dtype=np.float64)
        scores[min_idx] = minority_score
        return scores

    @staticmethod
    def _two_step_mad(
        scores: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Two-step MAD:
        mad1 = median absolute deviation
        mad2 = weighted blend using ratio above mad1
        """
        median_score = float(np.median(scores))
        abs_dev = np.abs(scores - median_score)
        mad1 = float(np.median(abs_dev))

        above_mask = abs_dev > mad1
        ratio_above = float(np.mean(above_mask)) if len(abs_dev) > 0 else 0.0
        high_dev_mean = float(np.mean(abs_dev[above_mask])) if np.any(above_mask) else mad1

        mad2 = (1.0 - ratio_above) * mad1 + ratio_above * high_dev_mean
        if mad2 < 1e-12:
            mad2 = float(np.std(scores)) + 1e-12
        normalized = abs_dev / (mad2 + 1e-12)
        flags = normalized > threshold
        return flags, normalized, mad1, mad2

    def _update_temporal_suspicion(
        self, client_ids: List[str], current_scores: np.ndarray
    ) -> np.ndarray:
        """EMA memory over rounds for each client's suspicion score."""
        ema_scores = np.zeros_like(current_scores, dtype=np.float64)
        alpha = self.defense_temporal_alpha

        for i, cid in enumerate(client_ids):
            current = float(current_scores[i])
            prev = float(self._suspicion_ema.get(cid, current))
            updated = alpha * current + (1.0 - alpha) * prev
            self._suspicion_ema[cid] = updated
            ema_scores[i] = updated
        return ema_scores

    def _choose_gmm_components_cv(self, scores: np.ndarray) -> int:
        """Choose 1 vs 2 Gaussian components via K-fold likelihood CV."""
        n = len(scores)
        if n < max(6, self.defense_cv_folds * 2):
            return 1

        x = scores.reshape(-1, 1)
        kf = KFold(n_splits=self.defense_cv_folds, shuffle=True, random_state=42)

        best_k = 1
        best_ll = -np.inf
        for k in (1, 2):
            fold_ll = []
            for train_idx, valid_idx in kf.split(x):
                model = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-6)
                model.fit(x[train_idx])
                fold_ll.append(float(model.score(x[valid_idx])))
            mean_ll = float(np.mean(fold_ll))
            if mean_ll > best_ll:
                best_ll = mean_ll
                best_k = k

        return best_k

    def _adaptive_valley_threshold(
        self, scores: np.ndarray, fallback_threshold: float
    ) -> Tuple[float, int]:
        """
        Adaptive thresholding:
        1) choose components by CV,
        2) if bimodal, use the density valley between component means.
        """
        if not self.defense_adaptive_threshold:
            return float(fallback_threshold), 0

        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) < 4 or float(np.std(scores)) < 1e-12:
            return float(fallback_threshold), 0

        try:
            best_k = self._choose_gmm_components_cv(scores)
            if best_k < 2:
                return float(fallback_threshold), 0

            x = scores.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42, reg_covar=1e-6)
            gmm.fit(x)

            means = gmm.means_.reshape(-1)
            order = np.argsort(means)
            low_mu = float(means[order[0]])
            high_mu = float(means[order[1]])
            if abs(high_mu - low_mu) < 1e-6:
                return float(fallback_threshold), 0

            grid = np.linspace(low_mu, high_mu, num=256).reshape(-1, 1)
            pdf = np.exp(gmm.score_samples(grid))
            valley = float(grid[int(np.argmin(pdf)), 0])
            valley = float(np.clip(valley, float(np.min(scores)), float(np.max(scores))))
            return valley, 1
        except Exception:
            return float(fallback_threshold), 0

    def _make_detector_features(
        self,
        raw_scores: np.ndarray,
        mad_normalized: np.ndarray,
        temporal_scores: np.ndarray,
        labels: np.ndarray,
        minority_label: int,
    ) -> np.ndarray:
        """Create low-dimensional robust features for ensemble detectors."""
        minority_indicator = (labels == minority_label).astype(np.float64)
        features = np.column_stack(
            [
                raw_scores.astype(np.float64),
                mad_normalized.astype(np.float64),
                temporal_scores.astype(np.float64),
                minority_indicator,
            ]
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _run_ensemble_detectors(
        self,
        features: np.ndarray,
        mad_normalized: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Run MAD + IsolationForest + LOF + OneClassSVM."""
        n = len(features)
        predictions: Dict[str, np.ndarray] = {
            "mad": (mad_normalized > self.defense_mad_threshold).astype(np.float64),
            "iforest": np.zeros(n, dtype=np.float64),
            "lof": np.zeros(n, dtype=np.float64),
            "ocsvm": np.zeros(n, dtype=np.float64),
        }
        active = ["mad"]

        if n < 4:
            return predictions, active

        contamination = float(np.clip(self.defense_expected_malicious_frac, 1.0 / max(4, n), 0.49))

        try:
            iforest = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
            )
            pred = iforest.fit_predict(features)
            predictions["iforest"] = (pred == -1).astype(np.float64)
            active.append("iforest")
        except Exception:
            pass

        try:
            n_neighbors = max(2, min(20, n - 1))
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
            )
            pred = lof.fit_predict(features)
            predictions["lof"] = (pred == -1).astype(np.float64)
            active.append("lof")
        except Exception:
            pass

        try:
            nu = float(np.clip(contamination, 1.0 / max(4, n), 0.49))
            ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
            pred = ocsvm.fit_predict(features)
            predictions["ocsvm"] = (pred == -1).astype(np.float64)
            active.append("ocsvm")
        except Exception:
            pass

        return predictions, active

    def _weighted_ensemble_vote(
        self,
        predictions: Dict[str, np.ndarray],
        active_detectors: List[str],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Weighted malicious vote in [0, 1] from active detectors."""
        if not active_detectors:
            n = len(next(iter(predictions.values())))
            return np.zeros(n, dtype=np.float64), {}

        raw_weights = {d: float(max(1e-6, self._detector_weights.get(d, 1.0))) for d in active_detectors}
        total = float(sum(raw_weights.values()))
        norm_weights = {d: w / total for d, w in raw_weights.items()}

        vote = np.zeros(len(next(iter(predictions.values()))), dtype=np.float64)
        for det in active_detectors:
            vote += norm_weights[det] * predictions[det]

        vote = np.clip(vote, 0.0, 1.0)
        return vote, norm_weights

    def _bayesian_malicious_posterior(self, scores: np.ndarray) -> np.ndarray:
        """
        Bayesian posterior P(malicious | score) from a 2-Gaussian mixture
        with prior correction to match defense_bayes_prior.
        """
        scores = np.asarray(scores, dtype=np.float64)
        n = len(scores)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        if n < 4 or float(np.std(scores)) < 1e-12:
            centered = scores - float(np.median(scores))
            scale = float(np.std(scores)) + 1e-6
            z = centered / scale
            probs = 1.0 / (1.0 + np.exp(-z))
            return np.clip(probs, 0.0, 1.0)

        try:
            x = scores.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42, reg_covar=1e-6)
            gmm.fit(x)

            means = gmm.means_.reshape(-1)
            malicious_component = int(np.argmax(means))
            responsibilities = gmm.predict_proba(x)[:, malicious_component]

            comp_prior = float(np.clip(gmm.weights_[malicious_component], 1e-6, 1.0 - 1e-6))
            target_prior = float(np.clip(self.defense_bayes_prior, 1e-6, 1.0 - 1e-6))

            odds = responsibilities / (1.0 - responsibilities + 1e-6)
            odds *= (target_prior / (1.0 - target_prior)) / (comp_prior / (1.0 - comp_prior))

            posterior = odds / (1.0 + odds)
            return np.clip(posterior, 0.0, 1.0)
        except Exception:
            centered = scores - float(np.median(scores))
            scale = float(np.std(scores)) + 1e-6
            z = centered / scale
            probs = 1.0 / (1.0 + np.exp(-z))
            return np.clip(probs, 0.0, 1.0)

    def _compute_trust_weights(
        self,
        temporal_scores: np.ndarray,
        adaptive_threshold: float,
        ensemble_vote: np.ndarray,
        bayes_posterior: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert uncertainty to soft trust:
        - sigmoid trust around adaptive threshold,
        - Bayesian malicious posterior,
        - ensemble malicious vote,
        then blend into final malicious probability and trust weight.
        """
        temp = self.defense_sigmoid_temperature
        sigmoid_trust = 1.0 / (1.0 + np.exp((temporal_scores - adaptive_threshold) / temp))

        final_malicious_prob = (
            self.defense_bayes_blend * bayes_posterior
            + (1.0 - self.defense_bayes_blend) * ensemble_vote
        )
        final_malicious_prob = np.clip(final_malicious_prob, 0.0, 1.0)

        trust = (1.0 - final_malicious_prob) * sigmoid_trust
        trust = np.clip(trust, self.defense_min_trust_weight, 1.0)
        return trust, final_malicious_prob

    def _update_detector_weights(
        self,
        predictions: Dict[str, np.ndarray],
        final_malicious_prob: np.ndarray,
        active_detectors: List[str],
    ) -> None:
        """Update detector reliability weights using consensus agreement."""
        if not active_detectors:
            return

        consensus = (final_malicious_prob >= 0.5).astype(np.float64)
        momentum = self.defense_detector_momentum

        for detector in active_detectors:
            pred = predictions[detector]
            agreement = 1.0 - float(np.mean(np.abs(pred - consensus)))
            target = max(0.05, agreement)
            prev = float(self._detector_weights.get(detector, 1.0))
            updated = momentum * prev + (1.0 - momentum) * target
            self._detector_weights[detector] = float(max(1e-6, updated))

        total = float(sum(max(1e-6, self._detector_weights[d]) for d in self._detector_weights))
        for detector in self._detector_weights:
            self._detector_weights[detector] = float(max(1e-6, self._detector_weights[detector]) / total)

    def _apply_defense(
        self, server_round: int, results: List[Tuple[object, object]]
    ) -> Tuple[List[Tuple[object, object]], Dict[str, Scalar], Dict[str, float]]:
        """
        Run differential testing and produce trust weights for soft aggregation.
        Returns:
            (results_used, stats, trust_weights_by_client_id)
        """
        trust_map = {
            str(getattr(client_proxy, "cid", idx)): 1.0
            for idx, (client_proxy, _) in enumerate(results)
        }

        stats: Dict[str, Scalar] = {
            "defense_applied": 0,
            "defense_flagged_clients": 0,
            "defense_retained_clients": len(results),
            "defense_mad1": 0.0,
            "defense_mad2": 0.0,
            "defense_threshold": float(self.defense_mad_threshold),
            "defense_adaptive_threshold_used": 0,
            "defense_avg_trust_weight": 1.0,
            "defense_avg_malicious_prob": 0.0,
            "defense_detector_weight_mad": float(self._detector_weights.get("mad", 0.25)),
            "defense_detector_weight_iforest": float(self._detector_weights.get("iforest", 0.25)),
            "defense_detector_weight_lof": float(self._detector_weights.get("lof", 0.25)),
            "defense_detector_weight_ocsvm": float(self._detector_weights.get("ocsvm", 0.25)),
        }

        min_clients = int(getattr(self, "min_fit_clients", 2))
        if not self.defense_enabled or len(results) < max(4, min_clients):
            self._last_client_trust_weights = trust_map
            self._last_client_malicious_probs = {cid: 0.0 for cid in trust_map}
            return results, stats, trust_map

        defense_images = self._get_defense_images()
        if defense_images is None:
            self._last_client_trust_weights = trust_map
            self._last_client_malicious_probs = {cid: 0.0 for cid in trust_map}
            return results, stats, trust_map

        models: List[torch.nn.Module] = []
        client_ids: List[str] = []
        for idx, (client_proxy, fit_res) in enumerate(results):
            models.append(self._load_model_from_parameters(fit_res.parameters))
            client_ids.append(str(getattr(client_proxy, "cid", idx)))

        try:
            base_embeddings = self._softmax_embeddings(models, defense_images)
            labels, majority_label, minority_label, _, _ = self._cluster_models(base_embeddings)
            majority_indices = np.where(labels == majority_label)[0]
            minority_indices = np.where(labels == minority_label)[0]
            if len(majority_indices) == 0 or len(minority_indices) == 0:
                self._last_client_trust_weights = trust_map
                self._last_client_malicious_probs = {cid: 0.0 for cid in trust_map}
                return results, stats, trust_map

            diff_images = self._generate_differential_inputs(
                models, defense_images, majority_indices, minority_indices
            )
            diff_embeddings = self._softmax_embeddings(models, diff_images)
            labels2, majority_label2, minority_label2, reduced2, centers2 = self._cluster_models(diff_embeddings)

            raw_scores = self._compute_model_scores(
                reduced2, labels2, majority_label2, minority_label2, centers2
            )
            _, mad_normalized, mad1, mad2 = self._two_step_mad(raw_scores, self.defense_mad_threshold)

            temporal_scores = self._update_temporal_suspicion(client_ids, mad_normalized)
            adaptive_threshold, adaptive_used = self._adaptive_valley_threshold(
                temporal_scores, self.defense_mad_threshold
            )

            features = self._make_detector_features(
                raw_scores=raw_scores,
                mad_normalized=mad_normalized,
                temporal_scores=temporal_scores,
                labels=labels2,
                minority_label=minority_label2,
            )
            detector_preds, active_detectors = self._run_ensemble_detectors(
                features=features,
                mad_normalized=mad_normalized,
            )
            ensemble_vote, _ = self._weighted_ensemble_vote(
                predictions=detector_preds,
                active_detectors=active_detectors,
            )

            bayes_posterior = self._bayesian_malicious_posterior(temporal_scores)
            trust_weights, malicious_probs = self._compute_trust_weights(
                temporal_scores=temporal_scores,
                adaptive_threshold=adaptive_threshold,
                ensemble_vote=ensemble_vote,
                bayes_posterior=bayes_posterior,
            )

            self._update_detector_weights(
                predictions=detector_preds,
                final_malicious_prob=malicious_probs,
                active_detectors=active_detectors,
            )

            self._last_client_malicious_probs = {}
            for cid, trust, mal_prob in zip(client_ids, trust_weights.tolist(), malicious_probs.tolist()):
                trust_map[cid] = float(trust)
                self._last_client_malicious_probs[cid] = float(mal_prob)

            self._last_client_trust_weights = dict(trust_map)

            flagged_ids = [cid for cid, p in self._last_client_malicious_probs.items() if p >= 0.5]
                # Store flagged client IDs for this round
                self.defense_flagged_clients_per_round.append(list(flagged_ids))

            stats.update(
                {
                    "defense_applied": 1,
                    "defense_mad1": float(mad1),
                    "defense_mad2": float(mad2),
                    "defense_threshold": float(adaptive_threshold),
                    "defense_adaptive_threshold_used": int(adaptive_used),
                    "defense_flagged_clients": len(flagged_ids),
                    "defense_retained_clients": len(results),
                    "defense_avg_trust_weight": float(np.mean(trust_weights)),
                    "defense_avg_malicious_prob": float(np.mean(malicious_probs)),
                    "defense_detector_weight_mad": float(self._detector_weights.get("mad", 0.25)),
                    "defense_detector_weight_iforest": float(self._detector_weights.get("iforest", 0.25)),
                    "defense_detector_weight_lof": float(self._detector_weights.get("lof", 0.25)),
                    "defense_detector_weight_ocsvm": float(self._detector_weights.get("ocsvm", 0.25)),
                }
            )

            print(
                f"  [Defense][Round {server_round}] "
                f"soft-flagged {len(flagged_ids)}/{len(results)} clients "
                f"| avg trust={float(np.mean(trust_weights)):.3f} "
                f"| avg P(mal)={float(np.mean(malicious_probs)):.3f} "
                f"| threshold={adaptive_threshold:.3f}"
            )
            return results, stats, trust_map
        except Exception as ex:
            # Fail-open: keep training even if defense computation fails this round.
            print(f"  [Defense][Round {server_round}] skipped due to error: {ex}")
            self._last_client_trust_weights = trust_map
            self._last_client_malicious_probs = {cid: 0.0 for cid in trust_map}
            return results, stats, trust_map
        finally:
            # Explicit cleanup helps when using CUDA.
            del models

    def _aggregate_with_trust(
        self,
        results: List[Tuple[object, object]],
        trust_weights: Dict[str, float],
    ) -> Optional[Parameters]:
        """FedAvg aggregation with per-client soft trust weights."""
        if not results:
            return None

        weighted_sums: Optional[List[np.ndarray]] = None
        param_dtypes: Optional[List[np.dtype]] = None
        total_weight = 0.0

        for idx, (client_proxy, fit_res) in enumerate(results):
            cid = str(getattr(client_proxy, "cid", idx))
            trust = float(np.clip(trust_weights.get(cid, 1.0), 0.0, 1.0))
            num_examples = float(max(0, int(getattr(fit_res, "num_examples", 0))))
            agg_weight = trust * num_examples
            if agg_weight <= 0.0:
                continue

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if weighted_sums is None:
                param_dtypes = [arr.dtype for arr in ndarrays]
                weighted_sums = [arr.astype(np.float64) * agg_weight for arr in ndarrays]
            else:
                for j, arr in enumerate(ndarrays):
                    weighted_sums[j] += arr.astype(np.float64) * agg_weight
            total_weight += agg_weight

        if weighted_sums is None or total_weight <= 0.0:
            return None

        assert param_dtypes is not None
        averaged = [
            (arr / total_weight).astype(param_dtypes[i], copy=False)
            for i, arr in enumerate(weighted_sums)
        ]
        return ndarrays_to_parameters(averaged)

    # ------------------------
    # Strategy hooks
    # ------------------------
    def aggregate_fit(self, server_round: int, results: List, failures: List):
        """Aggregate updates after optional defense-based soft weighting."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        defended_results, stats, trust_map = self._apply_defense(server_round, results)
        self._last_defense_stats = stats

        aggregated_params = self._aggregate_with_trust(defended_results, trust_map)
        if aggregated_params is None:
            # Fail-open fallback to base FedAvg behavior.
            aggregated_params, aggregated_metrics = super().aggregate_fit(
                server_round, defended_results, failures
            )
            return aggregated_params, aggregated_metrics

        aggregated_metrics: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_payload = []
            for idx, (client_proxy, fit_res) in enumerate(defended_results):
                cid = str(getattr(client_proxy, "cid", idx))
                eff_weight = float(max(0, int(getattr(fit_res, "num_examples", 0))))
                eff_weight *= float(np.clip(trust_map.get(cid, 1.0), 0.0, 1.0))
                metrics_payload.append((eff_weight, fit_res.metrics))
            aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_payload)

        return aggregated_params, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on clean and backdoor test sets."""
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        loss, acc = evaluate(self.model, self.test_loader, self.device)
        metrics: Dict[str, Scalar] = {
            "global_accuracy": acc,
            "global_loss": loss,
        }

        if self.pixel_backdoor_loader is not None:
            pixel_loss, pixel_asr = evaluate(self.model, self.pixel_backdoor_loader, self.device)
            metrics["pixel_backdoor_loss"] = pixel_loss
            metrics["pixel_backdoor_accuracy"] = pixel_asr

        if self.semantic_backdoor_loader is not None:
            semantic_loss, semantic_asr = evaluate(self.model, self.semantic_backdoor_loader, self.device)
            metrics["semantic_backdoor_loss"] = semantic_loss
            metrics["semantic_backdoor_accuracy"] = semantic_asr

        metrics.update(getattr(self, "_last_defense_stats", {}))
        self.tracker.record(round_num=server_round, **metrics)

        log = (
            f"  [Round {server_round:3d}] "
            f"Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%"
        )
        if "pixel_backdoor_accuracy" in metrics:
            log += f" | Pixel ASR: {float(metrics['pixel_backdoor_accuracy'])*100:.2f}%"
        if "semantic_backdoor_accuracy" in metrics:
            log += f" | Semantic ASR: {float(metrics['semantic_backdoor_accuracy'])*100:.2f}%"
        if float(metrics.get("defense_applied", 0)) > 0:
            log += (
                f" | Defense soft-flagged: {int(metrics.get('defense_flagged_clients', 0))}"
                f"/{int(metrics.get('defense_retained_clients', 0))}"
                f" | avg trust={float(metrics.get('defense_avg_trust_weight', 1.0)):.3f}"
            )
        print(log)

        return_metrics: Dict[str, Scalar] = {"accuracy": float(acc)}
        if "pixel_backdoor_accuracy" in metrics:
            return_metrics["pixel_backdoor_accuracy"] = float(metrics["pixel_backdoor_accuracy"])
        if "semantic_backdoor_accuracy" in metrics:
            return_metrics["semantic_backdoor_accuracy"] = float(metrics["semantic_backdoor_accuracy"])
        if "defense_flagged_clients" in metrics:
            return_metrics["defense_flagged_clients"] = float(metrics["defense_flagged_clients"])
        return float(loss), return_metrics


def build_server_app(
    test_loader,
    tracker: MetricsTracker,
    dataset: str,
    num_rounds: int,
    num_clients: int,
    pixel_backdoor_loader=None,
    semantic_backdoor_loader=None,
    defense_enabled: bool = False,
    defense_max_samples: int = 64,
    defense_pca_components: int = 5,
    defense_grad_steps: int = 3,
    defense_grad_step_size: float = 0.01,
    defense_mad_threshold: float = 2.5,
    defense_temporal_alpha: float = 0.3,
    defense_expected_malicious_frac: float = 0.1,
    defense_bayes_prior: float = 0.1,
    defense_sigmoid_temperature: float = 1.0,
    defense_min_trust_weight: float = 0.01,
    defense_detector_momentum: float = 0.8,
    defense_bayes_blend: float = 0.6,
    defense_adaptive_threshold: bool = True,
    defense_cv_folds: int = 3,
    fraction_fit: float = 0.5,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    device: torch.device = None,
) -> ServerApp:
    """Build and return a Flower ServerApp with optional advanced defense."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = get_model(dataset).to(device)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    )

    strategy = FedAvgWithEval(
        test_loader=test_loader,
        pixel_backdoor_loader=pixel_backdoor_loader,
        semantic_backdoor_loader=semantic_backdoor_loader,
        tracker=tracker,
        dataset=dataset,
        device=device,
        defense_enabled=defense_enabled,
        defense_max_samples=defense_max_samples,
        defense_pca_components=defense_pca_components,
        defense_grad_steps=defense_grad_steps,
        defense_grad_step_size=defense_grad_step_size,
        defense_mad_threshold=defense_mad_threshold,
        defense_temporal_alpha=defense_temporal_alpha,
        defense_expected_malicious_frac=defense_expected_malicious_frac,
        defense_bayes_prior=defense_bayes_prior,
        defense_sigmoid_temperature=defense_sigmoid_temperature,
        defense_min_trust_weight=defense_min_trust_weight,
        defense_detector_momentum=defense_detector_momentum,
        defense_bayes_blend=defense_bayes_blend,
        defense_adaptive_threshold=defense_adaptive_threshold,
        defense_cv_folds=defense_cv_folds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_fit_clients,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
