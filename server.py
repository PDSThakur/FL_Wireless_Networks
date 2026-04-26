"""
server.py - Flower ServerApp with FedAvg plus optional DifFense-style filtering.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics using weighted average by sample count."""
    if not metrics:
        return {}

    total_samples = sum(n for n, _ in metrics)
    aggregated: Metrics = {}
    all_keys = metrics[0][1].keys()
    for key in all_keys:
        weighted_sum = 0.0
        has_numeric = False
        for n, m in metrics:
            if key not in m:
                continue
            try:
                weighted_sum += n * float(m[key])
                has_numeric = True
            except (TypeError, ValueError):
                continue
        if has_numeric:
            aggregated[key] = weighted_sum / total_samples
    return aggregated


# ─────────────────────────────────────────────
# Layer 3 Defense: Trimmed Mean + Foolsgold
# ─────────────────────────────────────────────
def trimmed_mean_aggregate(
    updates: List[Tuple[np.ndarray, int]],  # (parameters, num_samples)
    trim_fraction: float = 0.1,
) -> np.ndarray:
    """
    Trimmed Mean: Remove top/bottom trim_fraction of updates before averaging.
    """
    if len(updates) < 3:
        return np.average([u for u, _ in updates], axis=0, weights=[n for _, n in updates])

    # Stack all parameter updates
    stacked = np.array([u for u, _ in updates])
    num_updates = len(updates)
    trim_count = max(1, int(num_updates * trim_fraction))

    # For each parameter dimension, trim extremes
    aggregated = []
    for i in range(stacked.shape[1]):
        layer_params = stacked[:, i, :]
        # Flatten for trimming
        flat = layer_params.reshape(num_updates, -1)
        
        # Compute mean per update for ranking
        norms = np.linalg.norm(flat, axis=1)
        
        # Get indices to keep (exclude top/bottom trim_count)
        keep_indices = np.argsort(norms)[trim_count:-trim_count] if trim_count > 0 else np.arange(num_updates)
        
        # Average remaining updates
        kept = stacked[keep_indices, i, :]
        aggregated.append(np.mean(kept, axis=0))

    return np.array(aggregated, dtype=object)


def foolsgold_aggregate(
    updates: List[Tuple[np.ndarray, int]],  # (parameters, num_samples)
    foolsgold_threshold: float = 0.5,
) -> Tuple[np.ndarray, List[int]]:
    """
    Foolsgold: Detect adversarial clients by measuring update similarity.
    Returns aggregated parameters and indices of flagged clients.
    """
    if len(updates) < 2:
        return np.array([u for u, _ in updates]), []

    # Flatten each update into a vector
    flattened = []
    for u, _ in updates:
        flat = np.concatenate([p.flatten() for p in u])
        flattened.append(flat)
    
    flattened = np.array(flattened)
    
    # Compute cosine similarity matrix
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # Avoid division by zero
    normalized = flattened / norms
    similarity_matrix = np.dot(normalized, normalized.T)

    # Compute maximum similarity for each client to any other
    np.fill_diagonal(similarity_matrix, 0)
    max_similarities = np.max(similarity_matrix, axis=1)

    # Flag clients with suspiciously high similarity
    flagged = np.where(max_similarities > foolsgold_threshold)[0].tolist()

    # Weight by inverse similarity (less similar = higher weight)
    weights = 1.0 - max_similarities
    weights = np.maximum(weights, 0.01)  # Ensure positive weights
    weights = weights / weights.sum()

    # Weighted average
    aggregated = np.average(flattened, axis=0, weights=weights)
    
    # Reshape back to original structure
    result = []
    idx = 0
    for u in updates:
        param_shapes = [p.shape for p in u]
        layer_sizes = [np.prod(s) for s in param_shapes]
        layers = []
        for shape in param_shapes:
            size = np.prod(shape)
            layers.append(aggregated[idx:idx+size].reshape(shape))
            idx += size
        result.append(np.array(layers, dtype=object))

    return np.array(result, dtype=object), flagged


class FedAvgWithEval(FedAvg):
    """
    FedAvg strategy with:
    1) server-side evaluation, and
    2) optional differential-testing + two-step MAD defense during aggregation.
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
        malicious_client_ids: Optional[List[int]] = None,
        # Layer 3 Defense: Trimmed Mean + Foolsgold
        layer3_trimmed_mean: bool = False,
        layer3_trim_fraction: float = 0.1,
        layer3_foolsgold: bool = False,
        layer3_foolsgold_threshold: float = 0.5,
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
        self.malicious_client_ids: Set[str] = set(str(int(cid)) for cid in (malicious_client_ids or []))
        
        # Layer 3 Defense Parameters
        self.layer3_trimmed_mean = layer3_trimmed_mean
        self.layer3_trim_fraction = max(0.0, min(0.5, float(layer3_trim_fraction)))
        self.layer3_foolsgold = layer3_foolsgold
        self.layer3_foolsgold_threshold = max(0.0, min(1.0, float(layer3_foolsgold_threshold)))
        
        self._defense_images: Optional[torch.Tensor] = None
        self._last_defense_stats: Dict[str, Scalar] = {
            "defense_flagged_clients": 0,
            "defense_retained_clients": 0,
            "defense_applied": 0,
            "defense_precision": 0.0,
            "defense_recall": 0.0,
            "defense_f1": 0.0,
            "defense_false_positive_rate": 0.0,
            "defense_true_positive_rate": 0.0,
            "defense_true_positives": 0,
            "defense_false_positives": 0,
            "defense_false_negatives": 0,
            "defense_true_negatives": 0,
            "defense_malicious_participants": 0,
            "defense_benign_participants": 0,
            "layer3_trimmed_mean_applied": 0,
            "layer3_foolsgold_applied": 0,
            "layer3_foolsgold_flagged": 0,
        }

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

    @staticmethod
    def _canonicalize_client_id(raw_id) -> str:
        """Normalize client ids for consistent matching/logging."""
        if raw_id is None:
            return ""
        try:
            return str(int(raw_id))
        except (TypeError, ValueError):
            return str(raw_id)

    def _compute_detection_metrics(
        self,
        flagged_ids: List[str],
        participant_ids: List[str],
    ) -> Dict[str, Scalar]:
        """
        Compute round-level defense detection quality using known malicious ids.
        Metrics are computed over clients that participated in the current round.
        """
        flagged = set(flagged_ids)
        participants = set(participant_ids)
        malicious_round = participants.intersection(self.malicious_client_ids)
        benign_round = participants.difference(malicious_round)

        tp = len(flagged.intersection(malicious_round))
        fp = len(flagged.intersection(benign_round))
        fn = len(malicious_round.difference(flagged))
        tn = len(benign_round.difference(flagged))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = recall

        return {
            "defense_true_positives": tp,
            "defense_false_positives": fp,
            "defense_false_negatives": fn,
            "defense_true_negatives": tn,
            "defense_precision": float(precision),
            "defense_recall": float(recall),
            "defense_f1": float(f1),
            "defense_false_positive_rate": float(fpr),
            "defense_true_positive_rate": float(tpr),
            "defense_malicious_participants": len(malicious_round),
            "defense_benign_participants": len(benign_round),
        }

    def _apply_defense(
        self, server_round: int, results: List[Tuple[object, object]]
    ) -> Tuple[List[Tuple[object, object]], Dict[str, Scalar]]:
        """Run differential testing + two-step MAD and filter suspicious updates."""
        stats: Dict[str, Scalar] = {
            "defense_applied": 0,
            "defense_flagged_clients": 0,
            "defense_retained_clients": len(results),
            "defense_mad1": 0.0,
            "defense_mad2": 0.0,
            "defense_precision": 0.0,
            "defense_recall": 0.0,
            "defense_f1": 0.0,
            "defense_false_positive_rate": 0.0,
            "defense_true_positive_rate": 0.0,
            "defense_true_positives": 0,
            "defense_false_positives": 0,
            "defense_false_negatives": 0,
            "defense_true_negatives": 0,
            "defense_malicious_participants": 0,
            "defense_benign_participants": len(results),
        }

        min_clients = int(getattr(self, "min_fit_clients", 2))
        if not self.defense_enabled or len(results) < max(4, min_clients):
            return results, stats

        defense_images = self._get_defense_images()
        if defense_images is None:
            return results, stats

        models: List[torch.nn.Module] = []
        client_ids: List[str] = []
        for idx, (client_proxy, fit_res) in enumerate(results):
            models.append(self._load_model_from_parameters(fit_res.parameters))
            metric_cid = None
            if getattr(fit_res, "metrics", None):
                metric_cid = fit_res.metrics.get("client_id")
            raw_id = metric_cid if metric_cid is not None else getattr(client_proxy, "cid", idx)
            client_ids.append(self._canonicalize_client_id(raw_id))

        try:
            base_embeddings = self._softmax_embeddings(models, defense_images)
            labels, majority_label, minority_label, reduced, centers = self._cluster_models(base_embeddings)
            majority_indices = np.where(labels == majority_label)[0]
            minority_indices = np.where(labels == minority_label)[0]
            if len(majority_indices) == 0 or len(minority_indices) == 0:
                return results, stats

            diff_images = self._generate_differential_inputs(
                models, defense_images, majority_indices, minority_indices
            )
            diff_embeddings = self._softmax_embeddings(models, diff_images)
            labels2, majority_label2, minority_label2, reduced2, centers2 = self._cluster_models(diff_embeddings)

            scores = self._compute_model_scores(
                reduced2, labels2, majority_label2, minority_label2, centers2
            )
            flags, normalized, mad1, mad2 = self._two_step_mad(scores, self.defense_mad_threshold)
            stats.update(
                {
                    "defense_applied": 1,
                    "defense_mad1": float(mad1),
                    "defense_mad2": float(mad2),
                }
            )

            flagged_indices = np.where(flags)[0].tolist()
            if len(flagged_indices) == 0:
                stats.update(self._compute_detection_metrics(flagged_ids=[], participant_ids=client_ids))
                return results, stats

            flagged_ids = [client_ids[i] for i in flagged_indices]
            keep_indices = [i for i in range(len(results)) if i not in flagged_indices]
            if len(keep_indices) < min_clients:
                # Keep the least suspicious clients to satisfy minimum requirement.
                order = np.argsort(normalized)
                keep_indices = order[:min_clients].tolist()
                flagged_ids = [client_ids[i] for i in range(len(results)) if i not in keep_indices]

            filtered_results = [results[i] for i in keep_indices]
            stats.update(
                {
                    "defense_flagged_clients": len(flagged_ids),
                    "defense_retained_clients": len(filtered_results),
                }
            )
            stats.update(self._compute_detection_metrics(flagged_ids=flagged_ids, participant_ids=client_ids))
            print(
                f"  [Defense][Round {server_round}] "
                f"flagged {len(flagged_ids)}/{len(results)} clients: {flagged_ids}"
            )
            return filtered_results, stats
        except Exception as ex:
            # Fail-open: keep training even if defense computation fails this round.
            print(f"  [Defense][Round {server_round}] skipped due to error: {ex}")
            return results, stats
        finally:
            # Explicit cleanup helps when using CUDA.
            del models

    # ------------------------
    # Strategy hooks
    # ------------------------
    def aggregate_fit(self, server_round: int, results: List, failures: List):
        """Aggregate updates after optional defense-based filtering."""
        # Layer 1+2: DifFense (directional testing + MAD)
        filtered_results, stats = self._apply_defense(server_round, results)
        self._last_defense_stats = stats
        
        # Layer 3: Trimmed Mean + Foolsgold
        if len(filtered_results) >= 3:
            # Extract parameters and sample counts
            layer3_updates = []
            for _, fit_res in filtered_results:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                num_samples = fit_res.num_examples
                layer3_updates.append((ndarrays, num_samples))
            
            # Apply Trimmed Mean if enabled
            if self.layer3_trimmed_mean:
                try:
                    aggregated = trimmed_mean_aggregate(
                        layer3_updates, 
                        trim_fraction=self.layer3_trim_fraction
                    )
                    # Create new parameters from trimmed mean result
                    from flwr.common import Parameters
                    new_params = Parameters(tensor=aggregated.tobytes(), tensor_type=aggregated.dtype)
                    
                    # Override the parent's aggregation with trimmed mean
                    aggregated_params, aggregated_metrics = super().aggregate_fit(
                        server_round, filtered_results, failures
                    )
                    # Replace with trimmed mean parameters
                    aggregated_params = new_params
                    self._last_defense_stats["layer3_trimmed_mean_applied"] = 1
                    print(f"  [Layer3][Round {server_round}] Trimmed Mean applied (trim={self.layer3_trim_fraction})")
                except Exception as e:
                    print(f"  [Layer3][Round {server_round}] Trimmed Mean failed: {e}")
                    aggregated_params, aggregated_metrics = super().aggregate_fit(
                        server_round, filtered_results, failures
                    )
            
            # Apply Foolsgold if enabled
            elif self.layer3_foolsgold:
                try:
                    aggregated, flagged_indices = foolsgold_aggregate(
                        layer3_updates,
                        foolsgold_threshold=self.layer3_foolsgold_threshold
                    )
                    self._last_defense_stats["layer3_foolsgold_applied"] = 1
                    self._last_defense_stats["layer3_foolsgold_flagged"] = len(flagged_indices)
                    print(f"  [Layer3][Round {server_round}] Foolsgold flagged {len(flagged_indices)} clients")
                except Exception as e:
                    print(f"  [Layer3][Round {server_round}] Foolsgold failed: {e}")
                    aggregated_params, aggregated_metrics = super().aggregate_fit(
                        server_round, filtered_results, failures
                    )
            else:
                # No Layer 3 - use standard FedAvg
                aggregated_params, aggregated_metrics = super().aggregate_fit(
                    server_round, filtered_results, failures
                )
        else:
            # Not enough clients for Layer 3
            aggregated_params, aggregated_metrics = super().aggregate_fit(
                server_round, filtered_results, failures
            )
        
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

        metrics.update(self._last_defense_stats)
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
                f" | Defense flagged: {int(metrics.get('defense_flagged_clients', 0))}"
                f"/{int(metrics.get('defense_retained_clients', 0)) + int(metrics.get('defense_flagged_clients', 0))}"
            )
        print(log)

        return_metrics: Dict[str, Scalar] = {"accuracy": float(acc)}
        if "pixel_backdoor_accuracy" in metrics:
            return_metrics["pixel_backdoor_accuracy"] = float(metrics["pixel_backdoor_accuracy"])
        if "semantic_backdoor_accuracy" in metrics:
            return_metrics["semantic_backdoor_accuracy"] = float(metrics["semantic_backdoor_accuracy"])
        if "defense_flagged_clients" in metrics:
            return_metrics["defense_flagged_clients"] = float(metrics["defense_flagged_clients"])
        if "defense_precision" in metrics:
            return_metrics["defense_precision"] = float(metrics["defense_precision"])
        if "defense_recall" in metrics:
            return_metrics["defense_recall"] = float(metrics["defense_recall"])
        if "defense_f1" in metrics:
            return_metrics["defense_f1"] = float(metrics["defense_f1"])
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
    malicious_client_ids: Optional[List[int]] = None,
    # Layer 3 Defense parameters
    layer3_trimmed_mean: bool = False,
    layer3_trim_fraction: float = 0.1,
    layer3_foolsgold: bool = False,
    layer3_foolsgold_threshold: float = 0.5,
    fraction_fit: float = 0.5,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    device: torch.device = None,
) -> ServerApp:
    """Build and return a Flower ServerApp with optional defense."""
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
        malicious_client_ids=malicious_client_ids,
        # Layer 3 Defense
        layer3_trimmed_mean=layer3_trimmed_mean,
        layer3_trim_fraction=layer3_trim_fraction,
        layer3_foolsgold=layer3_foolsgold,
        layer3_foolsgold_threshold=layer3_foolsgold_threshold,
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
