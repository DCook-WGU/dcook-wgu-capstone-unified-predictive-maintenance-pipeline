from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

from utils.synthetic.generator.synthetic_profiles import SensorRichProfile

from utils.synthetic.generator.synthetic_missingness import (
    MissingnessSpec,
    build_present_counts_for_block,
    apply_clustered_missingness_mask,
)


FaultType = Literal[
    "drift_up",
    "drift_down",
    "spike",
    "stuck_constant",
    "variance_burst",
    "step_shift",
    "intermittent_dropout",
    "sawtooth",
]

StateCalibrationTargets = Dict[str, Dict[str, Dict[str, float]]]
# shape:
# {
#   "normal":  {"sensor_00": {"mean": 1.2, "std": 0.4}, ...},
#   "abnormal":{"sensor_00": {"mean": 2.2, "std": 0.9}, ...},
#   "recovery":{...},
#   "buildup": { ... optional ... }
# }

BRIDGE_PAIRS = [("sensor_25", "sensor_26")]

@dataclass(frozen=True)
class EpisodeSpec:
    primary_sensor: str
    primary_fault_type: FaultType
    magnitude: float

    normal_before: int
    buildup: int
    failure: int
    recovery: int
    normal_after: int


class SyntheticGenerator:
    """
    Rich synthetic generator:
    - distribution-aware sampling (family + percentile bounds)
    - correlation-aware group drift
    - pairing-aware fault propagation
    - recovery attached to every abnormal episode
    """


        
    def __init__(
        self,
        *,
        normal_profiles: Dict[str, SensorRichProfile],
        abnormal_profiles: Dict[str, SensorRichProfile],
        recovery_profiles: Dict[str, SensorRichProfile],
        correlation_pairs_dataframe: pd.DataFrame,
        group_map_dataframe: pd.DataFrame,
        fault_pairings_dataframe: pd.DataFrame,
        correlation_hotspot_clusters: Optional[List[List[str]]] = None,
        correlation_cluster_derivation: Optional[Dict[str, object]] = None,
        fault_excluded_sensors: Optional[List[str]] = None,
        correlation_tuning: Optional[Dict[str, object]] = None,
        random_seed: int = 42,
        missingness_spec: Optional[MissingnessSpec] = None,
        state_calibration_targets: Optional[StateCalibrationTargets] = None,
        mean_within_k_std: float = 1.0,
        std_ratio_bounds: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        #self.rng = np.random.default_rng(int(random_seed))

        self.normal = normal_profiles
        self.abnormal = abnormal_profiles
        self.recovery = recovery_profiles

        self.sensors = sorted(self.normal.keys())

        self.state_calibration_targets = state_calibration_targets
        self.mean_within_k_std = float(mean_within_k_std)
        self.std_ratio_bounds = (float(std_ratio_bounds[0]), float(std_ratio_bounds[1]))

        self.fault_excluded_sensors: set[str] = {
            str(sensor_name).strip()
            for sensor_name in (fault_excluded_sensors or ["sensor_15", "sensor_50"])
            if str(sensor_name).strip()
        }

        # sensor -> group
        self.sensor_to_group: Dict[str, str] = {}
        for _, row in group_map_dataframe.iterrows():
            self.sensor_to_group[str(row["sensor"])] = str(row["group_name"])

        # group -> sensors
        self.group_to_sensors: Dict[str, List[str]] = {}
        for sensor, group in self.sensor_to_group.items():
            self.group_to_sensors.setdefault(group, []).append(sensor)

        # correlation lookup (pearson preferred)
        self.corr: Dict[Tuple[str, str], float] = {}
        for _, row in correlation_pairs_dataframe.iterrows():
            a = str(row["sensor_a"])
            b = str(row["sensor_b"])
            c = float(row["pearson_corr"])
            if np.isfinite(c):
                self.corr[(a, b)] = c
                self.corr[(b, a)] = c

        self.correlation_cluster_derivation = dict(correlation_cluster_derivation or {})
        
        self.correlation_tuning = dict(correlation_tuning or {})

        self.correlation_hotspot_clusters: List[List[str]] = self._resolve_hotspot_clusters(
            configured_clusters=correlation_hotspot_clusters
        )

        # propagation lookup: primary -> [{secondary, strength, lag}]
        self.propagation: Dict[str, List[dict]] = {}
        for _, row in fault_pairings_dataframe.iterrows():
            p = str(row["sensor_primary"])
            self.propagation.setdefault(p, []).append(
                {
                    "secondary": str(row["sensor_secondary"]),
                    "strength": float(row["fault_coupling_strength"]),
                    "lag": int(row["lag_cycles"]),
                    "recommended_secondary_fault": str(row.get("recommended_secondary_fault", "variance_burst")),
                }
            )

        self.missingness_spec = missingness_spec
        self.rng = np.random.default_rng(random_seed)

    # -------------------------
    # Distribution-aware sampling
    # -------------------------
    def _sample_series(self, profile: SensorRichProfile, n: int, smoothing: float) -> np.ndarray:
        family = str(profile.distribution_family).strip().lower()
        std = max(float(profile.std), 1e-6)
        rstd = max(float(profile.robust_std), 1e-6)

        if family in {"near_constant", "bounded_near_constant"}:
            base = np.full(n, float(profile.median), dtype=float)
            values = base + self.rng.normal(0.0, std * 0.05, size=n)

        elif family == "right_skewed":
            base = self.rng.normal(float(profile.mean), std, size=n)
            tail = np.abs(self.rng.normal(0.0, std * 2.0, size=n))
            mix = (self.rng.random(n) < 0.20).astype(float)
            values = base + tail * mix

        elif family == "left_skewed":
            base = self.rng.normal(float(profile.mean), std, size=n)
            tail = np.abs(self.rng.normal(0.0, std * 2.0, size=n))
            mix = (self.rng.random(n) < 0.20).astype(float)
            values = base - tail * mix

        elif family == "robust_empirical":
            values = self.rng.normal(float(profile.median), rstd, size=n)

        else:
            # normal / bounded_normal default
            values = self.rng.normal(float(profile.mean), std, size=n)

        # AR(1)-style smoothing
        if n > 1:
            sm = np.empty_like(values)
            sm[0] = values[0]
            for i in range(1, n):
                sm[i] = smoothing * sm[i - 1] + (1.0 - smoothing) * values[i]
            values = sm

        # percentile bounds + absolute bounds
        values = np.clip(values, float(profile.lower_bound), float(profile.upper_bound))
        values = np.clip(values, float(profile.min_value), float(profile.max_value))
        return values

    # -------------------------
    # Correlation-aware group drift
    # -------------------------
    def _apply_group_driver(self, dataframe: pd.DataFrame, group_name: str, profiles: Dict[str, SensorRichProfile], strength: float) -> None:
        sensors = [sensor for sensor in self.group_to_sensors.get(group_name, []) if sensor in dataframe.columns]
        if len(sensors) < 2:
            return

        n = len(dataframe)
        driver = self.rng.normal(0.0, 1.0, size=n)

        # smooth the driver so it behaves like a latent process shift
        if n > 1:
            for i in range(1, n):
                driver[i] = 0.7 * driver[i - 1] + 0.3 * driver[i]

        for sensor in sensors:
            prof = profiles[sensor]
            base_std = max(float(prof.std), 1e-6)

            x = dataframe[sensor].to_numpy(copy=True)
            x = x + driver * base_std * strength

            x = np.clip(x, float(prof.lower_bound), float(prof.upper_bound))
            x = np.clip(x, float(prof.min_value), float(prof.max_value))
            dataframe[sensor] = x

    def _build_group_correlation_matrix(
        self,
        group_name: str,
        profiles: Dict[str, SensorRichProfile],
        min_corr: float = 0.008,
        shrinkage: float = 0.04,
    ) -> tuple[list[str], np.ndarray]:
        sensors = [
            s for s in self.group_to_sensors.get(group_name, [])
            if s in profiles and s in self.sensors
        ]

        if len(sensors) < 2:
            return sensors, np.eye(len(sensors), dtype=float)

        n = len(sensors)
        corr = np.eye(n, dtype=float)

        for i, a in enumerate(sensors):
            for j in range(i + 1, n):
                b = sensors[j]
                c = float(self.corr.get((a, b), 0.0))
                if abs(c) < min_corr:
                    c = 0.0
                corr[i, j] = c
                corr[j, i] = c

        # shrink toward identity for stability
        corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(n)

        # PSD repair by clipping eigenvalues
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-6, None)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        d = np.sqrt(np.diag(corr_psd))
        corr_psd = corr_psd / np.outer(d, d)
        corr_psd = np.clip(corr_psd, -0.999, 0.999)
        np.fill_diagonal(corr_psd, 1.0)

        return sensors, corr_psd

    def _apply_group_correlated_residuals(
        self,
        dataframe: pd.DataFrame,
        group_name: str,
        profiles: Dict[str, SensorRichProfile],
        strength: float = 0.30,
        smooth_alpha: float = 0.65,
    ) -> None:
        sensors, corr = self._build_group_correlation_matrix(group_name, profiles=profiles)
        if len(sensors) < 2 or len(dataframe) == 0:
            return

        n_rows = len(dataframe)
        z = self.rng.multivariate_normal(
            mean=np.zeros(len(sensors), dtype=float),
            cov=corr,
            size=n_rows,
        )

        # smooth each latent series so it behaves like a temporal residual, not white noise
        if n_rows > 1:
            for j in range(z.shape[1]):
                for i in range(1, n_rows):
                    z[i, j] = smooth_alpha * z[i - 1, j] + (1.0 - smooth_alpha) * z[i, j]

        for j, sensor in enumerate(sensors):
            prof = profiles[sensor]
            std = max(float(prof.std), 1e-6)

            x = pd.to_numeric(dataframe[sensor], errors="coerce").to_numpy(dtype=float, copy=True)
            x = x + z[:, j] * std * float(strength)

            x = np.clip(x, float(prof.lower_bound), float(prof.upper_bound))
            x = np.clip(x, float(prof.min_value), float(prof.max_value))
            dataframe[sensor] = x

    def _apply_top_pairwise_overlay(
        self,
        dataframe: pd.DataFrame,
        profiles: Dict[str, SensorRichProfile],
        *,
        min_abs_corr: float = 0.08,
        top_n: int = 120,
        strength: float = 0.24,
        smooth_alpha: float = 0.90,
    ) -> None:
        if len(dataframe) == 0:
            return

        available = set(dataframe.columns)
        pairs = []

        seen = set()
        for (a, b), c in self.corr.items():
            if a == b:
                continue
            if a not in available or b not in available:
                continue
            if a not in profiles or b not in profiles:
                continue
            if abs(float(c)) < min_abs_corr:
                continue

            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)

            pairs.append((a, b, float(c)))

        if not pairs:
            return

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        pairs = pairs[:top_n]

        n = len(dataframe)

        for a, b, corr_value in pairs:
            latent = self.rng.normal(0.0, 1.0, size=n)

            if n > 1:
                for i in range(1, n):
                    latent[i] = smooth_alpha * latent[i - 1] + (1.0 - smooth_alpha) * latent[i]

            sign_b = 1.0 if corr_value >= 0 else -1.0
            
            if "sensor_51" in {a, b} and corr_value > 0:
                sign_b = 1.0

            weight = min(abs(corr_value), 0.95)

            prof_a = profiles[a]
            prof_b = profiles[b]

            std_a = max(float(prof_a.std), 1e-6)
            std_b = max(float(prof_b.std), 1e-6)

            xa = pd.to_numeric(dataframe[a], errors="coerce").to_numpy(dtype=float, copy=True)
            xb = pd.to_numeric(dataframe[b], errors="coerce").to_numpy(dtype=float, copy=True)

            xa = xa + latent * std_a * strength * weight
            xb = xb + latent * sign_b * std_b * strength * weight

            xa = np.clip(xa, float(prof_a.lower_bound), float(prof_a.upper_bound))
            xa = np.clip(xa, float(prof_a.min_value), float(prof_a.max_value))

            xb = np.clip(xb, float(prof_b.lower_bound), float(prof_b.upper_bound))
            xb = np.clip(xb, float(prof_b.min_value), float(prof_b.max_value))

            dataframe[a] = xa
            dataframe[b] = xb

    def _apply_named_cluster_overlay(
        self,
        dataframe: pd.DataFrame,
        profiles: Dict[str, SensorRichProfile],
        *,
        clusters: list[list[str]],
        strength: float = 0.35,
        smooth_alpha: float = 0.92,
    ) -> None:
        if len(dataframe) == 0:
            return

        n = len(dataframe)

        for cluster in clusters:
            sensors = [s for s in cluster if s in dataframe.columns and s in profiles]
            if len(sensors) < 2:
                continue

            latent = self.rng.normal(0.0, 1.0, size=n)
            if n > 1:
                for i in range(1, n):
                    latent[i] = smooth_alpha * latent[i - 1] + (1.0 - smooth_alpha) * latent[i]

            for sensor in sensors:
                prof = profiles[sensor]
                std = max(float(prof.std), 1e-6)

                x = pd.to_numeric(dataframe[sensor], errors="coerce").to_numpy(dtype=float, copy=True)
                x = x + latent * std * float(strength)

                x = np.clip(x, float(prof.lower_bound), float(prof.upper_bound))
                x = np.clip(x, float(prof.min_value), float(prof.max_value))
                dataframe[sensor] = x

    def _normalize_cluster_list(
        self,
        clusters: Optional[List[List[str]]],
    ) -> List[List[str]]:
        cleaned: List[List[str]] = []

        for cluster in (clusters or []):
            deduped: List[str] = []
            seen: set[str] = set()

            for sensor_name in cluster:
                sensor = str(sensor_name).strip()
                if not sensor:
                    continue
                if sensor not in self.sensors:
                    continue
                if sensor in seen:
                    continue
                deduped.append(sensor)
                seen.add(sensor)

            if len(deduped) >= 2:
                cleaned.append(deduped)

        return cleaned


    def _derive_hotspot_clusters_from_corr(self) -> List[List[str]]:
        cfg = dict(self.correlation_cluster_derivation or {})

        if not bool(cfg.get("enabled", True)):
            return []

        min_abs_corr = float(cfg.get("min_abs_corr", 0.20))
        top_n_pairs = int(cfg.get("top_n_pairs", 80))
        min_cluster_size = int(cfg.get("min_cluster_size", 2))
        max_cluster_size = int(cfg.get("max_cluster_size", 8))

        pairs: List[Tuple[str, str, float]] = []
        seen_pairs: set[Tuple[str, str]] = set()

        for (a, b), corr_value in self.corr.items():
            if a == b:
                continue

            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            if a not in self.sensors or b not in self.sensors:
                continue
            if not np.isfinite(corr_value):
                continue
            if abs(float(corr_value)) < min_abs_corr:
                continue

            pairs.append((a, b, float(corr_value)))

        if not pairs:
            return []

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        pairs = pairs[:top_n_pairs]

        adjacency: Dict[str, set[str]] = {sensor: set() for sensor in self.sensors}
        edge_strength: Dict[Tuple[str, str], float] = {}

        for a, b, corr_value in pairs:
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_strength[tuple(sorted((a, b)))] = abs(float(corr_value))

        visited: set[str] = set()
        raw_clusters: List[List[str]] = []

        candidate_nodes = [sensor for sensor, neighbors in adjacency.items() if neighbors]

        for start in candidate_nodes:
            if start in visited:
                continue

            stack = [start]
            component: List[str] = []

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(component) < min_cluster_size:
                continue

            if len(component) <= max_cluster_size:
                raw_clusters.append(sorted(component))
                continue

            # split oversized connected components greedily by strongest nodes
            node_strength = {
                node: sum(
                    edge_strength.get(tuple(sorted((node, neighbor))), 0.0)
                    for neighbor in adjacency[node]
                    if neighbor in component
                )
                for node in component
            }

            remaining = set(component)
            ordered_nodes = sorted(component, key=lambda s: (-node_strength[s], s))

            for seed in ordered_nodes:
                if seed not in remaining:
                    continue

                cluster = [seed]
                remaining.remove(seed)

                neighbors = sorted(
                    [n for n in adjacency[seed] if n in remaining],
                    key=lambda n: (-edge_strength.get(tuple(sorted((seed, n))), 0.0), n),
                )

                for neighbor in neighbors:
                    if len(cluster) >= max_cluster_size:
                        break
                    cluster.append(neighbor)
                    remaining.remove(neighbor)

                if len(cluster) >= min_cluster_size:
                    raw_clusters.append(cluster)

        return self._normalize_cluster_list(raw_clusters)


    def _resolve_hotspot_clusters(
        self,
        configured_clusters: Optional[List[List[str]]],
    ) -> List[List[str]]:
        cleaned = self._normalize_cluster_list(configured_clusters)

        if cleaned:
            return cleaned

        return self._derive_hotspot_clusters_from_corr()


    def _get_corr_tuning_section(
        self,
        section_name: str,
    ) -> Dict[str, object]:
        return dict((self.correlation_tuning or {}).get(str(section_name), {}) or {})

    def _get_corr_tuning_block(
        self,
        section_name: str,
        block_name: str,
        defaults: Dict[str, object],
    ) -> Dict[str, object]:
        section = self._get_corr_tuning_section(section_name)
        block = dict(section.get(block_name, {}) or {})
        merged = dict(defaults)
        merged.update(block)
        return merged

    def _get_chain_family_split_threshold(self) -> float:
        """
        Threshold used to split 3-sensor clusters into strong vs weak chain families.
        Pulled from correlation_tuning.family_split_rules.chain_cluster_avg_abs_corr_threshold.
        """
        rules = dict((self.correlation_tuning or {}).get("family_split_rules", {}) or {})
        return float(rules.get("chain_cluster_avg_abs_corr_threshold", 0.75))

    def _cluster_avg_abs_corr(
        self,
        cluster: Sequence[str],
    ) -> float:
        """
        Average absolute pairwise correlation inside a cluster, using self.corr.
        Missing pairs contribute 0.0.
        """
        sensors = [sensor for sensor in cluster if sensor in self.sensors]
        if len(sensors) < 2:
            return 0.0

        values: List[float] = []

        for i, a in enumerate(sensors):
            for b in sensors[i + 1:]:
                values.append(abs(float(self.corr.get((a, b), 0.0))))

        if not values:
            return 0.0

        return float(np.mean(values))


    def _classify_cluster_family(
        self,
        cluster: Sequence[str],
    ) -> str:
        """
        Classify hotspot clusters into family types.

        Rules:
        - 2 sensors  -> tight_pair
        - 3 sensors  -> strong_chain_cluster or weak_chain_cluster
                        based on average internal absolute correlation
        - 4+ sensors -> dense_cluster
        """
        valid = [sensor for sensor in cluster if sensor in self.sensors]
        n = len(valid)

        if n <= 2:
            return "tight_pair"

        if n == 3:
            avg_abs_corr = self._cluster_avg_abs_corr(valid)
            threshold = self._get_chain_family_split_threshold()
            if avg_abs_corr >= threshold:
                return "strong_chain_cluster"
            return "weak_chain_cluster"

        return "dense_cluster"
    
    def _get_family_corr_tuning_block(
        self,
        section_name: str,
        family_name: str,
        block_name: str,
        defaults: Dict[str, object],
    ) -> Dict[str, object]:
        base_block = self._get_corr_tuning_block(
            section_name,
            block_name,
            defaults=defaults,
        )

        section = self._get_corr_tuning_section(section_name)
        family_overrides = dict(section.get("family_overrides", {}) or {})
        family_block = dict(
            (family_overrides.get(str(family_name), {}) or {}).get(block_name, {}) or {}
        )

        merged = dict(base_block)
        merged.update(family_block)
        return merged


    def _get_bridge_pairs_from_tuning(
        self,
        section_name: str,
    ) -> List[Tuple[str, str]]:
        block = self._get_corr_tuning_block(
            section_name,
            "bridge_pair_generation",
            defaults={"bridge_pairs": BRIDGE_PAIRS},
        )

        bridge_pairs_raw = list(block.get("bridge_pairs", BRIDGE_PAIRS) or [])
        cleaned: List[Tuple[str, str]] = []

        for pair in bridge_pairs_raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue

            a = str(pair[0]).strip()
            b = str(pair[1]).strip()

            if not a or not b:
                continue

            cleaned.append((a, b))

        return cleaned or list(BRIDGE_PAIRS)


    def _get_priority_pair_specs_from_tuning(
        self,
        section_name: str,
    ) -> List[Dict[str, object]]:
        block = self._get_corr_tuning_block(
            section_name,
            "priority_pair_generation",
            defaults={
                "enabled": False,
                "pair_specs": [],
            },
        )

        if not bool(block.get("enabled", False)):
            return []

        pair_specs_raw = list(block.get("pair_specs", []) or [])
        cleaned: List[Dict[str, object]] = []

        for raw in pair_specs_raw:
            if not isinstance(raw, dict):
                continue

            anchor_sensor = str(raw.get("anchor_sensor", "")).strip()
            member_sensor = str(raw.get("member_sensor", "")).strip()

            if not anchor_sensor or not member_sensor:
                continue

            cleaned.append(
                {
                    "anchor_sensor": anchor_sensor,
                    "member_sensor": member_sensor,
                    "blend": float(raw.get("blend", 0.985)),
                    "min_abs_corr": float(raw.get("min_abs_corr", 0.05)),
                    "residual_floor": float(raw.get("residual_floor", 0.010)),
                    "smooth_alpha": float(raw.get("smooth_alpha", 0.985)),
                }
            )

        return cleaned


    def _apply_priority_pair_generation(
        self,
        dataframe: pd.DataFrame,
        profiles: Dict[str, SensorRichProfile],
        *,
        pair_specs: List[Dict[str, object]],
    ) -> None:
        """
        Pair-specific post-bridge reinforcement for a small set of stubborn target pairs.
        This is intentionally narrower than hotspot/bridge family logic.
        """
        if len(dataframe) == 0 or not pair_specs:
            return

        for spec in pair_specs:
            anchor_sensor = str(spec["anchor_sensor"])
            member_sensor = str(spec["member_sensor"])

            if anchor_sensor not in dataframe.columns or member_sensor not in dataframe.columns:
                continue
            if anchor_sensor not in profiles or member_sensor not in profiles:
                continue

            target_corr = float(self.corr.get((anchor_sensor, member_sensor), 0.0))
            target_corr = float(np.clip(target_corr, -0.98, 0.98))

            min_abs_corr = float(spec["min_abs_corr"])
            if abs(target_corr) < min_abs_corr:
                continue

            blend = float(spec["blend"])
            residual_floor = float(spec["residual_floor"])
            smooth_alpha = float(spec["smooth_alpha"])

            anchor_profile = profiles[anchor_sensor]
            member_profile = profiles[member_sensor]

            anchor_values = pd.to_numeric(
                dataframe[anchor_sensor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            anchor_mask = np.isfinite(anchor_values)
            if anchor_mask.sum() < 3:
                continue

            anchor_current = anchor_values[anchor_mask]
            anchor_mean = float(np.mean(anchor_current))
            anchor_std = float(np.std(anchor_current, ddof=1)) if anchor_current.size > 1 else 0.0
            anchor_std = max(anchor_std, max(float(anchor_profile.std), 1e-6))

            anchor_z = (anchor_values - anchor_mean) / anchor_std
            anchor_z = self._smooth_vector(anchor_z, alpha=smooth_alpha)

            eps = self.rng.normal(0.0, 1.0, size=len(anchor_values))
            eps = self._smooth_vector(eps, alpha=smooth_alpha)

            residual_scale = max(1.0 - target_corr**2, residual_floor**2) ** 0.5
            member_z = target_corr * anchor_z + residual_scale * eps

            generated = float(member_profile.mean) + member_z * max(float(member_profile.std), 1e-6)

            existing = pd.to_numeric(
                dataframe[member_sensor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            combined = (1.0 - blend) * existing + blend * generated
            combined = np.clip(combined, float(member_profile.lower_bound), float(member_profile.upper_bound))
            combined = np.clip(combined, float(member_profile.min_value), float(member_profile.max_value))

            dataframe[member_sensor] = combined
    
    def _pick_cluster_anchor(
        self,
        cluster: List[str],
    ) -> Optional[str]:
        """
        Choose the anchor sensor in a hotspot cluster as the sensor with the
        highest mean absolute correlation to the other cluster members.
        """
        sensors = [s for s in cluster if s in self.sensors]
        if len(sensors) < 2:
            return None

        best_sensor = None
        best_score = -1.0

        for sensor in sensors:
            scores = []
            for other in sensors:
                if other == sensor:
                    continue
                scores.append(abs(float(self.corr.get((sensor, other), 0.0))))
            mean_score = float(np.mean(scores)) if scores else 0.0
            if mean_score > best_score:
                best_score = mean_score
                best_sensor = sensor

        return best_sensor


    def _apply_anchor_cluster_relationships(
        self,
        dataframe: pd.DataFrame,
        profiles: Dict[str, SensorRichProfile],
        *,
        clusters: List[List[str]],
        min_abs_corr: float = 0.35,
        residual_scale: float = 0.35,
        smooth_alpha: float = 0.90,
    ) -> None:
        """
        Rebuild hotspot cluster members from an anchor sensor path using
        corr/std/mean-implied linear relationships.

        This is stronger than additive overlay and is intended to improve the
        tight correlation families from Silver EDA hotspot clusters.
        """
        if len(dataframe) == 0:
            return

        n = len(dataframe)

        for cluster in clusters:
            sensors = [
                s for s in cluster
                if s in dataframe.columns and s in profiles
            ]
            if len(sensors) < 2:
                continue

            anchor = self._pick_cluster_anchor(sensors)
            if anchor is None or anchor not in dataframe.columns or anchor not in profiles:
                continue

            anchor_profile = profiles[anchor]
            anchor_mean = float(anchor_profile.mean)
            anchor_std = max(float(anchor_profile.std), 1e-6)

            anchor_values = pd.to_numeric(
                dataframe[anchor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            anchor_values = np.clip(anchor_values, float(anchor_profile.lower_bound), float(anchor_profile.upper_bound))
            anchor_values = np.clip(anchor_values, float(anchor_profile.min_value), float(anchor_profile.max_value))
            dataframe[anchor] = anchor_values

            for sensor in sensors:
                if sensor == anchor:
                    continue

                corr_value = float(self.corr.get((anchor, sensor), 0.0))
                if abs(corr_value) < float(min_abs_corr):
                    continue

                sensor_profile = profiles[sensor]
                sensor_mean = float(sensor_profile.mean)
                sensor_std = max(float(sensor_profile.std), 1e-6)

                beta = corr_value * (sensor_std / anchor_std)
                intercept = sensor_mean - beta * anchor_mean

                member_values = intercept + beta * anchor_values

                residual_std = sensor_std * max(1e-6, np.sqrt(max(0.0, 1.0 - corr_value**2)))
                residual_noise = self.rng.normal(
                    0.0,
                    residual_std * float(residual_scale),
                    size=n,
                )

                if n > 1:
                    for i in range(1, n):
                        residual_noise[i] = (
                            smooth_alpha * residual_noise[i - 1]
                            + (1.0 - smooth_alpha) * residual_noise[i]
                        )

                member_values = member_values + residual_noise

                member_values = np.clip(member_values, float(sensor_profile.lower_bound), float(sensor_profile.upper_bound))
                member_values = np.clip(member_values, float(sensor_profile.min_value), float(sensor_profile.max_value))

                dataframe[sensor] = member_values

    def _smooth_vector(self, values: np.ndarray, alpha: float = 0.90) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        if values.size <= 1:
            return values

        for i in range(1, values.size):
            values[i] = alpha * values[i - 1] + (1.0 - alpha) * values[i]
        return values


    def _choose_cluster_anchor_sensor(
        self,
        cluster: List[str],
    ) -> Optional[str]:
        valid = [sensor for sensor in cluster if sensor in self.sensors]
        if len(valid) == 0:
            return None
        if len(valid) == 1:
            return valid[0]

        best_sensor = None
        best_score = -1.0

        for sensor in valid:
            score = 0.0
            for other in valid:
                if other == sensor:
                    continue
                score += abs(float(self.corr.get((sensor, other), 0.0)))

            if score > best_score:
                best_score = score
                best_sensor = sensor

        return best_sensor


    def _apply_anchor_cluster_generation(
        self,
        dataframe: pd.DataFrame,
        profiles: Dict[str, SensorRichProfile],
        *,
        clusters: List[List[str]],
        blend: float = 0.85,
        min_abs_corr: float = 0.20,
        residual_floor: float = 0.08,
        smooth_alpha: float = 0.90,
    ) -> None:
        """
        Strengthen hotspot clusters by choosing an anchor sensor path and
        generating other cluster members from that anchor using the pairwise
        correlation structure.
        """
        if len(dataframe) == 0:
            return

        for cluster in clusters:
            sensors = [
                sensor
                for sensor in cluster
                if sensor in dataframe.columns and sensor in profiles
            ]
            if len(sensors) < 2:
                continue

            anchor_sensor = self._choose_cluster_anchor_sensor(sensors)
            if anchor_sensor is None:
                continue

            anchor_profile = profiles[anchor_sensor]
            anchor_values = pd.to_numeric(
                dataframe[anchor_sensor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            anchor_mask = np.isfinite(anchor_values)
            if anchor_mask.sum() < 3:
                continue

            anchor_current = anchor_values[anchor_mask]
            anchor_mean = float(np.mean(anchor_current))
            anchor_std = float(np.std(anchor_current, ddof=1)) if anchor_current.size > 1 else 0.0
            anchor_std = max(anchor_std, max(float(anchor_profile.std), 1e-6))

            anchor_z = (anchor_values - anchor_mean) / anchor_std
            anchor_z = self._smooth_vector(anchor_z, alpha=smooth_alpha)

            for sensor in sensors:
                if sensor == anchor_sensor:
                    continue

                target_corr = float(self.corr.get((anchor_sensor, sensor), 0.0))
                target_corr = float(np.clip(target_corr, -0.98, 0.98))

                if abs(target_corr) < min_abs_corr:
                    continue

                prof = profiles[sensor]
                existing = pd.to_numeric(
                    dataframe[sensor],
                    errors="coerce",
                ).to_numpy(dtype=float, copy=True)

                eps = self.rng.normal(0.0, 1.0, size=len(existing))
                eps = self._smooth_vector(eps, alpha=smooth_alpha)

                residual_scale = max(1.0 - target_corr**2, residual_floor**2) ** 0.5
                member_z = target_corr * anchor_z + residual_scale * eps

                generated = float(prof.mean) + member_z * max(float(prof.std), 1e-6)

                combined = (1.0 - float(blend)) * existing + float(blend) * generated
                combined = np.clip(combined, float(prof.lower_bound), float(prof.upper_bound))
                combined = np.clip(combined, float(prof.min_value), float(prof.max_value))

                dataframe[sensor] = combined

    def _apply_bridge_pair_generation(
        self,
        dataframe: pd.DataFrame,
        profiles: Dict[str, SensorRichProfile],
        *,
        bridge_pairs: List[Tuple[str, str]],
        blend: float = 0.94,
        min_abs_corr: float = 0.20,
        residual_floor: float = 0.03,
        smooth_alpha: float = 0.94,
    ) -> None:
        """
        Strongly tie selected bridge-pair sensors together without forcing the
        secondary sensor into the full hotspot cluster.

        Example use:
            ("sensor_25", "sensor_26")
        """
        if len(dataframe) == 0:
            return

        for anchor_sensor, member_sensor in bridge_pairs:
            if anchor_sensor not in dataframe.columns or member_sensor not in dataframe.columns:
                continue
            if anchor_sensor not in profiles or member_sensor not in profiles:
                continue

            target_corr = float(self.corr.get((anchor_sensor, member_sensor), 0.0))
            target_corr = float(np.clip(target_corr, -0.98, 0.98))

            if abs(target_corr) < float(min_abs_corr):
                continue

            anchor_profile = profiles[anchor_sensor]
            member_profile = profiles[member_sensor]

            anchor_values = pd.to_numeric(
                dataframe[anchor_sensor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            anchor_mask = np.isfinite(anchor_values)
            if anchor_mask.sum() < 3:
                continue

            anchor_current = anchor_values[anchor_mask]
            anchor_mean = float(np.mean(anchor_current))
            anchor_std = float(np.std(anchor_current, ddof=1)) if anchor_current.size > 1 else 0.0
            anchor_std = max(anchor_std, max(float(anchor_profile.std), 1e-6))

            anchor_z = (anchor_values - anchor_mean) / anchor_std
            anchor_z = self._smooth_vector(anchor_z, alpha=smooth_alpha)

            eps = self.rng.normal(0.0, 1.0, size=len(anchor_values))
            eps = self._smooth_vector(eps, alpha=smooth_alpha)

            residual_scale = max(1.0 - target_corr**2, residual_floor**2) ** 0.5
            member_z = target_corr * anchor_z + residual_scale * eps

            generated = float(member_profile.mean) + member_z * max(float(member_profile.std), 1e-6)

            existing = pd.to_numeric(
                dataframe[member_sensor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            combined = (1.0 - float(blend)) * existing + float(blend) * generated
            combined = np.clip(combined, float(member_profile.lower_bound), float(member_profile.upper_bound))
            combined = np.clip(combined, float(member_profile.min_value), float(member_profile.max_value))

            dataframe[member_sensor] = combined

    def _is_fault_excluded_sensor(self, sensor_name: str) -> bool:
        return str(sensor_name).strip() in self.fault_excluded_sensors

    def _apply_local_normal_noise(
        self,
        dataframe: pd.DataFrame,
        *,
        idx: np.ndarray,
        profiles: Dict[str, SensorRichProfile],
        base_scale: float = 0.08,
        derivative_scale: float = 0.05,
    ) -> None:
        """
        Add small local jitter to preserve realistic short-range fluctuation
        in normal windows without blowing up stable sensors.
        """
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return

        for sensor in self.sensors:
            if sensor not in dataframe.columns:
                continue

            prof = profiles.get(sensor)
            if prof is None:
                continue

            x = pd.to_numeric(
                dataframe.loc[idx, sensor],
                errors="coerce",
            ).to_numpy(dtype=float, copy=True)

            mask = np.isfinite(x)
            if mask.sum() == 0:
                continue

            cur = x[mask]
            std = max(float(prof.std), 1e-6)

            diffs = np.abs(np.diff(np.r_[cur[0], cur]))
            positive_diffs = diffs[diffs > 0]
            diff_base = float(np.median(positive_diffs)) if positive_diffs.size else std * 0.05
            diff_base = max(diff_base, 1e-6)

            diff_weight = np.clip(diffs / diff_base, 0.0, 3.0)
            noise_sd = std * (base_scale + derivative_scale * diff_weight)

            cur = cur + self.rng.normal(0.0, noise_sd, size=cur.shape[0])

            cur = np.clip(cur, float(prof.lower_bound), float(prof.upper_bound))
            cur = np.clip(cur, float(prof.min_value), float(prof.max_value))

            x_new = x.copy()
            x_new[mask] = cur
            dataframe.loc[idx, sensor] = x_new

    def _apply_sensor_variance_floor(
        self,
        dataframe: pd.DataFrame,
        *,
        idx: np.ndarray,
        profiles: Dict[str, SensorRichProfile],
        std_floor_ratio: float = 0.92,
        max_extra_noise_ratio: float = 0.20,
    ) -> None:
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return

        for sensor in self.sensors:
            if sensor not in dataframe.columns:
                continue

            prof = profiles.get(sensor)
            if prof is None:
                continue

            target_std = max(float(prof.std), 1e-6)
            floor_std = target_std * float(std_floor_ratio)

            x = pd.to_numeric(dataframe.loc[idx, sensor], errors="coerce").to_numpy(dtype=float, copy=True)
            mask = np.isfinite(x)
            if mask.sum() < 3:
                continue

            cur = x[mask]
            cur_std = float(np.std(cur, ddof=1)) if cur.size > 1 else 0.0

            if cur_std >= floor_std:
                continue

            needed = max(floor_std**2 - cur_std**2, 0.0) ** 0.5
            needed = min(needed, target_std * float(max_extra_noise_ratio))

            cur = cur + self.rng.normal(0.0, needed, size=cur.shape[0])
            cur = np.clip(cur, float(prof.lower_bound), float(prof.upper_bound))
            cur = np.clip(cur, float(prof.min_value), float(prof.max_value))

            x_new = x.copy()
            x_new[mask] = cur
            dataframe.loc[idx, sensor] = x_new

    def _apply_sensor_mean_anchor(
        self,
        dataframe: pd.DataFrame,
        *,
        idx: np.ndarray,
        profiles: Dict[str, SensorRichProfile],
        blend: float = 0.18,
        trigger_std_mult: float = 0.35,
    ) -> None:
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return

        for sensor in self.sensors:
            if sensor not in dataframe.columns:
                continue

            prof = profiles.get(sensor)
            if prof is None:
                continue

            target_mean = float(prof.mean)
            target_std = max(float(prof.std), 1e-6)

            x = pd.to_numeric(dataframe.loc[idx, sensor], errors="coerce").to_numpy(dtype=float, copy=True)
            mask = np.isfinite(x)
            if mask.sum() < 3:
                continue

            cur = x[mask]
            cur_mean = float(np.mean(cur))

            if abs(cur_mean - target_mean) < (trigger_std_mult * target_std):
                continue

            shift = (target_mean - cur_mean) * float(blend)
            cur = cur + shift

            cur = np.clip(cur, float(prof.lower_bound), float(prof.upper_bound))
            cur = np.clip(cur, float(prof.min_value), float(prof.max_value))

            x_new = x.copy()
            x_new[mask] = cur
            dataframe.loc[idx, sensor] = x_new



    def _apply_buildup_transition_noise(
        self,
        values: np.ndarray,
        *,
        profile: SensorRichProfile,
        fault_type: FaultType,
        magnitude: float,
    ) -> np.ndarray:
        """
        Make buildup behave like a transition:
        - keep it closer to normal than failure
        - increase perturbation as progress increases
        - tie some of the noise to local movement
        """
        out = values.astype(float, copy=True)
        n = int(out.shape[0])

        if n == 0:
            return out

        std = max(float(profile.std), 1e-6)
        progress = np.linspace(0.0, 1.0, n)

        fault_component = self._inject_fault(
            out.copy(),
            fault_type,
            magnitude=max(0.05, float(magnitude) * 0.60),
            profile=profile,
        )

        # Gradually blend toward the fault-shaped path
        blend_weight = 0.15 + 0.70 * progress
        out = out + (fault_component - out) * blend_weight

        # Add transition noise linked to local change
        diffs = np.abs(np.diff(np.r_[out[0], out]))
        positive_diffs = diffs[diffs > 0]
        diff_base = float(np.median(positive_diffs)) if positive_diffs.size else std * 0.05
        diff_base = max(diff_base, 1e-6)

        diff_weight = np.clip(diffs / diff_base, 0.0, 3.0)
        noise_sd = std * (0.04 + 0.10 * progress + 0.04 * diff_weight)

        out = out + self.rng.normal(0.0, noise_sd, size=n)

        out = np.clip(out, float(profile.lower_bound), float(profile.upper_bound))
        out = np.clip(out, float(profile.min_value), float(profile.max_value))
        return out

    def _first_consecutive_true_index(
        self,
        mask: np.ndarray,
        min_run: int = 3,
    ) -> Optional[int]:
        """
        Return the first index where a boolean mask is True for `min_run`
        consecutive rows. Returns None if no such run exists.
        """
        run_len = 0
        for idx, flag in enumerate(np.asarray(mask, dtype=bool)):
            if bool(flag):
                run_len += 1
                if run_len >= int(min_run):
                    return int(idx - run_len + 1)
            else:
                run_len = 0
        return None

    def _build_episode_truth_columns(
        self,
        dataframe: pd.DataFrame,
        *,
        spec: EpisodeSpec,
        episode_id: Optional[int],
        buildup_start_idx: int,
        failure_start_idx: int,
        is_fault_episode: bool,
        observable_zscore_threshold: float = 2.5,
        observable_min_consecutive: int = 3,
    ) -> pd.DataFrame:
        """
        Stamp row-level truth and observable-onset metadata for a generated episode.

        Definitions:
        - fault_onset_truth_row: first row where generator begins injected buildup behavior
        - failure_truth_row: first BROKEN / abnormal row
        - observable_onset_row: first row where the primary sensor exceeds the
          configured z-score threshold for N consecutive rows
        """
        out = dataframe.copy()
        n_rows = len(out)
        row_idx = np.arange(n_rows, dtype=int)

        # primary-sensor observable thresholding vs normal profile
        primary_profile = self.normal[spec.primary_sensor]
        primary_values = pd.to_numeric(
            out[spec.primary_sensor],
            errors="coerce",
        ).to_numpy(dtype=float, copy=True)

        normal_mean = float(primary_profile.mean)
        normal_std = max(float(primary_profile.std), 1e-6)

        primary_abs_z = np.abs((primary_values - normal_mean) / normal_std)
        primary_threshold_crossed = np.isfinite(primary_abs_z) & (
            primary_abs_z >= float(observable_zscore_threshold)
        )

        observable_onset_row: Optional[int] = None
        if is_fault_episode:
            search_mask = row_idx >= int(buildup_start_idx)
            observable_onset_row = self._first_consecutive_true_index(
                primary_threshold_crossed & search_mask,
                min_run=int(observable_min_consecutive),
            )

        # repeated scalar metadata
        if episode_id is not None:
            out["meta__episode_id"] = pd.Series(
                [int(episode_id)] * n_rows,
                index=out.index,
                dtype="Int64",
            )
        else:
            out["meta__episode_id"] = pd.Series(
                [pd.NA] * n_rows,
                index=out.index,
                dtype="Int64",
            )

        out["meta__is_fault_episode"] = bool(is_fault_episode)
        out["meta__phase_truth"] = out["phase"].astype(str)
        out["meta__primary_sensor"] = str(spec.primary_sensor)
        out["meta__primary_fault_type"] = str(spec.primary_fault_type)
        out["meta__primary_magnitude"] = float(spec.magnitude)

        # row-level primary-sensor observability
        out["meta__primary_sensor_abs_zscore"] = primary_abs_z
        out["meta__primary_sensor_threshold_crossed"] = primary_threshold_crossed

        # generator-truth onset
        out["meta__fault_onset_truth_flag"] = False
        out["meta__fault_onset_truth_row"] = pd.Series(
            [int(buildup_start_idx)] * n_rows if is_fault_episode else [pd.NA] * n_rows,
            index=out.index,
            dtype="Int64",
        )

        if is_fault_episode and 0 <= int(buildup_start_idx) < n_rows:
            out.loc[out.index[int(buildup_start_idx)], "meta__fault_onset_truth_flag"] = True

        # failure row
        out["meta__failure_truth_flag"] = False
        out["meta__failure_truth_row"] = pd.Series(
            [int(failure_start_idx)] * n_rows if is_fault_episode else [pd.NA] * n_rows,
            index=out.index,
            dtype="Int64",
        )

        if is_fault_episode and 0 <= int(failure_start_idx) < n_rows:
            out.loc[out.index[int(failure_start_idx)], "meta__failure_truth_flag"] = True

        # observable onset row
        out["meta__observable_onset_flag"] = False
        out["meta__observable_onset_row"] = pd.Series(
            [int(observable_onset_row)] * n_rows if observable_onset_row is not None else [pd.NA] * n_rows,
            index=out.index,
            dtype="Int64",
        )

        if observable_onset_row is not None and 0 <= int(observable_onset_row) < n_rows:
            out.loc[out.index[int(observable_onset_row)], "meta__observable_onset_flag"] = True

        # buildup progress only inside buildup window
        buildup_progress = np.zeros(n_rows, dtype=float)
        if is_fault_episode and int(failure_start_idx) > int(buildup_start_idx):
            buildup_len = int(failure_start_idx) - int(buildup_start_idx)
            buildup_progress[int(buildup_start_idx):int(failure_start_idx)] = np.linspace(
                0.0,
                1.0,
                buildup_len,
                endpoint=False,
            )
        out["meta__buildup_progress"] = buildup_progress

        # lead-time style helpers
        rows_until_failure = pd.Series([pd.NA] * n_rows, index=out.index, dtype="Int64")
        rows_since_truth_onset = pd.Series([pd.NA] * n_rows, index=out.index, dtype="Int64")
        rows_since_observable_onset = pd.Series([pd.NA] * n_rows, index=out.index, dtype="Int64")

        if is_fault_episode:
            pre_failure_mask = row_idx <= int(failure_start_idx)
            rows_until_failure.loc[out.index[pre_failure_mask]] = (
                int(failure_start_idx) - row_idx[pre_failure_mask]
            ).astype(int)

            post_truth_mask = row_idx >= int(buildup_start_idx)
            rows_since_truth_onset.loc[out.index[post_truth_mask]] = (
                row_idx[post_truth_mask] - int(buildup_start_idx)
            ).astype(int)

            if observable_onset_row is not None:
                post_observable_mask = row_idx >= int(observable_onset_row)
                rows_since_observable_onset.loc[out.index[post_observable_mask]] = (
                    row_idx[post_observable_mask] - int(observable_onset_row)
                ).astype(int)

        out["meta__rows_until_failure"] = rows_until_failure
        out["meta__rows_since_truth_onset"] = rows_since_truth_onset
        out["meta__rows_since_observable_onset"] = rows_since_observable_onset

        return out

    # -------------------------
    # Normal batch
    # -------------------------
    def generate_normal_batch(
        self,
        n_rows: int,
        smoothing: float = 0.50,
        add_local_noise: bool = True,
    ) -> pd.DataFrame:
        data: Dict[str, np.ndarray] = {}
        for sensor in self.sensors:
            data[sensor] = self._sample_series(
                self.normal[sensor],
                n_rows,
                smoothing=smoothing,
            )

        dataframe = pd.DataFrame(data)
        dataframe["stream_state"] = "normal"

        # light group co-movement
        for group_name in sorted(set(self.sensor_to_group.values())):
            self._apply_group_driver(dataframe, group_name, self.normal, strength=0.06)
            self._apply_group_correlated_residuals(dataframe, group_name, self.normal, strength=0.45)

        if add_local_noise and n_rows > 0:
            self._apply_local_normal_noise(
                dataframe,
                idx=dataframe.index.to_numpy(),
                profiles=self.normal,
                base_scale=0.08,
                derivative_scale=0.05,
            )

        if n_rows > 0:
            self._apply_sensor_variance_floor(
                dataframe,
                idx=dataframe.index.to_numpy(),
                profiles=self.normal,
                std_floor_ratio=1.00,
                max_extra_noise_ratio=0.35,
            )

            self._apply_sensor_mean_anchor(
                dataframe,
                idx=dataframe.index.to_numpy(),
                profiles=self.normal,
                blend=0.16,
                trigger_std_mult=0.30,
            )

            normal_top_pair_cfg = self._get_corr_tuning_block(
                "normal",
                "top_pairwise_overlay",
                defaults={
                    "strength": 0.16,
                    "top_n": 120,
                    "min_abs_corr": 0.08,
                    "smooth_alpha": 0.90,
                },
            )

            normal_bridge_cfg = self._get_corr_tuning_block(
                "normal",
                "bridge_pair_generation",
                defaults={
                    "bridge_pairs": BRIDGE_PAIRS,
                    "blend": 0.96,
                    "min_abs_corr": 0.20,
                    "residual_floor": 0.03,
                    "smooth_alpha": 0.95,
                },
            )

            self._apply_top_pairwise_overlay(
                dataframe,
                self.normal,
                min_abs_corr=float(normal_top_pair_cfg["min_abs_corr"]),
                top_n=int(normal_top_pair_cfg["top_n"]),
                strength=float(normal_top_pair_cfg["strength"]),
                smooth_alpha=float(normal_top_pair_cfg["smooth_alpha"]),
            )

            if self.correlation_hotspot_clusters:
                for cluster in self.correlation_hotspot_clusters:
                    family_name = self._classify_cluster_family(cluster)

                    named_cfg = self._get_family_corr_tuning_block(
                        "normal",
                        family_name,
                        "named_cluster_overlay",
                        defaults={
                            "strength": 0.16,
                            "smooth_alpha": 0.94,
                        },
                    )

                    anchor_cfg = self._get_family_corr_tuning_block(
                        "normal",
                        family_name,
                        "anchor_cluster_generation",
                        defaults={
                            "blend": 0.90,
                            "min_abs_corr": 0.15,
                            "residual_floor": 0.04,
                            "smooth_alpha": 0.94,
                        },
                    )

                    self._apply_named_cluster_overlay(
                        dataframe,
                        self.normal,
                        clusters=[cluster],
                        strength=float(named_cfg["strength"]),
                        smooth_alpha=float(named_cfg["smooth_alpha"]),
                    )

                    self._apply_anchor_cluster_generation(
                        dataframe,
                        self.normal,
                        clusters=[cluster],
                        blend=float(anchor_cfg["blend"]),
                        min_abs_corr=float(anchor_cfg["min_abs_corr"]),
                        residual_floor=float(anchor_cfg["residual_floor"]),
                        smooth_alpha=float(anchor_cfg["smooth_alpha"]),
                    )

            self._apply_bridge_pair_generation(
                dataframe,
                self.normal,
                bridge_pairs=self._get_bridge_pairs_from_tuning("normal"),
                blend=float(normal_bridge_cfg["blend"]),
                min_abs_corr=float(normal_bridge_cfg["min_abs_corr"]),
                residual_floor=float(normal_bridge_cfg["residual_floor"]),
                smooth_alpha=float(normal_bridge_cfg["smooth_alpha"]),
            )
            

        
        return dataframe

    # -------------------------
    # Fault primitives
    # -------------------------
    def _inject_fault(self, values: np.ndarray, fault_type: FaultType, magnitude: float, profile: SensorRichProfile) -> np.ndarray:
        n = len(values)
        std = max(float(profile.std), 1e-6)

        if fault_type == "drift_up":
            values = values + np.linspace(0, magnitude * std * 4.0, n)

        elif fault_type == "drift_down":
            values = values - np.linspace(0, magnitude * std * 4.0, n)

        elif fault_type == "spike":
            k = max(1, min(6, n // 10 + 1))
            idx = self.rng.choice(np.arange(n), size=k, replace=False)
            for i in idx:
                direction = self.rng.choice([-1.0, 1.0])
                values[i] = values[i] + direction * magnitude * std * 8.0

        elif fault_type == "stuck_constant":
            stuck = float(np.nanmedian(values))
            values[:] = stuck

        elif fault_type == "variance_burst":
            values = values + self.rng.normal(0.0, magnitude * std * 3.0, size=n)

        elif fault_type == "step_shift":
            direction = self.rng.choice([-1.0, 1.0])
            values = values + direction * magnitude * std * 5.0

        elif fault_type == "intermittent_dropout":
            mask = self.rng.random(n) < 0.15
            low_value = max(float(profile.min_value), float(profile.mean) - magnitude * std * 6.0)
            values[mask] = low_value

        elif fault_type == "sawtooth":
            cycle = 20
            x = np.arange(n)
            wave = ((x % cycle) / cycle) * 2.0 - 1.0
            values = values + wave * magnitude * std * 3.0

        values = np.clip(values, float(profile.lower_bound), float(profile.upper_bound))
        values = np.clip(values, float(profile.min_value), float(profile.max_value))
        return values
    
    def _calibrate_block_mean_std(
        self,
        df: pd.DataFrame,
        *,
        idx: np.ndarray,
        profiles: Dict[str, SensorRichProfile],
        state_name: str,
    ) -> None:
        """
        In-place calibration on df.loc[idx, sensor] for each sensor:
        - mean forced into [target_mean - k*target_std, target_mean + k*target_std]
        - std clamped into [lo_ratio*target_std, hi_ratio*target_std]
        Preserves NaNs. Re-clips to profile bounds afterward.
        """
        if self.state_calibration_targets is None:
            return

        targets_for_state = self.state_calibration_targets.get(str(state_name), {})
        if not targets_for_state:
            return

        k = float(self.mean_within_k_std)
        lo_ratio, hi_ratio = self.std_ratio_bounds

        for sensor in self.sensors:
            if sensor not in df.columns:
                continue
            if sensor not in targets_for_state:
                continue

            t = targets_for_state[sensor] or {}
            if "mean" not in t or "std" not in t:
                continue

            target_mean = float(t["mean"])
            target_std = max(float(t["std"]), 1e-6)

            # current values (ignore NaNs)
            x = pd.to_numeric(df.loc[idx, sensor], errors="coerce").to_numpy(dtype=float, copy=True)
            mask = np.isfinite(x)
            if mask.sum() == 0:
                continue

            cur = x[mask]
            cur_mean = float(cur.mean())
            cur_std = float(cur.std(ddof=1)) if cur.size > 1 else 0.0

            # --- mean clamp ---
            lo_mean = target_mean - k * target_std
            hi_mean = target_mean + k * target_std
            desired_mean = min(max(cur_mean, lo_mean), hi_mean)

            # --- std clamp (ratio band around target std) ---
            desired_std = target_std
            if cur_std > 1e-12:
                min_std = lo_ratio * target_std
                max_std = hi_ratio * target_std
                desired_std = min(max(cur_std, min_std), max_std)
            else:
                # if nearly constant, allow only small variance
                desired_std = lo_ratio * target_std

            # --- affine adjust on non-null values only ---
            if cur_std > 1e-12:
                cur_adj = (cur - cur_mean) / cur_std
                cur_adj = cur_adj * desired_std + desired_mean
            else:
                cur_adj = np.full_like(cur, desired_mean, dtype=float)

            x_new = x.copy()
            x_new[mask] = cur_adj

            # re-clip to profile bounds
            prof = profiles.get(sensor)
            if prof is not None:
                x_new = np.clip(x_new, float(prof.lower_bound), float(prof.upper_bound))
                x_new = np.clip(x_new, float(prof.min_value), float(prof.max_value))

            df.loc[idx, sensor] = x_new


    def _calibrate_by_phase(self, df: pd.DataFrame) -> None:
        """
        Calibrate per-phase using the closest empirical state target.

        normal  -> normal
        buildup -> buildup targets if provided, else normal
        abnormal-> abnormal
        recovery-> recovery
        """
        if self.state_calibration_targets is None:
            return
        if "phase" not in df.columns:
            return

        phase_map = {
            "normal": "normal",
            "buildup": "buildup" if "buildup" in self.state_calibration_targets else "normal",
            "abnormal": "abnormal",
            "recovery": "recovery",
        }

        for phase_val, state_name in phase_map.items():
            mask = df["phase"].astype(str).eq(str(phase_val))
            idx = df.index[mask].to_numpy()
            if idx.size == 0:
                continue

            if state_name in {"normal", "buildup"}:
                profs = self.normal
            elif state_name == "recovery":
                profs = self.recovery
            else:
                profs = self.abnormal

            self._calibrate_block_mean_std(
                df,
                idx=idx,
                profiles=profs,
                state_name=state_name,
            )
    # -------------------------
    # Abnormal episode (buildup + failure + recovery)
    # -------------------------
    def generate_episode(
        self,
        spec: EpisodeSpec,
        *,
        episode_id: Optional[int] = None,
        observable_zscore_threshold: float = 2.5,
        observable_min_consecutive: int = 3,
    ) -> pd.DataFrame:
        
        if spec.primary_sensor not in self.sensors:
            raise ValueError(f"Unknown primary sensor: {spec.primary_sensor}")

        if self._is_fault_excluded_sensor(spec.primary_sensor):
            raise ValueError(
                f"Primary fault sensor '{spec.primary_sensor}' is excluded "
                f"(configured fault_excluded_sensors)."
            )

        is_fault_episode = int(spec.failure) > 0

        normal_before = int(max(0, spec.normal_before))
        buildup = int(max(0, spec.buildup)) if is_fault_episode else 0
        failure = int(max(0, spec.failure)) if is_fault_episode else 0
        recovery = int(max(0, spec.recovery)) if is_fault_episode else 0
        normal_after = int(max(0, spec.normal_after))

        total = normal_before + buildup + failure + recovery + normal_after
        if total <= 0:
            total = 1
            normal_before = 1

        dataframe = self.generate_normal_batch(total)

        b0 = normal_before
        f0 = b0 + buildup
        r0 = f0 + failure
        n0 = r0 + recovery

        dataframe["phase"] = "normal"
        dataframe["stream_state"] = "normal"

        if is_fault_episode:
            if buildup > 0:
                dataframe.loc[b0:f0 - 1, "phase"] = "buildup"
                # keep surfaced status collapsed to NORMAL
                dataframe.loc[b0:f0 - 1, "stream_state"] = "normal"

            if failure > 0:
                dataframe.loc[f0:r0 - 1, "phase"] = "abnormal"
                dataframe.loc[f0:r0 - 1, "stream_state"] = "broken"

            if recovery > 0:
                dataframe.loc[r0:n0 - 1, "phase"] = "recovery"
                dataframe.loc[r0:n0 - 1, "stream_state"] = "recovering"

        p_norm = self.normal[spec.primary_sensor]

        # Add extra local noise to the explicitly normal windows
        if normal_before > 0:
            idx = dataframe.index[:normal_before].to_numpy()
            self._apply_local_normal_noise(
                dataframe,
                idx=idx,
                profiles=self.normal,
                base_scale=0.08,
                derivative_scale=0.05,
            )

        if normal_after > 0:
            idx = dataframe.index[n0:n0 + normal_after].to_numpy()
            self._apply_local_normal_noise(
                dataframe,
                idx=idx,
                profiles=self.normal,
                base_scale=0.08,
                derivative_scale=0.05,
            )

        # Buildup should stay closer to normal, but trend toward failure behavior
        if is_fault_episode and buildup > 0:
            seg = dataframe.loc[b0:f0 - 1, spec.primary_sensor].to_numpy(copy=True)
            seg = self._apply_buildup_transition_noise(
                seg,
                profile=p_norm,
                fault_type=spec.primary_fault_type,
                magnitude=spec.magnitude,
            )
            dataframe.loc[b0:f0 - 1, spec.primary_sensor] = seg

        # Failure window = the actual BROKEN marker
        if is_fault_episode and failure > 0:
            p_ab = self.abnormal.get(spec.primary_sensor, p_norm)
            failure_type: FaultType = (
                "step_shift"
                if spec.primary_fault_type in {"drift_up", "drift_down"}
                else spec.primary_fault_type
            )

            seg = dataframe.loc[f0:r0 - 1, spec.primary_sensor].to_numpy(copy=True)
            if seg.size:
                seg = self._inject_fault(seg, failure_type, spec.magnitude, p_ab)
                dataframe.loc[f0:r0 - 1, spec.primary_sensor] = seg

        # Propagate paired effects across buildup + failure
        if is_fault_episode and (buildup + failure) > 0:
            prop_start_base = b0
            prop_end_base = r0

            for link in self.propagation.get(spec.primary_sensor, []):
                sec = str(link["secondary"])

                if sec not in dataframe.columns:
                    continue

                if self._is_fault_excluded_sensor(sec):
                    continue

                strength = float(link["strength"])
                lag = int(link["lag"])

                start = prop_start_base + lag
                end = prop_end_base
                if start >= end:
                    continue

                sec_profile = self.abnormal.get(sec, self.normal[sec])

                sec_values = dataframe.loc[start:end - 1, sec].to_numpy(copy=True)
                if sec_values.size == 0:
                    continue

                recommended_secondary_fault = str(
                    link.get("recommended_secondary_fault", "variance_burst")
                )
                if recommended_secondary_fault not in {
                    "drift_up",
                    "drift_down",
                    "spike",
                    "stuck_constant",
                    "variance_burst",
                    "step_shift",
                    "intermittent_dropout",
                    "sawtooth",
                }:
                    recommended_secondary_fault = "variance_burst"

                sec_values = self._inject_fault(
                    sec_values,
                    recommended_secondary_fault,
                    max(0.05, spec.magnitude * strength),
                    sec_profile,
                )
                dataframe.loc[start:end - 1, sec] = sec_values

        # abnormal-window group boost must be written back explicitly
        if is_fault_episode and failure > 0:
            group_name = self.sensor_to_group.get(spec.primary_sensor)
            if group_name is not None and r0 > f0:
                df_window = dataframe.iloc[f0:r0].copy()
                self._apply_group_driver(
                    df_window,
                    group_name,
                    self.abnormal,
                    strength=0.18,
                )
                self._apply_group_correlated_residuals(
                    df_window,
                    group_name,
                    self.abnormal,
                    strength=0.52,
                )
                self._apply_top_pairwise_overlay(
                    df_window,
                    self.abnormal,
                    min_abs_corr=0.08,
                    top_n=40,
                    strength=0.18,
                    smooth_alpha=0.85,
                )
                dataframe.loc[df_window.index, df_window.columns] = df_window

        self._calibrate_by_phase(dataframe)

        # restore some variance after calibration so phase calibration
        # does not leave normal/recovery blocks too narrow
        normal_idx = dataframe.index[dataframe["phase"].astype(str).eq("normal")].to_numpy()
        if normal_idx.size > 0:
            self._apply_sensor_variance_floor(
                dataframe,
                idx=normal_idx,
                profiles=self.normal,
                std_floor_ratio=1.02,
                max_extra_noise_ratio=0.40,
            )
            self._apply_sensor_mean_anchor(
                dataframe,
                idx=normal_idx,
                profiles=self.normal,
                blend=0.18,
                trigger_std_mult=0.25,
            )

            df_window = dataframe.loc[normal_idx].copy()

            normal_top_pair_cfg = self._get_corr_tuning_block(
                "normal",
                "top_pairwise_overlay",
                defaults={
                    "strength": 0.16,
                    "top_n": 120,
                    "min_abs_corr": 0.08,
                    "smooth_alpha": 0.92,
                },
            )

            normal_bridge_cfg = self._get_corr_tuning_block(
                "normal",
                "bridge_pair_generation",
                defaults={
                    "bridge_pairs": BRIDGE_PAIRS,
                    "blend": 0.97,
                    "min_abs_corr": 0.20,
                    "residual_floor": 0.03,
                    "smooth_alpha": 0.96,
                },
            )

            self._apply_top_pairwise_overlay(
                df_window,
                self.normal,
                min_abs_corr=float(normal_top_pair_cfg["min_abs_corr"]),
                top_n=int(normal_top_pair_cfg["top_n"]),
                strength=float(normal_top_pair_cfg["strength"]),
                smooth_alpha=float(normal_top_pair_cfg["smooth_alpha"]),
            )

            if self.correlation_hotspot_clusters:
                for cluster in self.correlation_hotspot_clusters:
                    family_name = self._classify_cluster_family(cluster)

                    named_cfg = self._get_family_corr_tuning_block(
                        "normal",
                        family_name,
                        "named_cluster_overlay",
                        defaults={
                            "strength": 0.18,
                            "smooth_alpha": 0.95,
                        },
                    )

                    anchor_cfg = self._get_family_corr_tuning_block(
                        "normal",
                        family_name,
                        "anchor_cluster_generation",
                        defaults={
                            "blend": 0.92,
                            "min_abs_corr": 0.15,
                            "residual_floor": 0.04,
                            "smooth_alpha": 0.95,
                        },
                    )

                    self._apply_named_cluster_overlay(
                        df_window,
                        self.normal,
                        clusters=[cluster],
                        strength=float(named_cfg["strength"]),
                        smooth_alpha=float(named_cfg["smooth_alpha"]),
                    )

                    self._apply_anchor_cluster_generation(
                        df_window,
                        self.normal,
                        clusters=[cluster],
                        blend=float(anchor_cfg["blend"]),
                        min_abs_corr=float(anchor_cfg["min_abs_corr"]),
                        residual_floor=float(anchor_cfg["residual_floor"]),
                        smooth_alpha=float(anchor_cfg["smooth_alpha"]),
                    )

            self._apply_bridge_pair_generation(
                df_window,
                self.normal,
                bridge_pairs=self._get_bridge_pairs_from_tuning("normal"),
                blend=float(normal_bridge_cfg["blend"]),
                min_abs_corr=float(normal_bridge_cfg["min_abs_corr"]),
                residual_floor=float(normal_bridge_cfg["residual_floor"]),
                smooth_alpha=float(normal_bridge_cfg["smooth_alpha"]),
            )

            self._apply_priority_pair_generation(
                df_window,
                self.normal,
                pair_specs=self._get_priority_pair_specs_from_tuning("normal"),
            )

            dataframe.loc[df_window.index, df_window.columns] = df_window

        buildup_idx = dataframe.index[dataframe["phase"].astype(str).eq("buildup")].to_numpy()
        if buildup_idx.size > 0:
            self._apply_sensor_variance_floor(
                dataframe,
                idx=buildup_idx,
                profiles=self.normal,
                std_floor_ratio=0.96,
                max_extra_noise_ratio=0.25,
            )

            df_window = dataframe.loc[buildup_idx].copy()

            buildup_top_pair_cfg = self._get_corr_tuning_block(
                "normal",
                "top_pairwise_overlay",
                defaults={
                    "strength": 0.16,
                    "top_n": 80,
                    "min_abs_corr": 0.08,
                    "smooth_alpha": 0.90,
                },
            )

            buildup_bridge_cfg = self._get_corr_tuning_block(
                "normal",
                "bridge_pair_generation",
                defaults={
                    "bridge_pairs": BRIDGE_PAIRS,
                    "blend": 0.97,
                    "min_abs_corr": 0.20,
                    "residual_floor": 0.03,
                    "smooth_alpha": 0.96,
                },
            )

            self._apply_top_pairwise_overlay(
                df_window,
                self.normal,
                min_abs_corr=float(buildup_top_pair_cfg["min_abs_corr"]),
                top_n=int(buildup_top_pair_cfg["top_n"]),
                strength=float(buildup_top_pair_cfg["strength"]),
                smooth_alpha=float(buildup_top_pair_cfg["smooth_alpha"]),
            )

            if self.correlation_hotspot_clusters:
                for cluster in self.correlation_hotspot_clusters:
                    family_name = self._classify_cluster_family(cluster)

                    named_cfg = self._get_family_corr_tuning_block(
                        "normal",
                        family_name,
                        "named_cluster_overlay",
                        defaults={
                            "strength": 0.18,
                            "smooth_alpha": 0.95,
                        },
                    )

                    anchor_cfg = self._get_family_corr_tuning_block(
                        "normal",
                        family_name,
                        "anchor_cluster_generation",
                        defaults={
                            "blend": 0.92,
                            "min_abs_corr": 0.15,
                            "residual_floor": 0.04,
                            "smooth_alpha": 0.95,
                        },
                    )

                    self._apply_named_cluster_overlay(
                        df_window,
                        self.normal,
                        clusters=[cluster],
                        strength=float(named_cfg["strength"]),
                        smooth_alpha=float(named_cfg["smooth_alpha"]),
                    )

                    self._apply_anchor_cluster_generation(
                        df_window,
                        self.normal,
                        clusters=[cluster],
                        blend=float(anchor_cfg["blend"]),
                        min_abs_corr=float(anchor_cfg["min_abs_corr"]),
                        residual_floor=float(anchor_cfg["residual_floor"]),
                        smooth_alpha=float(anchor_cfg["smooth_alpha"]),
                    )

            self._apply_bridge_pair_generation(
                df_window,
                self.normal,
                bridge_pairs=self._get_bridge_pairs_from_tuning("normal"),
                blend=float(buildup_bridge_cfg["blend"]),
                min_abs_corr=float(buildup_bridge_cfg["min_abs_corr"]),
                residual_floor=float(buildup_bridge_cfg["residual_floor"]),
                smooth_alpha=float(buildup_bridge_cfg["smooth_alpha"]),
            )

            self._apply_priority_pair_generation(
                df_window,
                self.normal,
                pair_specs=self._get_priority_pair_specs_from_tuning("normal"),
            )

            dataframe.loc[df_window.index, df_window.columns] = df_window

        recovery_idx = dataframe.index[dataframe["phase"].astype(str).eq("recovery")].to_numpy()
        if recovery_idx.size > 0:
            self._apply_sensor_variance_floor(
                dataframe,
                idx=recovery_idx,
                profiles=self.recovery,
                std_floor_ratio=0.97,
                max_extra_noise_ratio=0.25,
            )
            self._apply_sensor_mean_anchor(
                dataframe,
                idx=recovery_idx,
                profiles=self.recovery,
                blend=0.15,
                trigger_std_mult=0.30,
            )

            df_window = dataframe.loc[recovery_idx].copy()

            recovery_top_pair_cfg = self._get_corr_tuning_block(
                "recovery",
                "top_pairwise_overlay",
                defaults={
                    "strength": 0.14,
                    "top_n": 100,
                    "min_abs_corr": 0.08,
                    "smooth_alpha": 0.90,
                },
            )

            recovery_bridge_cfg = self._get_corr_tuning_block(
                "recovery",
                "bridge_pair_generation",
                defaults={
                    "bridge_pairs": BRIDGE_PAIRS,
                    "blend": 0.90,
                    "min_abs_corr": 0.20,
                    "residual_floor": 0.05,
                    "smooth_alpha": 0.93,
                },
            )

            self._apply_top_pairwise_overlay(
                df_window,
                self.recovery,
                min_abs_corr=float(recovery_top_pair_cfg["min_abs_corr"]),
                top_n=int(recovery_top_pair_cfg["top_n"]),
                strength=float(recovery_top_pair_cfg["strength"]),
                smooth_alpha=float(recovery_top_pair_cfg["smooth_alpha"]),
            )

            if self.correlation_hotspot_clusters:
                for cluster in self.correlation_hotspot_clusters:
                    family_name = self._classify_cluster_family(cluster)

                    named_cfg = self._get_family_corr_tuning_block(
                        "recovery",
                        family_name,
                        "named_cluster_overlay",
                        defaults={
                            "strength": 0.14,
                            "smooth_alpha": 0.93,
                        },
                    )

                    anchor_cfg = self._get_family_corr_tuning_block(
                        "recovery",
                        family_name,
                        "anchor_cluster_generation",
                        defaults={
                            "blend": 0.82,
                            "min_abs_corr": 0.15,
                            "residual_floor": 0.06,
                            "smooth_alpha": 0.92,
                        },
                    )

                    self._apply_named_cluster_overlay(
                        df_window,
                        self.recovery,
                        clusters=[cluster],
                        strength=float(named_cfg["strength"]),
                        smooth_alpha=float(named_cfg["smooth_alpha"]),
                    )

                    self._apply_anchor_cluster_generation(
                        df_window,
                        self.recovery,
                        clusters=[cluster],
                        blend=float(anchor_cfg["blend"]),
                        min_abs_corr=float(anchor_cfg["min_abs_corr"]),
                        residual_floor=float(anchor_cfg["residual_floor"]),
                        smooth_alpha=float(anchor_cfg["smooth_alpha"]),
                    )

            self._apply_bridge_pair_generation(
                df_window,
                self.recovery,
                bridge_pairs=self._get_bridge_pairs_from_tuning("recovery"),
                blend=float(recovery_bridge_cfg["blend"]),
                min_abs_corr=float(recovery_bridge_cfg["min_abs_corr"]),
                residual_floor=float(recovery_bridge_cfg["residual_floor"]),
                smooth_alpha=float(recovery_bridge_cfg["smooth_alpha"]),
            )

            self._apply_priority_pair_generation(
                df_window,
                self.recovery,
                pair_specs=self._get_priority_pair_specs_from_tuning("recovery"),
            )

            dataframe.loc[df_window.index, df_window.columns] = df_window

        abnormal_idx = dataframe.index[dataframe["phase"].astype(str).eq("abnormal")].to_numpy()
        if abnormal_idx.size > 0:
            df_window = dataframe.loc[abnormal_idx].copy()
            self._apply_top_pairwise_overlay(
                df_window,
                self.abnormal,
                min_abs_corr=0.08,
                top_n=80,
                strength=0.22,
                smooth_alpha=0.93,
            )
            dataframe.loc[df_window.index, df_window.columns] = df_window

        dataframe = self._build_episode_truth_columns(
            dataframe,
            spec=spec,
            episode_id=episode_id,
            buildup_start_idx=b0,
            failure_start_idx=f0,
            is_fault_episode=is_fault_episode,
            observable_zscore_threshold=observable_zscore_threshold,
            observable_min_consecutive=observable_min_consecutive,
        )

        if self.missingness_spec is not None:
            dataframe = self.apply_missingness(
                dataframe,
                missingness=self.missingness_spec,
                feature_columns=self.sensors,
                rng=self.rng,
            )

        return dataframe

    # -------------------------
    # Apply Missingness Replication
    # -------------------------
    def apply_missingness(
        self,
        dataframe: pd.DataFrame,
        *,
        missingness: MissingnessSpec,
        feature_columns: List[str],
        rng: np.random.Generator,
        # default policy: do NOT remove faults
        apply_to_phases: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Applies clustered masking to attempt to repliace the missingness.

        apply_to_phases maps df["phase"] values -> state_scope to use ("normal"/"recovery"/"abnormal").
        If phase not present, falls back to missingness.state_col_synth if present.
        If neither present, applies GLOBAL missingness to entire df.

        Recommended default:
        normal_before -> normal
        buildup      -> normal   (optional; set to None to skip)
        failure      -> None     (skip so faults are not masked)
        recovery     -> recovery
        normal_after -> normal
        """
        out = dataframe.copy()

        # choose default mapping
        if apply_to_phases is None:
            apply_to_phases = {
                "normal": "normal",
                "buildup": "normal",
                "abnormal": None,
                "recovery": "recovery",
            }

        # only keep sensors that exist
        features = [column for column in feature_columns if column in out.columns]
        if not features:
            return out

        # Case A: phase-driven masking
        if "phase" in out.columns:
            for phase_value, state_scope in apply_to_phases.items():
                mask = out["phase"].astype(str).eq(str(phase_value))
                eligible_idx = out.index[mask].to_numpy()

                if eligible_idx.size == 0:
                    continue

                # skip phases you don't want masked
                if state_scope is None:
                    continue

                # decide whether this sensor uses state-dependent missingness
                # If your gate says False, fall back to global for that sensor even inside state blocks.
                pct_by_state = missingness.missingness_pct_by_state.get(state_scope, {})

                # build per-sensor present counts with per-sensor state-dependent selection
                present_counts: Dict[str, int] = {}
                for feature in features:
                    use_state = bool(missingness.missingness_state_dependent_flag.get(feature, False))
                    pct = pct_by_state.get(feature) if use_state else missingness.missingness_pct_all.get(feature, 0.0)
                    present_counts[feature] = int(round(len(eligible_idx) * (1.0 - float(pct) / 100.0)))
                    present_counts[feature] = max(0, min(len(eligible_idx), present_counts[feature]))

                out = apply_clustered_missingness_mask(
                    out,
                    sensor_cols=features,
                    rng=rng,
                    present_counts=present_counts,
                    eligible_row_idx=eligible_idx,
                    mean_gap_len=3,
                    long_gap_probability=0.15,
                )

            return out

        # Case B: state_col_synth-driven masking
        if missingness.state_col_synth in out.columns:
            for state in missingness.state_list:
                mask = out[missingness.state_col_synth].astype(str).eq(str(state))
                eligible_idx = out.index[mask].to_numpy()
                if eligible_idx.size == 0:
                    continue

                # Optional: skip abnormal if you never want masking there
                if str(state) == "abnormal":
                    continue

                pct_by_state = missingness.missingness_pct_by_state.get(state, {})
                present_counts: Dict[str, int] = {}
                for feature in features:
                    use_state = bool(missingness.missingness_state_dependent_flag.get(feature, False))
                    pct = pct_by_state.get(feature) if use_state else missingness.missingness_pct_all.get(feature, 0.0)
                    present_counts[feature] = int(round(len(eligible_idx) * (1.0 - float(pct) / 100.0)))
                    present_counts[feature] = max(0, min(len(eligible_idx), present_counts[feature]))

                out = apply_clustered_missingness_mask(
                    out,
                    sensor_cols=features,
                    rng=rng,
                    present_counts=present_counts,
                    eligible_row_idx=eligible_idx,
                    mean_gap_len=3,
                    long_gap_probability=0.15,
                )

            return out

        # Case C: global masking on whole frame
        present_counts = build_present_counts_for_block(
            sensors=features,
            n_rows=len(out),
            pct_all=missingness.missingness_pct_all,
            pct_by_state=None,
            use_by_state=False,
        )
        out = apply_clustered_missingness_mask(
            out,
            sensor_cols=features,
            rng=rng,
            present_counts=present_counts,
            eligible_row_idx=out.index.to_numpy(),
            mean_gap_len=3,
            long_gap_probability=0.15,
        )
        return out
        

