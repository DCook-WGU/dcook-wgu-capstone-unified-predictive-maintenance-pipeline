from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from utils.synthetic_profiles import SensorRichProfile


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
        random_seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(int(random_seed))

        self.normal = normal_profiles
        self.abnormal = abnormal_profiles
        self.recovery = recovery_profiles

        self.sensors = sorted(self.normal.keys())

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

    # -------------------------
    # Normal batch
    # -------------------------
    def generate_normal_batch(self, n_rows: int, smoothing: float = 0.85) -> pd.DataFrame:
        data: Dict[str, np.ndarray] = {}
        for sensor in self.sensors:
            data[sensor] = self._sample_series(self.normal[sensor], n_rows, smoothing=smoothing)

        dataframe = pd.DataFrame(data)
        dataframe["stream_state"] = "normal"

        # light group co-movement
        for group_name in sorted(set(self.sensor_to_group.values())):
            self._apply_group_driver(dataframe, group_name, self.normal, strength=0.25)

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

    # -------------------------
    # Abnormal episode (buildup + failure + recovery)
    # -------------------------
    def generate_episode(self, spec: EpisodeSpec) -> pd.DataFrame:
        if spec.primary_sensor not in self.sensors:
            raise ValueError(f"Unknown primary sensor: {spec.primary_sensor}")

        total = spec.normal_before + spec.buildup + spec.failure + spec.recovery + spec.normal_after
        dataframe = self.generate_normal_batch(total)

        dataframe["phase"] = "normal"
        b0 = spec.normal_before
        f0 = b0 + spec.buildup
        r0 = f0 + spec.failure
        n0 = r0 + spec.recovery

        dataframe.loc[b0:f0 - 1, "phase"] = "buildup"
        dataframe.loc[f0:r0 - 1, "phase"] = "abnormal"
        dataframe.loc[r0:n0 - 1, "phase"] = "recovery"

        # primary buildup (mild)
        p_norm = self.normal[spec.primary_sensor]
        seg = dataframe.loc[b0:f0 - 1, spec.primary_sensor].to_numpy(copy=True)
        seg = self._inject_fault(seg, spec.primary_fault_type, spec.magnitude * 0.5, p_norm)
        dataframe.loc[b0:f0 - 1, spec.primary_sensor] = seg

        # primary failure (stronger; drift -> step_shift for distinctness)
        p_ab = self.abnormal.get(spec.primary_sensor, p_norm)
        failure_type: FaultType = "step_shift" if spec.primary_fault_type in {"drift_up", "drift_down"} else spec.primary_fault_type
        seg = dataframe.loc[f0:r0 - 1, spec.primary_sensor].to_numpy(copy=True)
        seg = self._inject_fault(seg, failure_type, spec.magnitude, p_ab)
        dataframe.loc[f0:r0 - 1, spec.primary_sensor] = seg

        # propagate to paired sensors (lag + strength)
        for link in self.propagation.get(spec.primary_sensor, []):
            sec = link["secondary"]
            if sec not in dataframe.columns:
                continue

            strength = float(link["strength"])
            lag = int(link["lag"])
            start = f0 + lag
            end = min(r0, start + (r0 - f0))
            if start >= end:
                continue

            sec_fault = str(link.get("recommended_secondary_fault", "variance_burst"))
            if sec_fault not in {
                "drift_up", "drift_down", "spike", "stuck_constant", "variance_burst",
                "step_shift", "intermittent_dropout", "sawtooth"
            }:
                sec_fault = "variance_burst"

            sec_prof = self.abnormal.get(sec, self.normal[sec])
            sec_seg = dataframe.loc[start:end - 1, sec].to_numpy(copy=True)
            sec_seg = self._inject_fault(sec_seg, sec_fault, spec.magnitude * strength, sec_prof)
            dataframe.loc[start:end - 1, sec] = sec_seg

        # recovery: pull each sensor toward recovery profile mean with mild noise
        if spec.recovery > 0:
            for sensor in self.sensors:
                rec_prof = self.recovery.get(sensor, self.normal[sensor])
                cur = dataframe.loc[r0:n0 - 1, sensor].to_numpy(copy=True)

                target = float(rec_prof.mean)
                start_val = float(cur[0]) if len(cur) else target
                curve = np.linspace(start_val, target, spec.recovery)

                noise_scale = max(float(rec_prof.std) * 0.25, 1e-6)
                curve = curve + self.rng.normal(0.0, noise_scale, size=spec.recovery)

                curve = np.clip(curve, float(rec_prof.lower_bound), float(rec_prof.upper_bound))
                curve = np.clip(curve, float(rec_prof.min_value), float(rec_prof.max_value))
                dataframe.loc[r0:n0 - 1, sensor] = curve

        # stream_state label
        dataframe["stream_state"] = "normal"
        dataframe.loc[b0:f0 - 1, "stream_state"] = "buildup"
        dataframe.loc[f0:r0 - 1, "stream_state"] = "abnormal"
        dataframe.loc[r0:n0 - 1, "stream_state"] = "recovery"

        # optional: make abnormal window slightly more group-correlated
        group_name = self.sensor_to_group.get(spec.primary_sensor)
        if group_name is not None:
            self._apply_group_driver(dataframe.iloc[f0:r0], group_name, self.abnormal, strength=0.55)

        return dataframe