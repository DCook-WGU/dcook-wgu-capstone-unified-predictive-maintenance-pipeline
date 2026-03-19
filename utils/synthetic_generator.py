from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from utils.synthetic_profiles import SensorRichProfile
from utils.synthetic_missingness import MissingnessSpec, build_present_counts_for_block, apply_exact_missingness_mask


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
        Calibrate per-phase (phase column) if present.
        Uses:
        normal  -> normal targets
        buildup -> buildup targets if provided else abnormal targets (recommended)
        abnormal-> abnormal targets
        recovery-> recovery targets
        """
        if self.state_calibration_targets is None:
            return
        if "phase" not in df.columns:
            return

        phase_map = {
            "normal": "normal",
            "buildup": "buildup" if "buildup" in self.state_calibration_targets else "abnormal",
            "abnormal": "abnormal",
            "recovery": "recovery",
        }

        for phase_val, state_name in phase_map.items():
            mask = df["phase"].astype(str).eq(str(phase_val))
            idx = df.index[mask].to_numpy()
            if idx.size == 0:
                continue

            # choose profiles to clip against
            if state_name == "normal":
                profs = self.normal
            elif state_name == "recovery":
                profs = self.recovery
            else:
                profs = self.abnormal

            self._calibrate_block_mean_std(df, idx=idx, profiles=profs, state_name=state_name)
    # -------------------------
    # Abnormal episode (buildup + failure + recovery)
    # -------------------------
    def generate_episode(self, spec: EpisodeSpec) -> pd.DataFrame:
        if spec.primary_sensor not in self.sensors:
            raise ValueError(f"Unknown primary sensor: {spec.primary_sensor}")

        # Define whether this episode is a "fault episode"
        # (dataset-faithful: failure row is the event marker)
        is_fault_episode = int(spec.failure) > 0

        # Normalize ints defensively
        normal_before = int(max(0, spec.normal_before))
        buildup = int(max(0, spec.buildup)) if is_fault_episode else 0
        failure = int(max(0, spec.failure)) if is_fault_episode else 0
        recovery = int(max(0, spec.recovery)) if is_fault_episode else 0
        normal_after = int(max(0, spec.normal_after))

        total = normal_before + buildup + failure + recovery + normal_after
        if total <= 0:
            # safety: never generate an empty frame
            total = 1
            normal_before = 1

        dataframe = self.generate_normal_batch(total)

        # phase boundaries
        b0 = normal_before
        f0 = b0 + buildup
        r0 = f0 + failure
        n0 = r0 + recovery

        # default labels
        dataframe["phase"] = "normal"
        dataframe["stream_state"] = "normal"

        # Only label fault phases if this is a fault episode
        if is_fault_episode:
            if buildup > 0:
                dataframe.loc[b0:f0 - 1, "phase"] = "buildup"
                dataframe.loc[b0:f0 - 1, "stream_state"] = "buildup"

            if failure > 0:
                dataframe.loc[f0:r0 - 1, "phase"] = "abnormal"
                dataframe.loc[f0:r0 - 1, "stream_state"] = "abnormal"

            if recovery > 0:
                dataframe.loc[r0:n0 - 1, "phase"] = "recovery"
                dataframe.loc[r0:n0 - 1, "stream_state"] = "recovery"

        # -------------------------
        # Primary buildup (ramped intensity) — ONLY on fault episodes
        # -------------------------
        p_norm = self.normal[spec.primary_sensor]

        if is_fault_episode and buildup > 0:
            seg = dataframe.loc[b0:f0 - 1, spec.primary_sensor].to_numpy(copy=True)
            buildup_len = int(seg.shape[0])

            if buildup_len > 0:
                cut1 = max(1, int(round(buildup_len * 0.33)))
                cut2 = max(cut1 + 1, int(round(buildup_len * 0.66))) if buildup_len >= 3 else buildup_len

                early = seg[:cut1]
                mid = seg[cut1:cut2]
                late = seg[cut2:]

                if early.size:
                    early = self._inject_fault(early, spec.primary_fault_type, spec.magnitude * 0.2, p_norm)
                if mid.size:
                    mid = self._inject_fault(mid, spec.primary_fault_type, spec.magnitude * 0.5, p_norm)
                if late.size:
                    late = self._inject_fault(late, spec.primary_fault_type, spec.magnitude * 0.8, p_norm)

                seg2 = np.concatenate([early, mid, late]) if buildup_len > 1 else early
                dataframe.loc[b0:f0 - 1, spec.primary_sensor] = seg2

        # -------------------------
        # Primary failure marker — ONLY when failure slice exists
        # -------------------------
        if is_fault_episode and failure > 0:
            p_ab = self.abnormal.get(spec.primary_sensor, p_norm)
            failure_type: FaultType = (
                "step_shift"
                if spec.primary_fault_type in {"drift_up", "drift_down"}
                else spec.primary_fault_type
            )

            seg = dataframe.loc[f0:r0 - 1, spec.primary_sensor].to_numpy(copy=True)
            if seg.size:  # critical guard
                seg = self._inject_fault(seg, failure_type, spec.magnitude, p_ab)
                dataframe.loc[f0:r0 - 1, spec.primary_sensor] = seg

        # -------------------------
        # Propagate to paired sensors — ONLY on fault episodes
        # Span buildup + failure (not recovery), so pairings still matter with 1-row failure.
        # -------------------------
        if is_fault_episode and (buildup + failure) > 0:
            prop_start_base = b0
            prop_end_base = r0  # right before recovery begins

            for link in self.propagation.get(spec.primary_sensor, []):
                sec = link["secondary"]
                if sec not in dataframe.columns:
                    continue

                strength = float(link["strength"])
                lag = int(link["lag"])

                start = prop_start_base + lag
                end = prop_end_base
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
                if sec_seg.size:  # critical guard
                    sec_seg = self._inject_fault(sec_seg, sec_fault, spec.magnitude * strength, sec_prof)
                    dataframe.loc[start:end - 1, sec] = sec_seg

        # -------------------------
        # Recovery — ONLY when recovery slice exists
        # -------------------------
        if is_fault_episode and recovery > 0:
            for sensor in self.sensors:
                rec_prof = self.recovery.get(sensor, self.normal[sensor])
                cur = dataframe.loc[r0:n0 - 1, sensor].to_numpy(copy=True)

                target = float(rec_prof.mean)
                start_val = float(cur[0]) if len(cur) else target
                curve = np.linspace(start_val, target, recovery)

                noise_scale = max(float(rec_prof.std) * 0.25, 1e-6)
                curve = curve + self.rng.normal(0.0, noise_scale, size=recovery)

                curve = np.clip(curve, float(rec_prof.lower_bound), float(rec_prof.upper_bound))
                curve = np.clip(curve, float(rec_prof.min_value), float(rec_prof.max_value))
                dataframe.loc[r0:n0 - 1, sensor] = curve

        # -------------------------
        # Optional: group correlation boost (abnormal window only)
        # -------------------------
        if is_fault_episode and failure > 0:
            group_name = self.sensor_to_group.get(spec.primary_sensor)
            if group_name is not None and r0 > f0:
                self._apply_group_driver(dataframe.iloc[f0:r0], group_name, self.abnormal, strength=0.55)

        # optional: calibrate per-phase mean/std toward empirical targets
        self._calibrate_by_phase(dataframe)

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
        Apply exact missingness replication.

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

                out = apply_exact_missingness_mask(
                    out,
                    sensor_cols=features,
                    rng=rng,
                    present_counts=present_counts,
                    eligible_row_idx=eligible_idx,
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

                out = apply_exact_missingness_mask(
                    out,
                    sensor_cols=features,
                    rng=rng,
                    present_counts=present_counts,
                    eligible_row_idx=eligible_idx,
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
        out = apply_exact_missingness_mask(
            out,
            sensor_cols=features,
            rng=rng,
            present_counts=present_counts,
            eligible_row_idx=out.index.to_numpy(),
        )
        return out
    

