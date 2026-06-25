## Gold 03A Cascade Default Model: Fixed Multi-Stage Anomaly Detection

The Gold 03A notebook introduced the first multi-stage cascade model as an extension of the Gold 02 baseline Isolation Forest. The baseline model used a single broad anomaly detector, while the 03A cascade applied three sequential stages: a broad Stage 1 Isolation Forest, a narrower Stage 2 Isolation Forest, and a basic Stage 3 rule-based confirmation layer. The purpose of this experiment was to determine whether a fixed, hand-configured cascade could reduce the alert burden created by the baseline model.

The Gold 02 baseline Isolation Forest established the initial benchmark. It detected all held-out anomaly rows, achieving a recall of `1.0000`, but it also produced `31,200` test alerts. Its precision was only `0.0038`, and its F1 score was `0.0075`. This made the baseline useful as a high-recall detector, but not operationally practical by itself because it generated too many alerts.

Gold 03A tested whether a staged architecture could narrow that alert stream. Stage 1 used a fixed broad Isolation Forest threshold to identify candidate anomalies. Stage 2 then applied a narrower Isolation Forest using a smaller feature set, and Stage 3 applied basic confirmation logic using rule-based sensor checks. The selected 03A configuration used a Stage 1 threshold percentile of `95.0` and a Stage 2 fixed threshold percentile of `99.0`. Stage 2 used `300` estimators, `max_samples` set to `auto`, `contamination` set to `auto`, `max_features` set to `1.0`, and bootstrap disabled.

### Model Comparison Results

| Model | Test Alerts | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Gold 02 Baseline Isolation Forest | 31,200 | 0.0038 | 1.0000 | 0.0075 |
| Gold 03A Cascade Default | 24,895 | 0.0031 | 0.6525 | 0.0062 |

The default cascade reduced the number of test alerts from `31,200` to `24,895`. This was a reduction of `6,305` alerts, or approximately `20.2%` fewer test alerts than the baseline. This result showed that the cascade architecture could filter out a portion of the baseline alert stream.

However, the reduction in alerts came with a tradeoff. Recall decreased from `1.0000` to `0.6525`, meaning the fixed cascade missed more anomaly rows than the baseline. F1 also decreased from `0.0075` to `0.0062`, and precision remained very low at `0.0031`. Because of this, the 03A cascade should not be interpreted as a full performance improvement over the baseline. Its improvement was primarily in reducing alert volume, not in improving overall detection quality.

### Early-Onset Warning Evaluation

The 03A cascade still produced warnings before the first `BROKEN` row, but it did not improve early-onset behavior compared with the baseline. Its main contribution was filtering some alerts, not producing an earlier or cleaner warning signal. The fixed cascade reduced the total alert burden, but the lower recall and lower F1 score showed that the fixed thresholds were too restrictive in some places and not selective enough in others.

This distinction is important for the capstone interpretation. The 03A model demonstrated that a cascade structure can reduce alerts, but it did not yet show that the cascade was a better anomaly detector overall. The model narrowed the alert stream, but it also removed some useful detections. Therefore, 03A served as an initial cascade benchmark rather than the final cascade solution.

### Interpretation

Gold 03A should be viewed as a fixed-threshold cascade experiment. It answered the question: can a manually configured multi-stage model reduce the baseline alert stream? The answer was yes. The cascade reduced test alerts by about `20.2%`. However, it did not improve recall, F1 score, or early-warning quality. This means that the default cascade improved alert filtering, but not overall model performance.

This result was still valuable because it revealed the core tradeoff of the cascade design. Adding a second screening stage can reduce false-positive burden, but fixed thresholds may also remove true anomaly detections. This created a clear need for the next experiment. Gold 03B introduced Stage 2 parameter and threshold search to determine whether the narrow re-screening stage could be tuned more effectively.

### Capstone Conclusion for Gold 03A

The Gold 03A default cascade reduced the alert burden produced by the baseline Isolation Forest, lowering test alerts from `31,200` to `24,895`. This demonstrated that a multi-stage cascade can filter the broad baseline alert stream. However, the fixed cascade also reduced recall and did not improve F1 score or early-onset warning behavior. Therefore, 03A should be interpreted as an initial cascade benchmark rather than a final improvement over the baseline.

The main value of 03A was that it showed the staged architecture could reduce alerts, while also exposing the limitations of fixed Stage 2 thresholds and basic Stage 3 rules. This directly motivated Gold 03B, where Stage 2 parameter and threshold search was used to improve the tradeoff between alert reduction, recall, precision, and F1 score.