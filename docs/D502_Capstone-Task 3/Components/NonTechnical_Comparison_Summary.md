For a non-technical audience, we can translate the metrics into alert burden, true catches, missed anomaly rows, and how many alerts someone would need to review to find one true anomaly row.

Plain-English model comparison
Model	Test Alerts	True Anomaly Rows Caught	Missed Anomaly Rows	False-Positive Alert Rows	True Alerts per 100 Alerts	Approx. Alerts Reviewed per True Catch	Alert Reduction vs Baseline	F1 as “Balance Score”
02 Baseline	31,200	118 / 118	0	31,082	0.38 per 100	1 true catch per ~264 alerts	—	0.75%
03A Cascade Default	24,894	77 / 118	41	24,817	0.31 per 100	1 true catch per ~323 alerts	20.2% fewer alerts	0.62%
03B Cascade Tuned	15,153	81 / 118	37	15,072	0.53 per 100	1 true catch per ~187 alerts	51.4% fewer alerts	1.06%
03C Stage 3 Improved	6,594	70 / 118	48	6,524	1.06 per 100	1 true catch per ~94 alerts	78.9% fewer alerts	2.09%

The 03C final model produced 6,594 test alerts, precision 0.0106, recall 0.5932, and F1 0.0209; the comparison artifacts show the earlier baseline, 03A, and 03B values used in the table.

Easier interpretation

02 Baseline:
The baseline caught every known anomaly row, but it produced a very large alert stream. In practical terms, a reviewer would need to look through about 264 alerts to find one true anomaly row. This makes it useful as a high-recall benchmark, but not as a practical alerting model.

03A Cascade Default:
The fixed cascade reduced the alert count by about 20%, but it also missed more anomaly rows and had a worse true-alert rate than the baseline. This showed that the cascade structure could filter alerts, but fixed settings alone were not enough.

03B Cascade Tuned:
The tuned Stage 2 cascade was a clear improvement. It cut the baseline alert count by more than half and improved the review burden from about 1 true catch per 264 alerts to about 1 true catch per 187 alerts. It also improved the balance score over both the baseline and 03A.

03C Stage 3 Improved:
The Stage 3 improved model created the cleanest practical alert stream so far. It reduced the baseline alert count by almost 79% and improved the review burden to about 1 true catch per 94 alerts. It caught fewer anomaly rows than the baseline, but it produced a much more usable alert stream and nearly doubled the F1 balance score compared with 03B.

Capstone-friendly wording

The baseline model was highly sensitive and detected all held-out anomaly rows, but it produced too many alerts to be operationally useful. The 03A default cascade reduced alert volume, but the fixed thresholds also removed too many useful detections. The 03B tuned cascade improved the tradeoff by reducing alerts by more than half while improving precision and F1. The 03C Stage 3 improved model produced the cleanest alert stream, reducing baseline alerts by almost 79% and improving the review burden from roughly one true anomaly row per 264 alerts to roughly one per 94 alerts. This shows that the final cascade moved the model from broad anomaly detection toward a more practical predictive-maintenance alerting system.