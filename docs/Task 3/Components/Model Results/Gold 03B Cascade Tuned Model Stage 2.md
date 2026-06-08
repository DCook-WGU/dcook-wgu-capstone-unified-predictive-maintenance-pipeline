## Gold 03B Cascade Tuned Model: Stage 2 Parameter Search and Early-Warning Evaluation

The Gold 03B notebook expanded the cascade modeling approach by adding parameter and threshold search to Stage 2 of the anomaly-detection pipeline. The purpose of this experiment was to determine whether the fixed cascade from Gold 03A could be improved by tuning the narrow Isolation Forest re-screening stage while keeping the overall cascade structure consistent. In this sequence, Gold 02 served as the single-model baseline, Gold 03A tested a fixed multi-stage cascade, and Gold 03B evaluated whether systematic Stage 2 tuning could improve the balance between alert reduction, recall, precision, and F1 score.

The baseline Isolation Forest achieved perfect recall on the held-out test set, but it produced a very large number of alerts. The baseline generated `31,200` test alerts, with precision of `0.0038`, recall of `1.0000`, and an F1 score of `0.0075`. This made the baseline useful as a high-recall benchmark, but not as an operationally practical alerting model because of its high false-positive burden.

Gold 03A introduced the first cascade model using fixed Stage 1 and Stage 2 thresholds with basic Stage 3 confirmation logic. This default cascade reduced test alerts from `31,200` to `24,894`, showing that a staged architecture could filter part of the baseline alert stream. However, 03A also reduced recall to `0.6525` and produced an F1 score of only `0.0062`, which was lower than the baseline F1. Therefore, 03A demonstrated the value of cascade filtering, but it did not yet improve the overall detection tradeoff.

Gold 03B improved this design by tuning Stage 2 through parameter and threshold search. Stage 1 remained fixed as the broad anomaly detector, while Stage 2 searched for a better narrow Isolation Forest configuration. The selected Stage 2 configuration used a `99.5` threshold percentile, `100` estimators, `max_samples` set to `auto`, `contamination` set to `auto`, `max_features` set to `1.0`, and bootstrap disabled. This configuration was selected from the parameter-search process and became the tuned cascade output. 

### Model Comparison Results

| Model | Test Alerts | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Gold 02 Baseline Isolation Forest | 31,200 | 0.0038 | 1.0000 | 0.0075 |
| Gold 03A Cascade Default | 24,894 | 0.0031 | 0.6525 | 0.0062 |
| Gold 03B Cascade Tuned | 15,153 | 0.0053 | 0.6864 | 0.0106 |

The tuned cascade produced `15,153` test alerts, reducing the baseline alert count by `16,047` alerts. This represents a `51.4%` reduction in test alerts compared with the baseline. It also improved precision from `0.0038` to `0.0053` and improved F1 from `0.0075` to `0.0106`, although recall decreased from `1.0000` to `0.6864`. Compared with the 03A default cascade, 03B improved every major tradeoff metric: it reduced alerts, increased precision, increased recall, and increased F1. 

These results show that Stage 2 tuning provided a measurable improvement over the fixed cascade. The 03A cascade reduced alerts but lost too much detection performance. The 03B tuned cascade found a better balance by reducing alert volume further while improving recall and F1 relative to 03A. It also improved F1 over the original baseline, which is important because the baseline’s perfect recall came at the cost of many false positives.

### Early-Onset Warning Evaluation

Gold 05 was used to evaluate the selected `cascade_tuned` run as an early-warning model. The tuned cascade produced its first final alert at row `5,201`, while the first `BROKEN` row occurred at row `17,155`. Because the project treats one row as one minute, this produced a lead time of `11,954` rows, or approximately `11,954` minutes before the first failure row. This is roughly `199` hours, or about `8.3` days of early-warning lead time. 

| Early-Warning Metric | Value |
|---|---:|
| First final cascade alert row | 5,201 |
| First BROKEN row | 17,155 |
| Lead rows / minutes | 11,954 |
| Approximate lead time | 8.3 days |
| Early-warning rows | 365 |
| Failure-hit rows | 1 |
| Recovery-alert rows | 1,181 |
| Total final alert rows | 54,597 |
| False-positive rows | 53,050 |

The early-warning result is meaningful because the tuned cascade identified abnormal behavior well before the first failure row. However, the result is not yet operationally clean. Although the model produced a long lead time, it also produced `53,050` false-positive rows across the full timeline. This means that the tuned cascade was sensitive to early abnormal behavior, but it still produced too many alerts to be considered a fully refined alerting system.

One important improvement is that the cascade did not simply reproduce the earliest Stage 1 noise. Stage 1 first flagged a row at index `260`, but the final cascade alert did not occur until row `5,201`. This shows that Stage 2 and Stage 3 filtered some of the broad Stage 1 alert stream before producing a final alert. However, the final alert stream still contained too many false positives, indicating that additional confirmation logic is needed. 

### Interpretation

Gold 03B successfully improved the cascade modeling sequence. The tuned cascade reduced alert volume more effectively than the default cascade and achieved a better F1 score than both the baseline and 03A. This demonstrates that Stage 2 parameter search can improve the cascade’s ability to balance detection sensitivity with alert reduction.

At the same time, the 03B model is not the final optimal solution. The model still produced a large number of false-positive rows across the full timeline, even though it provided strong early-warning lead time. This indicates that Stage 2 tuning improved the model, but additional Stage 3 refinement is needed to make the alert stream more operationally useful.

### Capstone Conclusion for Gold 03B

The Gold 03B tuned cascade represents a meaningful improvement over the fixed cascade experiment. By adding Stage 2 parameter and threshold search, the model reduced test alerts from `24,894` in 03A to `15,153`, increased precision, increased recall, and improved F1 score. Compared with the baseline, it reduced test alerts by more than half and improved F1, while still maintaining useful failure detection.

The early-warning evaluation also showed that the tuned cascade produced a substantial lead time before the first failure event, with the first final alert occurring `11,954` minutes before the first `BROKEN` row. However, the model still produced a high false-positive burden across the full dataset. Therefore, 03B demonstrates that Stage 2 tuning improves the cascade, but it also motivates the next experiment: improving Stage 3 confirmation logic in Gold 03C to reduce false positives and produce a cleaner, more operationally credible alert stream.