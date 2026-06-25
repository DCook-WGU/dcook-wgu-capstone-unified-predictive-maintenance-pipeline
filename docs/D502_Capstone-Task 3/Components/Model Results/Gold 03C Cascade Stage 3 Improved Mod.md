## Gold 03C Cascade Stage 3 Improved Model: Tuned Confirmation Layer

The Gold 03C notebook represents the final cumulative cascade experiment in the Gold modeling sequence. Gold 02 established the single-model Isolation Forest baseline, Gold 03A introduced a fixed three-stage cascade, and Gold 03B added dynamic Stage 2 parameter and threshold search. Gold 03C builds on that progression by retaining the dynamic Stage 2 search from 03B and adding an improved Stage 3 confirmation layer.

The goal of 03C was not simply to maximize recall. The baseline model had already shown that very high recall could be achieved by producing a large number of alerts. Instead, the purpose of 03C was to improve alert quality by reducing false positives, improving precision, improving F1 score, and maintaining useful recall. This makes 03C the most operationally focused version of the cascade design.

### 03C Design

Gold 03C used the same broad Stage 1 Isolation Forest structure as the earlier cascade models. Stage 1 remained fixed as the initial anomaly-candidate detector. Stage 2 retained the dynamic parameter-search functionality introduced in 03B. The selected Stage 2 configuration used a 99.5 threshold percentile with 100 estimators, max_samples set to auto, contamination set to auto, max_features set to 1.0, and bootstrap disabled.

The new contribution in 03C was the improved Stage 3 confirmation layer. Stage 3 evaluated candidate alerts using rule-based evidence from primary sensor breaches, secondary sensor corroboration, persistence across a rolling window, and drift behavior. The selected Stage 3 configuration used a minimum weighted score of 3.25, a rolling window size of 3, a requirement of 3 flags within the window, 3 strong primary sensor hits, and a drift threshold multiplier of 1.5.

### Model Comparison Results
| Model | Test Alerts | Precision | Recall | F1 Score | 
| ----- | ----------- | --------- | ------ | -------- | 

| Gold 02 Baseline Isolation Forest | 31,200 | 0.0038 | 1.0000 | 0.0075 | 
| Gold 03A Cascade Default | 24,895 | 0.0031 | 0.6525 | 0.0062 | 
| Gold 03B Cascade Tuned | 15,153 | 0.0053 | 0.6864 | 0.0106 | 
| Gold 03C Stage 3 Improved | 6,594 | 0.0106 | 0.5932 | 0.0209 | 

The 03C Stage 3 improved model produced 6,594 test alerts, reducing the alert count substantially compared with all prior main models. Compared with the baseline, this was a reduction of 24,606 test alerts, or approximately 78.9% fewer alerts. Compared with the 03B tuned cascade, 03C reduced test alerts from 15,153 to 6,594, a reduction of 8,559 alerts.

The 03C model also improved precision and F1 score. Precision increased from 0.0053 in 03B to 0.0106 in 03C, meaning the final model produced a higher proportion of useful alerts. F1 increased from 0.0106 in 03B to 0.0209 in 03C, showing that the improved confirmation layer created a better balance between alert quality and detection coverage. Recall decreased from 0.6864 in 03B to 0.5932 in 03C, but it remained above the minimum target while producing a much cleaner alert stream.

### Plain-English Interpretation

The baseline model caught all known anomaly rows, but it produced too many alerts to be useful as a practical alerting system. Gold 03A showed that a basic cascade could reduce alerts, but the fixed settings also reduced performance. Gold 03B improved the cascade by tuning Stage 2, cutting alert volume by more than half compared with the baseline and improving F1.

Gold 03C made the largest practical improvement. Instead of allowing Stage 2 alerts to pass through as final alerts, 03C required additional Stage 3 evidence before confirming an alert. This made the model more selective. The result was a much smaller and more focused alert stream.

In non-technical terms, the model moved from broad detection toward operational alerting. The baseline was highly sensitive but noisy. The 03B tuned cascade was better balanced. The 03C improved cascade was more selective and produced the cleanest main-model alert stream.

### Stage 3 Operating Mode Comparison

Gold 03C also produced relaxed, medium, and strict Stage 3 operating-mode variants. These variants show how changing the confirmation threshold affects alert burden and detection performance.

| Stage 3 Mode | Test Alerts | Precision | Recall | F1 Score |
| -------------| ----------- | --------- | ------ | -------- |
| Stage 3 Improved, selected final model | 6,594 | 0.0106 | 0.5932 | 0.0209 |
| Stage 3 Relaxed | 13,713 | 0.0058 | 0.6780 | 0.0116 |
| Stage 3 Medium | 7,286 | 0.0107 | 0.6610 | 0.0211 |
| Stage 3 Strict | 61 | 0.4426 | 0.2288 | 0.3017 |

The strict mode had the highest precision and F1 score, but it only produced 61 test alerts and had much lower recall. This makes it useful as a very conservative operating mode, but not necessarily the best general-purpose predictive-maintenance setting. The medium mode had a slightly higher F1 score than the selected 03C final model, but it produced more alerts. The selected Stage 3 improved model provided a leaner alert profile while maintaining similar overall balance.

### Capstone Conclusion for Gold 03C

Gold 03C successfully improved the cascade model by adding a tuned Stage 3 confirmation layer. The model reduced test alerts from 31,200 in the baseline to 6,594, while improving precision and F1 score. Compared with 03B, the 03C model produced fewer alerts, higher precision, and a stronger F1 score, while preserving useful recall.

This result supports the main capstone argument: predictive-maintenance modeling is not only about detecting anomalies, but also about producing alerts that are practical to review. The baseline demonstrated that anomaly signal existed in the data. The 03A and 03B cascade experiments showed that staged filtering could improve the alert stream. The 03C model completed that progression by using improved confirmation logic to reduce false positives and create a cleaner, more operationally credible alerting system.