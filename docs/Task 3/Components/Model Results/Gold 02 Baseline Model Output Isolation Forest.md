# Gold Baseline Model Output: Isolation Forest

The baseline model used an unsupervised Isolation Forest to establish an initial anomaly-detection benchmark before applying the multi-stage cascade model. The model was trained on the Gold-layer normal-only fit dataset and scored against the full scaled Gold dataset. The baseline used 50 Stage 1 sensor features, 200 estimators, and a 97th-percentile anomaly-score threshold. The final baseline threshold was 0.5040984527. 

## Baseline Configuration
| Item | Value |
| ---- | ----- |
| Model | Isolation Forest |    
| Feature count | 50 |    
| Estimators | 200 |    
| Threshold percentile | 97.0 |    
| Threshold value | 0.5040984527 |    
| Full scored rows | 220,320 |    
| Training/fit rows | 77,395 |    
| Test rows | 83,889 |    
| Test anomaly rows | 118 |

The row-tracking validation confirmed that meta__row_id remained unique across the full baseline modeling dataframe. This allowed the model output to be merged back to the Gold dataset without losing row identity. 

## Baseline Results
| Metric | Value |
| ------ | ----- |
| Alert count, all rows | 87,735 |
| Alert count, test rows | 31,200 |
| True positives | 118 |
| False positives | 31,082 |
| True negatives | 52,689 |
| False negatives | 0 |
| Precision | 0.0038 |
| Recall | 1.0000 |
| F1 score | 0.0075 |
| ROC-AUC | 0.9407 |
| PR-AUC | 0.1220 |

The baseline model detected all held-out anomaly rows, producing a recall of 1.0000 with no false negatives. However, it also generated 31,082 false-positive alerts in the test set, resulting in very low precision of 0.0038. 

## Baseline Interpretation
The baseline Isolation Forest was effective as a broad anomaly detector because it successfully captured all known anomaly rows in the held-out test data. This is important because it shows that the selected Gold-layer features contain meaningful anomaly signal. The ROC-AUC of 0.9407 also indicates that the baseline anomaly scores have strong ranking ability, even though the selected threshold produces too many operational alerts. 
At the same time, the baseline is not operationally sufficient by itself. The model produced 31,200 test alerts for only 118 true anomaly rows, which means most baseline alerts were false positives. This confirms the need for the later cascade stages: the baseline can identify broad abnormal behavior, but additional filtering and confirmation logic are needed to reduce alert burden and improve practical usefulness.

## Baseline Validation
The baseline output was checked after row-tracked scoring to ensure the anomaly-score direction was preserved. The initial baseline scoring step and the synchronized row-tracked output both produced the same test alert count, precision, recall, and F1 score. This confirms that the baseline results were not altered by the row-tracking merge process. 

## Capstone Conclusion for Baseline
The baseline Isolation Forest establishes a useful high-recall benchmark for the predictive-maintenance pipeline. It confirms that unsupervised anomaly detection can identify the held-out failure behavior, but it also demonstrates why a single broad detector is insufficient for deployment. The baseline’s perfect recall is valuable, but its low precision and high false-positive count justify the design of the multi-stage cascade model, where later stages are expected to narrow the alert stream and produce more operationally credible warnings.