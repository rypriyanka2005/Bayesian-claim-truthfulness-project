| model | accuracy | balanced_acc | macro_F1 | MCC | AUC | Brier | log_loss |
|---|---|---|---|---|---|---|---|
| Majority class (always False) | 0.646 | 0.5 | 0.392 | 0.0 | 0.5 | 0.229 | 0.65 |
| NB, leaky history (raw counts) | 0.687 | 0.668 | 0.665 | 0.33 | 0.744 | 0.224 | 0.751 |
| NB baseline (corrected history) | 0.64 | 0.619 | 0.616 | 0.234 | 0.671 | 0.252 | 0.795 |
| + Tb-NB threshold | 0.575 | 0.619 | 0.575 | 0.235 | 0.671 | 0.252 | 0.795 |
| + iTb-NB second stage | 0.605 | 0.606 | 0.593 | 0.204 | 0.671 | 0.252 | 0.795 |
| NB, kernel-density numeric likelihoods + threshold | 0.585 | 0.625 | 0.584 | 0.244 | 0.676 | 0.25 | 0.778 |
| NB, quantile-based numeric likelihoods + threshold (Paper 3) | 0.594 | 0.616 | 0.589 | 0.223 | 0.669 | 0.252 | 0.789 |
| + ensemble of evidence groups & feature selection (Paper 1) | 0.609 | 0.621 | 0.601 | 0.232 | 0.67 | 0.211 | 0.608 |
| FULL: kde likelihoods + ensemble + selection + threshold | 0.599 | 0.628 | 0.596 | 0.246 | 0.674 | 0.21 | 0.606 |
| Reference: logistic regression + threshold | 0.527 | 0.592 | 0.526 | 0.192 | 0.65 | 0.22 | 0.63 |