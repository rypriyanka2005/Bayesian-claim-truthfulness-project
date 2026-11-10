| model | accuracy | balanced_acc | macro_F1 | MCC | AUC | Brier | log_loss |
|---|---|---|---|---|---|---|---|
| Majority class (always False) | 0.673 | 0.5 | 0.402 | 0.0 | 0.5 | 0.221 | 0.634 |
| NB, leaky history (raw counts) | 0.704 | 0.673 | 0.67 | 0.34 | 0.742 | 0.22 | 0.74 |
| NB baseline (corrected history) | 0.655 | 0.617 | 0.615 | 0.23 | 0.67 | 0.249 | 0.798 |
| + Tb-NB threshold | 0.582 | 0.616 | 0.577 | 0.219 | 0.67 | 0.249 | 0.798 |
| + iTb-NB second stage | 0.631 | 0.625 | 0.61 | 0.236 | 0.67 | 0.249 | 0.798 |
| NB, kernel-density numeric likelihoods + threshold | 0.584 | 0.614 | 0.577 | 0.214 | 0.672 | 0.249 | 0.795 |
| NB, quantile-based numeric likelihoods + threshold (Paper 3) | 0.606 | 0.616 | 0.593 | 0.218 | 0.673 | 0.246 | 0.784 |
| + ensemble of evidence groups & feature selection (Paper 1) | 0.617 | 0.618 | 0.599 | 0.223 | 0.669 | 0.204 | 0.593 |
| FULL: kde likelihoods + ensemble + selection + threshold | 0.594 | 0.613 | 0.584 | 0.211 | 0.67 | 0.204 | 0.593 |