import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Both y_true and y_pred must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}."
        )

    if y_true.size == 0:
        raise ValueError("Cannot compute accuracy score for empty arrays.")

    return float(np.mean(y_true == y_pred))