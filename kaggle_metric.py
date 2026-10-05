"""
Custom multilabel ROC AUC metric for a binary multilabel classification problem.

This metric expects two DataFrames:
- submission: contains predicted probabilities for each label
- solution: contains binary ground truth labels for each label

The metric calculates micro-average ROC AUC across all labels.

Required columns:
- submission: same label columns as solution, containing probabilities for class 1
- solution: same label columns as submission, containing binary ground truth labels
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class ParticipantVisibleError(Exception):
    """Raise this for participant-facing errors."""
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "Run No") -> float:
    """
    Compute micro-average ROC AUC for multilabel binary classification.

    The score is evaluated on the rows that exist in both solution and submission,
    which lets you score a subset such as solution_1 or solution_2 against a full submission.

    Args:
        solution (pd.DataFrame): Binary ground truth labels.
        submission (pd.DataFrame): Predicted probabilities for label 1.
        row_id_column_name: Optional ID column name if present in both DataFrames.

    Returns:
        float: micro-average ROC AUC
    """

    if row_id_column_name in solution.columns:
        solution = solution.set_index(row_id_column_name)
    if row_id_column_name in submission.columns:
        submission = submission.set_index(row_id_column_name)

    if solution.empty or submission.empty:
        raise ParticipantVisibleError("Solution and submission must not be empty.")

    missing_ids = solution.index.difference(submission.index)
    if not missing_ids.empty:
        raise ParticipantVisibleError(
            f"Submission is missing {len(missing_ids)} required ID(s), for example: {missing_ids[:5].tolist()}"
        )

    label_columns = [col for col in solution.columns if col in submission.columns]
    if not label_columns:
        raise ParticipantVisibleError("Solution and submission must share label columns.")

    solution = solution[label_columns]
    submission = submission[label_columns]

    common_index = solution.index.intersection(submission.index)
    if common_index.empty:
        raise ParticipantVisibleError("Solution and submission must share at least one row ID.")

    solution = solution.loc[common_index]
    submission = submission.loc[common_index].reindex(solution.index)

    if not all(pd.api.types.is_numeric_dtype(solution[col]) for col in solution.columns):
        raise ParticipantVisibleError("Solution columns must be numeric binary labels.")
    if not all(pd.api.types.is_numeric_dtype(submission[col]) for col in submission.columns):
        raise ParticipantVisibleError("Submission columns must be numeric probabilities.")

    try:
        result = roc_auc_score(solution.to_numpy(), submission.to_numpy(), average="micro")
    except Exception as e:
        raise ParticipantVisibleError(f"Error during ROC AUC calculation: {e}")

    if not np.isfinite(result):
        raise ParticipantVisibleError("Final ROC AUC is not a finite number.")

    return float(result)