from typing import List, Dict, Union, Tuple, Any
import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from code.utils import CHART_TYPE_TOKENS, X_START, X_END, Y_START, Y_END

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function for the given value.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the normalized Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The normalized Root Mean Square Error.
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0
        return 0.0

    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate the normalized Levenshtein distance between two lists of strings.

    Args:
        y_true (List[str]): The true values.
        y_pred (List[str]): The predicted values.

    Returns:
        float: The normalized Levenshtein distance.
    """
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(
    y_true: List[Union[float, str]], y_pred: List[Union[float, str]]
) -> float:
    """
    Calculate the score for a series of true and predicted values.

    Args:
        y_true (List[Union[float, str]]): The true values.
        y_pred (List[Union[float, str]]): The predicted values.

    Returns:
        float: The score for the series.
    """
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        print(ground_truth.columns, predictions.columns)
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
            if gt_type != pred_type:  # Check chart_type condition
                if (
                    (gt_type == "histogram" and pred_type == "vertical_bar") or 
                    (gt_type == "vertical_bar" and pred_type == "histogram")
                ):
                    scores.append(score_series(gt_series, pred_series))
                else:
                    scores.append(0.0)
            else:  # Score with RMSE or Levenshtein as appropriate
                scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score


def string2triplet(pred_string: str) -> Tuple[str, List[str], List[str]]:
    """
    Convert a prediction string to a triplet of chart type, x values, and y values.

    Args:
        pred_string (str): The prediction string.

    Returns:
        Tuple[str, List[str], List[str]]: A triplet of chart type, x values, and y values.
    """
    
    chart_type = "line"
    for tok in CHART_TYPE_TOKENS:
        if tok in pred_string:
            chart_type = tok.strip("<>")

    x = pred_string.split(X_START)[1].split(X_END)[0].strip().split(";")
    y = pred_string.split(Y_START)[1].split(Y_END)[0].strip().split(";")

    if chart_type == 'line':
        y = y[:len(x)]
        while len(y) < len(x):
            y.append(y[-1])
    """
    if chart_type == 'histogram':
        y = y[:len(x)-1]
        while len(y) < len(x)-1:
            y.append(y[-1])
    """

    if len(x) == 0:
        x = []
    if len(y) == 0:
        y = []

    return chart_type, x, y


def validation_metrics(val_outputs: List[str], val_ids: List[str], gt_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate validation metrics for a set of outputs, ids, and ground truth dataframe.

    Args:
        val_outputs (List[str]): A list of validation outputs.
        val_ids (List[str]): A list of validation ids.
        gt_df (pd.DataFrame): The ground truth dataframe.

    Returns:
        Dict[str, float]: A dictionary containing the validation scores.
    """
    pred_triplets = []

    for example_output in val_outputs:

        if not all([x in example_output for x in [X_START, X_END, Y_START, Y_END]]):
            pred_triplets.append(("line", [], []))
        else:
            pred_triplets.append(string2triplet(example_output))

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )

    overall_score, chart_type2score = benetech_score(
        gt_df.loc[pred_df.index.values], pred_df
    )

    return {
        "val_score": overall_score,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }

def non_mean_benetech_score(ground_truth, predictions):

    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        print(ground_truth.columns, predictions.columns)
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores, charts = [], []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
            charts.append(gt_type)
            if gt_type != pred_type:  # Check chart_type condition
                if (
                    (gt_type == "histogram" and pred_type == "vertical_bar") or 
                    (gt_type == "vertical_bar" and pred_type == "histogram")
                ):
                    scores.append(score_series(gt_series, pred_series))
                else:
                    scores.append(0.0)
            else:  # Score with RMSE or Levenshtein as appropriate
                scores.append(score_series(gt_series, pred_series))

    return scores, charts

def val_score(val_outputs, val_ids, gt_df):

    pred_triplets = []

    for example_output in val_outputs:

        if not all([x in example_output for x in [X_START, X_END, Y_START, Y_END]]):
            pred_triplets.append(("line", [], []))
        else:
            pred_triplets.append(string2triplet(example_output))

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )
    
    return non_mean_benetech_score(gt_df.loc[pred_df.index.values], pred_df)

from torchmetrics import Metric
import torch

#['line', 'vertical_bar', 'horizontal_bar', 'scatter', 'dot', 'histogram']

from code.data import is_nan

class CustomMetric(Metric):
    def __init__(self, gt_df, metric='score'):
        super().__init__()

        self.gt_df = gt_df
        self.metric = metric

        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, ids):
        
        scores, charts = val_score(preds, ids, self.gt_df)

        if self.metric == 'score':
            self.score += sum(scores)
            self.count += len(scores)
        
        else:

            for score, chart in zip(scores, charts):
                if self.metric == chart:
                    self.score += score
                    self.count += 1

    def compute(self):
        return self.score.float() / max(1, self.count)