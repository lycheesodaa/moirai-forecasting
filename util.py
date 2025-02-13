import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

import pandas.errors


def calculate_mape(y_true: list, y_pred: list) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Args:
    y_true: Array of true values
    y_pred: Array of predicted values

    Returns:
    float: The calculated MAPE

    Raises:
    ValueError: If the input arrays have different shapes
    ZeroDivisionError: If any true value is zero
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted arrays must have the same shape")

    if np.any(y_true == 0):
        raise ZeroDivisionError("MAPE is undefined when true values contain zeros")

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def log_into_csv(
    results_df: pd.DataFrame,
    name: str,
    stage: str,
    seq_len: int = 512,
    pred_len: int = 96,
    n_features: int = 0,
    lr: float = None,
    bsz: int = 16,
    log_file_name: str = 'demand',
    pred_col_name: str = 'actual',
):
    log_file = f'results/{log_file_name}_runs.csv'

    # Create sample first line in records
    if not os.path.exists(log_file):
        df = pd.DataFrame({
            'timestamp': datetime.now(),
            'name': 'sample',
            'stage': 'finetuned',
            'model': 'MOIRAI',
            'seq_len': 512,
            'pred_len': 96,
            'n_features': 0,
            'lr': 0.01,
            'bsz': 16,
            'score_type': 'mape',
            'score': 1.23,
        }, index=[0])
        df.to_csv(log_file)

    curr_run = pd.DataFrame({
        'timestamp': datetime.now(),
        'name': name,
        'stage': stage,
        'model': 'MOIRAI',
        'seq_len': seq_len,
        'pred_len': pred_len,
        'n_features': n_features,
        'lr': lr,
        'bsz': bsz,
        'score_type': 'mape',
        'score': calculate_mape(results_df['true'], results_df[pred_col_name])
    }, index=[0])

    df = pd.read_csv(log_file, index_col=0)
    assert len(df.columns) == len(curr_run.columns)

    df = pd.concat([df, curr_run]).reset_index(drop=True)
    df.to_csv(log_file)

# one-time run for existing files not logged
# for file in os.listdir('demand_aus'):
#     if os.path.isdir(f'demand_aus/{file}'):
#         num_features = int(file.removeprefix('top'))
#
#         for csv_file in os.listdir(f'demand_aus/{file}'):
#             filename_ints = list(map(int, re.findall(r'\d+', csv_file)))
#             pl = filename_ints[0]
#             if len(filename_ints) > 1:
#                 pl = filename_ints[1]
#
#             if 'finetuned' in csv_file:
#                 stage = 'finetuned'
#             else:
#                 stage = 'zeroshot'
#
#             try:
#                 log_into_csv(
#                     pd.read_csv(f'demand_aus/{file}/{csv_file}'),
#                     name=f'aus_{file}',
#                     stage=stage,
#                     pred_len=pl,
#                     pred_filter_len=None,
#                     pred_col_name='pred_mean'
#                 )
#             except pandas.errors.EmptyDataError:
#                 print(f'{csv_file} is empty. Skipping...')
#                 continue