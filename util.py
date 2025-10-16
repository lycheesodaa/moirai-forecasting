from typing import Union, Optional, List

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import re

import pandas.errors



def create_timestamps(
    last_timestamp: Union[datetime, pd.Timestamp],
    freq: Optional[Union[int, float, timedelta, pd.Timedelta, str]] = None,
    time_sequence: Optional[Union[List[int], List[float], List[datetime], List[pd.Timestamp]]] = None,
    periods: int = 1,
) -> List[pd.Timestamp]:
    """Simple utility to create a list of timestamps based on start, delta and number of periods

    Args:
        last_timestamp (Union[datetime.datetime, pd.Timestamp]): The last observed timestamp, new timestamps will be created
            after this timestamp.
        freq (Optional[Union[int, float, datetime.timedelta, pd.Timedelta, str]], optional): The frequency at which timestamps
            should be generated. Defaults to None.
        time_sequence (Optional[Union[List[int], List[float], List[datetime.datetime], List[pd.Timestamp]]], optional): A time sequence
            from which the frequency can be inferred. Defaults to None.
        periods (int, optional): The number of timestamps to generate. Defaults to 1.

    Raises:
        ValueError: If the frequency cannot be parsed from freq or inferred from time_sequence

    Returns:
        List[pd.Timestamp]: List of timestamps
    """

    if freq is None and time_sequence is None:
        raise ValueError("Neither `freq` nor `time_sequence` provided, cannot determine frequency.")

    if freq is None:
        # to do: make more robust
        freq = time_sequence[-1] - time_sequence[-2]

    # more complex logic is required to support all edge cases
    if isinstance(freq, (pd.Timedelta, timedelta, str)):
        try:
            # try date range directly
            return pd.date_range(
                last_timestamp,
                freq=freq,
                periods=periods + 1,
            ).tolist()[1:]
        except ValueError as e:
            # if it fails, we can try to compute a timedelta from the provided string
            if isinstance(freq, str):
                freq = pd._libs.tslibs.timedeltas.Timedelta(freq)
                return pd.date_range(
                    last_timestamp,
                    freq=freq,
                    periods=periods + 1,
                ).tolist()[1:]
            else:
                raise e
    else:
        # numerical timestamp column
        return [last_timestamp + i * freq for i in range(1, periods + 1)]


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
    function: str = 'mape',
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

    if function == '80% range':
        score = np.mean(results_df['p90'] - results_df['p10'])
    elif function == 'mape':
        score = calculate_mape(results_df['true'], results_df[pred_col_name])

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
        'score_type': function,
        'score': score
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