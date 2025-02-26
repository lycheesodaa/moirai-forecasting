import os
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


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


pred_lens = [1, 3, 7, 14, 30]
results_dir = './'
runs = os.listdir(Path(results_dir))
runs = [run for run in runs if '.' not in run]
runs.sort()

results_mean = {
    f'pl{pred_len}_mean': {} for pred_len in pred_lens
}
results_median = {
    f'pl{pred_len}_median': {} for pred_len in pred_lens
}
results = results_mean | results_median

for run_dir in runs:
    full_run_dir = Path(results_dir) / run_dir
    run_name = run_dir.replace('stocks_', '')

    for pred_len in pred_lens:
        # zero shot
        temp_df = pd.read_csv(full_run_dir / f'MOIRAI_pl{pred_len}_zero_shot.csv')
        temp_df.dropna(inplace=True)

        results[f'pl{pred_len}_mean'][run_name] = calculate_mape(temp_df['true'], temp_df['pred_mean'])
        results[f'pl{pred_len}_median'][run_name] = calculate_mape(temp_df['true'], temp_df['pred_median'])
#
#         # finetuned
#         try:
#             finetuned_name = run_name + '_finetuned'
#             temp_df = pd.read_csv(full_run_dir / f'MOIRAI_pl{pred_len}_finetuned.csv')
#             temp_df.drop(columns=['var', 'q10', 'q50', 'q90'], inplace=True)
#             temp_df.dropna(inplace=True)
#
#             results[f'pl{pred_len}_mean'][finetuned_name] = calculate_mape(temp_df['true'], temp_df['pred_mean'])
#             # results[f'pl{pred_len}_median'][finetuned_name] = calculate_mape(temp_df['true'], temp_df['pred_median'])
#         except Exception as e:
#             print(f'{full_run_dir}/MOIRAI_pl{pred_len}_finetuned.csv not found')
#
#         # lstm
#         lstm_name = run_name + '_lstm'
#         temp_df = pd.read_csv(full_run_dir / f'BiLSTM_pl{pred_len}_combined_training.csv')
#         temp_df.dropna(inplace=True)
#
#         results[f'pl{pred_len}_mean'][lstm_name] = calculate_mape(temp_df['true'], temp_df['pred_mean'])
#
results = pd.DataFrame(results)
#
# lstm_mask = results.index.str.contains('_lstm')
# finetuned_mask = results.index.str.contains('_finetuned')
#
# # Split the dataframe into LSTM and non-LSTM parts
# lstm_rows = results[lstm_mask]
# ft_rows = results[finetuned_mask]
# zs_rows = results[~lstm_mask & ~finetuned_mask]
#
# # Concatenate the parts in desired order
# results = pd.concat([zs_rows, ft_rows, lstm_rows])

results.to_csv('mape_losses.csv')