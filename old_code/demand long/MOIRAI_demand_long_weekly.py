from pathlib import Path

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
import seaborn as sns
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import sys
import os

from triton.language import dtype

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from util import log_into_csv, create_timestamps

output_dir = Path('results/long/demand_long_weekly')
output_dir.mkdir(parents=True, exist_ok=True)

PDT_LIST = [336]  # hourly demand

SIZE = "small"  # Using moirai_2.0_R_small model
CTX = 512  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 16  # batch size: any positive integer

# data formatting into GluonTS dataset
df = pd.read_csv('data/demand_full_weekly.csv', index_col=0, parse_dates=True)
if len(df) > 1:
    df = df.iloc[:, 0:1]
target = df.columns[0]

# evaluation
for PDT in PDT_LIST:
    print(f"Evaluating for PDT={PDT}...")
    # Prepare pre-trained model by downloading model weights from huggingface hub
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=32,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    # Reformat data for model processing
    model_input = df.iloc[-CTX:].values.reshape(1, CTX, 1)
    model_input = torch.Tensor(model_input)
    past_observed_target = torch.ones_like(model_input, dtype=torch.bool)
    past_is_pad = torch.zeros_like(model_input, dtype=torch.bool).reshape(1, -1)

    output = model(torch.Tensor(model_input), past_observed_target, past_is_pad)
    samples = output.reshape(PDT, -1).detach().cpu().numpy()
    mean_samples = np.mean(samples, axis=1)
    median_samples = np.median(samples, axis=1)
    p10 = np.percentile(samples, 10, axis=1)
    p90 = np.percentile(samples, 90, axis=1)
    output_df = pd.DataFrame({
        'mean': mean_samples,
        'median': median_samples,
        'p10': p10,
        'p90': p90,
    })

    output_df['datetime'] = create_timestamps(last_timestamp=df.index[-1], freq='W', periods=PDT)
    output_df.to_csv(output_dir / f'MOIRAI_ctx{CTX}_pl{PDT}_zeroshot.csv')

    samples_df = pd.DataFrame(samples)
    samples_df['datetime'] = create_timestamps(last_timestamp=df.index[-1], freq='W', periods=PDT)
    df_long = pd.melt(samples_df,
                      id_vars=['datetime'],
                      value_vars=list(range(100)),
                      var_name='sample',
                      value_name='system_demand_actual')

    # plot and export figure
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(df, x='datetime', y='system_demand_actual', label='true')
    sns.lineplot(df_long, x='datetime', y='system_demand_actual', label='pred')

    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=10, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.xlabel('Date')
    plt.ylabel('Demand (KwH)')
    plt.title('5-year ahead weekly demand forecasts (MOIRAI)')
    plt.legend()
    plt.grid()
    plt.show()
    plt.tight_layout()
    fig.savefig(output_dir / 'moirai_5year.png', bbox_inches='tight')