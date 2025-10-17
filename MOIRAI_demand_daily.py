from pathlib import Path
from types import NoneType

import torch
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import sys
import os

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
from util import log_into_csv

parser = argparse.ArgumentParser(description='MOIRAI')
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--run_name', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--yaml_prefix', type=str)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--finetuned', type=bool, default=False)
parser.add_argument("--moirai_ver", type=str, default=None)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

PDT_LIST = [1, 7, 14, 30, 60, 180, 365]  # daily demand

SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
MOIRAI_VER = "moirai_1.1_R_" + SIZE if args.moirai_ver is None else args.moirai_ver
CTX = 512  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 16  # batch size: any positive integer

# data formatting into GluonTS dataset
df = pd.read_csv(args.csv_path, index_col=0, parse_dates=True)
# df.drop(columns=['forecast'], inplace=True)
# print(pd.infer_freq(df.index))

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

total_len = len(df)
train_len = int(train_ratio * total_len)
val_len = int(val_ratio * total_len)
test_len = int(test_ratio * total_len)

border1s = [0, train_len - CTX, train_len + val_len - CTX]
border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

train_df = df[border1s[0]:border2s[0]]
val_df = df[border1s[1]:border2s[1]]
test_df = df[border1s[2]:border2s[2]]

if len(df.columns) == 1:
    past_dynamic_real_cols = None
    past_col_len = 0
else:
    past_dynamic_real_cols = list(df.columns[1:])
    past_col_len = len(past_dynamic_real_cols)

ds = PandasDataset(df, target=args.target, past_feat_dynamic_real=past_dynamic_real_cols)
train_ds = PandasDataset(train_df, target=args.target, past_feat_dynamic_real=past_dynamic_real_cols)
val_ds = PandasDataset(val_df, target=args.target, past_feat_dynamic_real=past_dynamic_real_cols)
test_ds = PandasDataset(test_df, target=args.target, past_feat_dynamic_real=past_dynamic_real_cols)

dummy_split, _ = split(
    ds, offset=-3
)  # dummy split the last 3 rows to align with the rest of the results
_, test_template = split(
    dummy_split, offset=-test_len + 1
)  # assign last test_len time steps as test set

# evaluation
for PDT in PDT_LIST:
    print(f"Evaluating for {MOIRAI_VER} PDT={PDT}...")
    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=test_len - PDT,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
        distance=1,  # number of time steps between each window (distance=PDT for non-overlapping windows)
    )

    if args.finetuned:
        if args.run_name == 'fast_eval':
            ckpt_path = Path(f'outputs/finetune/{MOIRAI_VER}/{args.yaml_prefix}/{args.run_name}/checkpoints/')
        else:
            ckpt_path = Path(f'outputs/finetune/{MOIRAI_VER}/{args.yaml_prefix}_{args.run_name}/{args.run_name}/checkpoints/')
        files = list(ckpt_path.glob('*'))
        ft_filepath = max(files, key=os.path.getmtime)  # get latest fine-tuned checkpoint
        print(f'Running model from {ft_filepath}')

        # load from FT'd checkpoint
        if '2.0' in MOIRAI_VER:
            model = Moirai2Forecast.load_from_checkpoint(
                prediction_length=PDT,
                context_length=CTX,
                target_dim=1,
                feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
                checkpoint_path=ft_filepath
            )
        else:
            model = MoiraiForecast.load_from_checkpoint(
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
                checkpoint_path=ft_filepath
            )
    else:
        # Prepare pre-trained model by downloading model weights from huggingface hub
        if '2.0' in MOIRAI_VER:
            model = Moirai2Forecast(
                module=Moirai2Module.from_pretrained(f"Salesforce/{MOIRAI_VER}"),
                prediction_length=PDT,
                context_length=CTX,
                target_dim=1,
                feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
            )
        else:
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/{MOIRAI_VER}"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
            )

    predictor = model.create_predictor(batch_size=BSZ, device=f'cuda:{args.gpu_id}')
    forecasts = predictor.predict(test_data.input)

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)

    forecast_out = []
    forecast_out_median = []
    label_out = []
    dates = []
    for _ in tqdm(range(test_data.windows)):
        inp = next(input_it)
        label = next(label_it)
        forecast = next(forecast_it)

        label_out.extend(label['target'])
        forecast_out.extend(forecast.mean)
        forecast_out_median.extend(forecast.median)
        dates.extend(forecast.index.to_timestamp())

    results = pd.DataFrame({
        'date': dates,
        'pred_mean': forecast_out,
        'pred_median': forecast_out_median,
        'true': label_out,
    })

    if args.finetuned:
        stage = 'finetuned'
    else:
        stage = 'zero_shot'

    results.to_csv(output_dir / f'{MOIRAI_VER}_{args.run_name}_pl{PDT}_feat{past_col_len}_{stage}.csv',
                   index=False)
    log_into_csv(results, args.run_name, stage,
                 seq_len=CTX, pred_len=PDT, n_features=past_col_len, bsz=BSZ,
                 log_file_name=args.yaml_prefix, pred_col_name='pred_mean')

    # Make predictions
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))
    # plot_next_multi(
    #     axes,
    #     input_it,
    #     label_it,
    #     forecast_it,
    #     context_length=CTX,
    #     intervals=(0.5, 0.9),
    #     dim=None,
    #     name="pred",
    #     show_label=True,
    # )
    # fig.savefig('test.png')
