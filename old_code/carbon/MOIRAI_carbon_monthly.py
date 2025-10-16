from pathlib import Path

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
from util import log_into_csv

parser = argparse.ArgumentParser(description='MOIRAI')
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--run_name', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--feats_pct', type=int)
parser.add_argument('--yaml_prefix', type=str)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--finetuned', type=bool, default=False)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

PDT_LIST = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # Predict X steps ahead
CTX_list = [2]

SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 16  # batch size: any positive integer

# Load in selected features based on the spearman correlation analysis
df = pd.read_csv(args.csv_path)
sel_features_df0 = pd.read_excel("data/carbon/res/ranked_abs_features_monthly.xlsx")
sel_features_df0.sort_values(by="Correlation", ascending=False, inplace=True)
num_features = int(len(sel_features_df0) * (args.feats_pct / 100))
sel_feature_names = sel_features_df0["Factor"][0:num_features].tolist()
sel_feature_names = [val for val in sel_feature_names if 'Historical Price' not in val]
df = df[['Month-Year', 'Price'] + sel_feature_names]
print(f"Selected features (total {len(sel_feature_names)}): {sel_feature_names}")

df['Month-Year'] = pd.to_datetime(df['Month-Year'])
df.set_index('Month-Year', inplace=True)
df.to_csv(f'data/carbon/res/merged_data_top{args.feats_pct}pct.csv')

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

total_len = len(df)
train_len = int(train_ratio * total_len)
val_len = int(val_ratio * total_len)
test_len = int(test_ratio * total_len)

# evaluation
for CTX in CTX_list:
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

    _, test_template = split(
        ds, offset=-test_len + 1
    )  # assign last test_len time steps as test set

    for PDT in PDT_LIST:
        print(f"Evaluating for CTX={CTX}, PDT={PDT}...")
        # Construct rolling window evaluation
        test_data = test_template.generate_instances(
            prediction_length=PDT,  # number of time steps for each prediction
            windows=test_len - PDT,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
            distance=1,  # number of time steps between each window (distance=PDT for non-overlapping windows)
            # max_history=CTX + PDT,
        )

        if args.finetuned:
            if args.run_name == 'fast_eval':
                ckpt_path = Path(f'outputs/finetune/moirai_1.0_R_large/{args.yaml_prefix}/{args.run_name}/checkpoints/')
            else:
                ckpt_path = Path(f'outputs/finetune/moirai_1.0_R_large/{args.yaml_prefix}_{args.run_name}/{args.run_name}/checkpoints/')
            files = list(ckpt_path.glob('*'))
            ft_filepath = max(files, key=os.path.getmtime)  # get latest fine-tuned checkpoint

            # load from FT'd checkpoint
            model = MoiraiForecast.load_from_checkpoint(
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
                # checkpoint_path='outputs/finetune/moirai_1.0_R_large/demand/fast_eval/checkpoints/epoch=25-step=2600.ckpt'
                checkpoint_path=ft_filepath
            )
        else:
            # Prepare pre-trained model by downloading model weights from huggingface hub
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
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

        results.to_csv(output_dir / f'MOIRAI_{args.run_name}_ctx_{CTX}_pl{PDT}_feat{past_col_len}_{stage}.csv',
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
