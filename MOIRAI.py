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

parser = argparse.ArgumentParser(description='MOIRAI')
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--desc_prefix', type=str, default='')
parser.add_argument('--finetuned', type=bool, default=False)
parser.add_argument('--yaml_cfg', type=str)
parser.add_argument('--run_name', type=str)
args = parser.parse_args()

SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
PDT_LIST = [1, 12, 24, 36, 48, 60, 72]  # prediction length: any positive integer
CTX = 512  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 16  # batch size: any positive integer

# data formatting into GluonTS dataset
# df = pd.read_csv('data/demand_data_all_cleaned_numerical.csv', index_col=0, parse_dates=True)
df = pd.read_csv(args.csv_path, index_col=0, parse_dates=True)
# df.drop(columns=['forecast'], inplace=True)

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
else:
    past_dynamic_real_cols = list(df.columns[1:])

ds = PandasDataset(df, target='actual', past_feat_dynamic_real=past_dynamic_real_cols)
train_ds = PandasDataset(train_df, target='actual', past_feat_dynamic_real=past_dynamic_real_cols)
val_ds = PandasDataset(val_df, target='actual', past_feat_dynamic_real=past_dynamic_real_cols)
test_ds = PandasDataset(test_df, target='actual', past_feat_dynamic_real=past_dynamic_real_cols)

# Group time series into multivariate dataset
# grouper = MultivariateGrouper(train_len)
# multivar_train_ds = grouper(train_ds)
# grouper = MultivariateGrouper(val_len)
# multivar_val_ds = grouper(val_ds)
# grouper = MultivariateGrouper(test_len)
# multivar_test_ds = grouper(test_ds)

dummy_split, _ = split(
    ds, offset=-3
)  # dummy split the last 3 rows to align with the rest of the results
_, test_template = split(
    dummy_split, offset=-test_len + 1
)  # assign last test_len time steps as test set

# TODO normalize dataset values during training?
# evaluation
for PDT in PDT_LIST:
    print(f"Evaluating for PDT={PDT}...")
    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        # windows=6,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
        windows=test_len - PDT,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
        # windows=test_len // PDT,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
        distance=1,  # number of time steps between each window (distance=PDT for non-overlapping windows)
        # max_history=CTX + PDT,
    )

    if args.finetuned:
        ckpt_path = f'outputs/finetune/moirai_1.0_R_large/{args.yaml_cfg}/{args.run_name}/checkpoints/'
        filename = os.listdir(ckpt_path)[0]

        # load from FT'd checkpoint
        model = MoiraiForecast.load_from_checkpoint(
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=test_ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=test_ds.num_past_feat_dynamic_real,
            # feat_dynamic_real_dim=0,
            # past_feat_dynamic_real_dim=0,
            # checkpoint_path='outputs/finetune/moirai_1.0_R_large/demand/fast_eval/checkpoints/epoch=25-step=2600.ckpt'
            checkpoint_path=ckpt_path + filename
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
            # feat_dynamic_real_dim=0,
            # past_feat_dynamic_real_dim=0,
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
        results.to_csv(f'MOIRAI{args.desc_prefix}_pl{PDT}_finetuned.csv', index=False)
    else:
        results.to_csv(f'MOIRAI{args.desc_prefix}_pl{PDT}_zero_shot.csv', index=False)

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