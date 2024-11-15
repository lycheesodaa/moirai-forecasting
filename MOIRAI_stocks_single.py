import warnings
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

parser = argparse.ArgumentParser(description='MOIRAI')
parser.add_argument('--folder_path', type=str, required=True)
parser.add_argument('--stock_name', type=str)
parser.add_argument('--output_dir', type=str, default='./results/data/')
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--desc_prefix', type=str, default='')
parser.add_argument('--finetuned', type=bool, default=False)
parser.add_argument('--combine_dfs', type=bool, default=False)
parser.add_argument('--yaml_cfg', type=str)
parser.add_argument('--run_name', type=str, default='headlines',
                    choices=['headlines_sentiment', 'content_sentiment',
                             'headlines_emotion', 'content_emotion',
                             'headlines_historical', 'content_historical'])
args = parser.parse_args()

SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
PDT_LIST = [1, 3, 7, 14]  # prediction length: any positive integer
CTX = 512  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32 # batch size: any positive integer

# setting up the covariates for model/data configuration
base_covars = ['open', 'high', 'low', 'adj close', 'volume']
num_features_map = {
    'sentiment': base_covars + ['positive', 'negative', 'neutral'],
    'emotion': base_covars + ['sadness', 'neutral_emotion', 'fear', 'anger', 'disgust', 'surprise', 'joy'],
    'historical': base_covars
}
past_dynamic_real_cols = num_features_map[str(args.run_name).split('_')[-1]]
num_past_feat_dynamic_real = len(past_dynamic_real_cols)

stock_csv = args.stock_name + '.csv'

# concatenate all singular files from each PDT and export
if args.combine_dfs:
    print('Combining all stock dataframes only...')
    all_files = os.listdir(Path(args.output_dir))
    for PDT in PDT_LIST:
        df_list = []
        pdt_files = [file for file in all_files if f'pl{PDT}' in file]
        for file in pdt_files:
            df_temp = pd.read_csv(Path(args.output_dir) / file)
            df_temp['stock'] = (Path(args.output_dir) / file.split('_')[-1]).stem
            df_list.append(df_temp)
        pd.concat(df_list).to_csv(Path(args.output_dir).parent / f'MOIRAI_pl{PDT}_finetuned.csv', index=False)
    exit()


def get_test_data(csv_path, pred_len):
    # data formatting into GluonTS dataset
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(df) == 0:
        raise IndexError(f'Dataset {Path(csv_path).stem} is empty.')
    elif len(df) <= CTX + pred_len:
        raise IndexError(f'Dataset {Path(csv_path).stem} is shorter than the required length {CTX + pred_len}.')

    df = df.asfreq('B')

    test_ratio = 0.15
    total_len = len(df)
    test_len = int(test_ratio * total_len)

    ds = PandasDataset(df, target='close', past_feat_dynamic_real=past_dynamic_real_cols, freq='B')

    # dummy split the last 3 rows to align with the rest of the results
    dummy_split, _ = split(
        ds, offset=-3
    )
    # assign last test_len time steps as test set
    _, test_template = split(
        dummy_split, offset=-test_len + 1
    )

    # Construct rolling window evaluation data
    return test_template.generate_instances(
        prediction_length=pred_len,  # number of time steps for each prediction
        windows=test_len - pred_len,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
        distance=1,  # number of time steps between each window (distance=PDT for non-overlapping windows)
    )


# TODO normalize dataset values during training?
# evaluation
for PDT in PDT_LIST:
    print(f"Evaluating for PDT={PDT}...")
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
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=num_past_feat_dynamic_real,
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
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=num_past_feat_dynamic_real,
        )
    predictor = model.create_predictor(batch_size=BSZ, device=f'cuda:{args.gpu_id}')

    print(f'Forecasting for {stock_csv}...')
    warnings.filterwarnings("ignore")
    test_data = get_test_data(Path(args.folder_path) / stock_csv, PDT)

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

    to_export = pd.DataFrame({
        'date': dates,
        'pred_mean': forecast_out,
        'pred_median': forecast_out_median,
        'true': label_out,
        'stock': (Path(args.folder_path) / stock_csv).stem,
    })

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    to_export.to_csv(args.output_dir + f'MOIRAI_pl{PDT}_{stock_csv}', index=False)
