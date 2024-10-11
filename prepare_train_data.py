import os
import argparse
from collections.abc import Generator
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import datasets
import pandas as pd
from datasets import Features, Sequence, Value

load_dotenv()

parser = argparse.ArgumentParser(description='prepare_train_data')
parser.add_argument('--dataset_name', type=str, required=True, default='Demand')
parser.add_argument('--data_path', type=str, required=True, default='data/demand_data_all_cleaned_numerical.csv')
parser.add_argument('--has_forecast', type=bool, default=False)
args = parser.parse_args()

save_path = Path(os.getenv('CUSTOM_DATA_PATH'))
dataset = args.dataset_name
types = ['train', 'val']

for ds_type in types:
    df = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
    if args.has_forecast:
        df.drop(columns=['forecast'], inplace=True)

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    if ds_type == 'train':
        df = df[:int(len(df) * train_ratio)]
    elif ds_type == 'val':
        dataset += '_eval'
        df = df[:int(len(df) * (train_ratio + val_ratio))]

    def example_gen_func() -> Generator[dict[str, Any]]:
        if 'top0' in args.dataset_name:
            yield {
                "target": df['actual'].to_numpy(),  # array of shape (time,)
                "start": df.index[0],
                "freq": pd.infer_freq(df.index),
                "item_id": f"actual",
            }
        else:
            yield {
                "target": df['actual'].to_numpy(),  # array of shape (time,)
                "start": df.index[0],
                "freq": pd.infer_freq(df.index),
                "item_id": f"actual",
                "past_feat_dynamic_real": df.iloc[:, 1:].to_numpy().T,
            }


    if 'top0' in args.dataset_name:
        features = Features(
            dict(
                target=Sequence(Value("float32")),
                start=Value("timestamp[s]"),
                freq=Value("string"),
                item_id=Value("string"),
            )
        )
    else:
        features = Features(
            dict(
                target=Sequence(Value("float32")),
                start=Value("timestamp[s]"),
                freq=Value("string"),
                item_id=Value("string"),
                past_feat_dynamic_real=Sequence(
                    Sequence(Value("float32")), length=len(df.columns) - 1
                ),
            )
        )

    hf_dataset = datasets.Dataset.from_generator(example_gen_func, features=features).with_format("numpy")
    hf_dataset.info.dataset_name = dataset
    hf_dataset.save_to_disk(save_path / dataset)