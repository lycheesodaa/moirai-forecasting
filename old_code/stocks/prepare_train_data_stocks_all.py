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
parser.add_argument('--dataset_name', type=str, required=True, default='Stocks')
parser.add_argument('--folder_path', type=str, required=True)
args = parser.parse_args()

save_path = Path(os.getenv('CUSTOM_DATA_PATH'))
types = ['train', 'val']

base_covars = ['open', 'high', 'low', 'adj close', 'volume']
num_features_map = {
    'sentiment': base_covars + ['positive', 'negative', 'neutral'],
    'emotion': base_covars + ['sadness', 'neutral_emotion', 'fear', 'anger', 'disgust', 'surprise', 'joy'],
    'historical': base_covars
}
emotion_type = str(args.dataset_name).split('_')[-1]
if emotion_type not in num_features_map.keys():
    raise Exception(f'Please enter a valid dataset_name in {list(num_features_map.keys())}.')

# Loop over all the stocks and concatenate them into one dataframe with the label column `stock`
df_train = pd.DataFrame(columns=num_features_map[emotion_type] + ['close', 'stock'])
df_val = pd.DataFrame(columns=num_features_map[emotion_type] + ['close', 'stock'])
print(len(df_train))
print(len(df_val))
for i, file in enumerate(os.listdir(Path(args.folder_path))):
    stock_name = (Path(args.folder_path) / file).stem
    temp_df = pd.read_csv(Path(args.folder_path) / file, index_col=0, parse_dates=True)
    temp_df['stock'] = stock_name
    temp_df = temp_df[num_features_map[emotion_type] + ['close', 'stock']]

    if len(temp_df) < 512 + 30:
        continue

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    df_train = pd.concat([df_train, temp_df[:int(len(temp_df) * train_ratio) + 1]])
    df_val = pd.concat([df_val, temp_df[:int(len(temp_df) * (train_ratio + val_ratio)) + 1]])

print(len(df_train))
print(len(df_val))

for ds_type in types:
    dataset = args.dataset_name

    if ds_type == 'train':
        df = df_train
    elif ds_type == 'val':
        df = df_val
        dataset += '_eval'

    def example_gen_func() -> Generator[dict[str, Any]]:
        for name, grouped_df in df.groupby('stock'):
            yield {
                "target": grouped_df['close'].to_numpy(),  # array of shape (time,)
                "start": grouped_df.index[0],
                "freq": 'B',
                "item_id": name,
                "past_feat_dynamic_real": grouped_df[num_features_map[emotion_type]].to_numpy().T,
            }

    features = Features(
        dict(
            target=Sequence(Value("float32")),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            item_id=Value("string"),
            past_feat_dynamic_real=Sequence(
                Sequence(Value("float32")), length=len(num_features_map[emotion_type])
            ),
        )
    )

    hf_dataset = datasets.Dataset.from_generator(example_gen_func, features=features).with_format("numpy")
    hf_dataset.info.dataset_name = dataset
    hf_dataset.save_to_disk(save_path / dataset)
    print(f'Exported to {dataset}.')