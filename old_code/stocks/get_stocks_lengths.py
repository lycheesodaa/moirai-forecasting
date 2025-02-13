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

# folder_path = 'data/stocks/candle_w_emotion/day_average_content'
folder_path = '../../data/stocks/candle_w_emotion/day_average_headlines'

lengths = []
# Loop over all the stocks and concatenate them into one dataframe with the label column `stock`
for i, file in enumerate(os.listdir(Path(folder_path))):
    stock_name = (Path(folder_path) / file).stem
    temp_df = pd.read_csv(Path(folder_path) / file, index_col=0, parse_dates=True)

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    lengths.append({
        'stock': stock_name,
        'offset': int(len(temp_df) * train_ratio),
        'eval_length': int(len(temp_df) * (val_ratio + train_ratio)) - int(len(temp_df) * train_ratio)
    })

pd.DataFrame(lengths).to_csv(f'{folder_path}.csv', index=False)