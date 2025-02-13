import warnings
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import sys
import os


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, context_length, prediction_length):
        self.features = features
        self.targets = targets
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.features) - self.context_length - self.prediction_length + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.context_length]
        y = self.targets[idx + self.context_length:idx + self.context_length + self.prediction_length]
        return torch.FloatTensor(x), torch.FloatTensor(y).squeeze(-1)


class BiLSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(BiLSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output(out)
        return out


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, savepath='outputs/lstm', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.savepath = Path(savepath)

        self.savepath.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss

        torch.save(model.state_dict(), self.savepath / 'checkpoint.pt')


def train_model(model, train_loader, val_loader, criterion, optimizer,
                device, epochs, scheduler=None, savepath='outputs/lstm'):
    early_stopping = EarlyStopping(patience=3, savepath=savepath, verbose=True)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]') as train_pbar:
            for X, y in train_pbar:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                train_batches += 1
                train_pbar.set_postfix({'loss': total_train_loss / train_batches})

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Valid]') as val_pbar:
                for X, y in val_pbar:
                    X, y = X.to(device), y.to(device)
                    output = model(X)
                    loss = criterion(output, y)

                    total_val_loss += loss.item()
                    val_batches += 1
                    val_pbar.set_postfix({'loss': total_val_loss / val_batches})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(savepath + '/checkpoint.pt', weights_only=True))
    return model, train_losses, val_losses, epoch + 1


# Data processing part
def process_stock_data(df, feature_columns):
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Forward fill NANs
    df = df.ffill()
    # Backward fill any remaining NANs at the start
    df = df.bfill()

    # Extract features and target
    feature_data = df[feature_columns].values
    target_data = df[['close']].values

    # Use RobustScaler instead of StandardScaler for better handling of outliers
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    # Fit and transform data
    feature_data_scaled = feature_scaler.fit_transform(feature_data)
    target_data_scaled = target_scaler.fit_transform(target_data)

    # Verify no NANs after processing
    if np.isnan(feature_data_scaled).any() or np.isnan(target_data_scaled).any():
        raise ValueError("NANs present after scaling")

    return feature_data_scaled, target_data_scaled, feature_scaler, target_scaler


# Parse arguments and set hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM Forecasting')
parser.add_argument('--folder_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./results/data/')
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--desc_prefix', type=str, default='')
parser.add_argument('--run_name', type=str, default='headlines',
                    choices=['headlines_sentiment', 'content_sentiment',
                             'headlines_emotion', 'content_emotion',
                             'headlines_historical', 'content_historical'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

CTX = 512
BSZ = 32
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

base_covars = ['open', 'high', 'low', 'adj close', 'volume']
num_features_map = {
    'sentiment': base_covars + ['positive', 'negative', 'neutral'],
    'emotion': base_covars + ['sadness', 'neutral_emotion', 'fear', 'anger', 'disgust', 'surprise', 'joy'],
    'historical': base_covars
}
past_dynamic_real_cols = num_features_map[str(args.run_name).split('_')[-1]]
feature_columns = ['close'] + past_dynamic_real_cols
INPUT_DIM = len(feature_columns)

device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
PDT_LIST = [1, 3, 7, 14, 30]

# Dictionary to store scalers for each stock
feature_scalers = {}
target_scalers = {}

for PDT in PDT_LIST:
    print(f"Processing prediction length {PDT}...")

    all_train_datasets = []
    all_val_datasets = []
    test_datasets = {}
    test_dates = {}
    error_files = {}

    # Process each file independently
    print("Processing individual datasets...")
    for file in os.listdir(args.folder_path):
        try:
            df = pd.read_csv(Path(args.folder_path) / file, index_col=0, parse_dates=True)
            if len(df) <= CTX + PDT:
                error_files[file] = "Dataset too short"
                continue

            # Print some statistics about the data
            # print(f"\nProcessing {file}")
            # print("Data shape:", df.shape)
            # print("NaN counts:", df[feature_columns].isna().sum())
            # print("Inf counts:", np.isinf(df[feature_columns]).sum())

            feature_data_scaled, target_data_scaled, feature_scaler, target_scaler = process_stock_data(df,
                                                                                                        feature_columns)

            # Print statistics about scaled data
            # print("Scaled features stats:")
            # print("Mean:", np.mean(feature_data_scaled))
            # print("Std:", np.std(feature_data_scaled))
            # print("Min:", np.min(feature_data_scaled))
            # print("Max:", np.max(feature_data_scaled))

            # Store scalers
            feature_scalers[file] = feature_scaler
            target_scalers[file] = target_scaler

            # Split points
            total_len = len(feature_data_scaled)
            train_end = int(0.6 * total_len)
            val_end = int(0.8 * total_len)

            # Include context window in validation and test sets
            train_features = feature_data_scaled[:train_end]
            train_targets = target_data_scaled[:train_end]

            val_features = feature_data_scaled[train_end - CTX:val_end]
            val_targets = target_data_scaled[train_end - CTX:val_end]

            test_features = feature_data_scaled[val_end - CTX:]
            test_targets = target_data_scaled[val_end - CTX:]

            if train_features.shape[0] < CTX + PDT:
                error_files[file] = "Dataset too short"
                continue

            # Create datasets
            train_dataset = TimeSeriesDataset(train_features, train_targets, CTX, PDT)
            val_dataset = TimeSeriesDataset(val_features, val_targets, CTX, PDT)
            test_datasets[file] = TimeSeriesDataset(test_features, test_targets, CTX, PDT)

            # Store test dates
            test_dates[file] = df.index[val_end:]

            all_train_datasets.append(train_dataset)
            all_val_datasets.append(val_dataset)

        except Exception as e:
            error_files[file] = str(e)
            continue

    # Combine datasets and create dataloaders
    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_val_dataset = ConcatDataset(all_val_datasets)

    train_loader = DataLoader(combined_train_dataset, batch_size=BSZ, shuffle=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=BSZ, shuffle=False)

    # Initialize and train model
    print("Training model...")
    model = BiLSTMForecast(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=PDT,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Train model
    model, train_losses, val_losses, epochs_run = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, args.epochs, scheduler, savepath=f'outputs/lstm/pl{PDT}'
    )


    # Evaluate on test sets
    print("Evaluating on individual test sets...")
    all_results = []

    for file, test_dataset in test_datasets.items():
        test_loader = DataLoader(test_dataset, batch_size=BSZ, shuffle=False)
        target_scaler = target_scalers[file]  # Get the specific target scaler for this stock

        model.eval()
        forecasts = []
        labels = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                forecasts.append(output.cpu().numpy())
                labels.append(y.cpu().numpy())

        forecasts = np.concatenate(forecasts)
        labels = np.concatenate(labels)

        # Inverse transform predictions and labels using target scaler
        forecasts_unscaled = target_scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()
        labels_unscaled = target_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

        test_dates_subset = test_dates[file][:len(forecasts) + PDT - 1]
        date_windows = []
        for i in range(len(test_dates_subset) - PDT + 1):
            # Add each date in the prediction window
            date_windows.extend(test_dates_subset[i:i + PDT])

        # print(len(date_windows))
        # print(forecasts_unscaled.shape)
        # print(labels_unscaled.shape)

        # Create results DataFrame
        to_export = pd.DataFrame({
            'date': date_windows,
            'pred_mean': forecasts_unscaled,
            'true': labels_unscaled,
            'stock': Path(file).stem,
        })
        all_results.append(to_export)

    # Save combined results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    result = pd.concat(all_results)
    result.to_csv(args.output_dir + f'BiLSTM_pl{PDT}_combined_training.csv', index=False)

    print('Errors occurred with files:')
    print(error_files)