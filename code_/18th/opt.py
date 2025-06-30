"""
LSTM volatility forecasting with hyper-parameter tuning via
RandomizedSearchCV + skorch wrapper around a PyTorch model.

Changes vs. the original snippet
--------------------------------
✓ Uses TimeSeriesSplit so every CV fold respects chronology  
✓ Pulls per-epoch training loss from best_model.history to plot it  
✓ Corrects the scatter plot to compare values in the *original* scale  
✓ Removes the NameError (n_epochs / train_losses now defined)

Dependencies
------------
pip install pandas numpy scikit-learn scipy torch skorch matplotlib
"""

# -------------------------------------------------------------------
# 1) Imports
# -------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, loguniform

import torch
import torch.nn as nn
from skorch import NeuralNetRegressor

import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 2) Load & sort data
# -------------------------------------------------------------------
df = pd.read_csv('../../data/dataV.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True, ignore_index=True)

# -------------------------------------------------------------------
# 3) Encode Date numerically + select columns
# -------------------------------------------------------------------
df['Date_ordinal'] = df['Date'].apply(lambda dt: dt.toordinal())

feature_cols = [
    'Date_ordinal',
    'Inflation',
    'CPI',
    'Treasury_Yield',
    'Close',
    'SP500_Adj_Close',
    'Volume',
    'GDP',
    'mortage',
    'unemployement',
    'fed_fund_rate',
    'volatility',
    'returns',
    'EWMA_VM',
    'GARCH_VM',
    'EGARCH_VM',
    'RogersSatchell_VM',
    'garman_klass',
    'parkinson',
    'yang_zhang',
    'move'
]

target_col = 'volatility_forcast'

X_raw = df[feature_cols].values
y_raw = df[[target_col]].values

# -------------------------------------------------------------------
# 4) Standard scaling
# -------------------------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# -------------------------------------------------------------------
# 5) Sliding-window sequences (seq_len historical steps → 1 target)
# -------------------------------------------------------------------
def create_sequences(X, y, seq_len: int):
    """Return arrays shaped (N, seq_len, n_features) and (N, 1)."""
    Xs, ys = [], []
    T = len(X)
    for t in range(seq_len, T):
        Xs.append(X[t - seq_len:t, :])
        ys.append(y[t, :])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 20
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

# -------------------------------------------------------------------
# 6) Chronological train / test split (80 % / 20 %)
# -------------------------------------------------------------------
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# -------------------------------------------------------------------
# 7) LSTM model definition
# -------------------------------------------------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_dim1,
                             num_layers=1,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim1,
                             hidden_size=hidden_dim2,
                             num_layers=1,
                             batch_first=True)
        self.fc = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        last = out2[:, -1, :]          # shape: (batch, hidden_dim2)
        return self.fc(last)           # shape: (batch, 1)

# -------------------------------------------------------------------
# 8) Wrap with skorch NeuralNetRegressor
# -------------------------------------------------------------------
net = NeuralNetRegressor(
    module=LSTMForecast,
    module__input_dim=len(feature_cols),
    criterion=nn.MSELoss,
    optimizer=torch.optim.Adam,
    max_epochs=50,
    batch_size=128,            # will be tuned
    optimizer__lr=1e-3,        # will be tuned
    train_split=None,          # CV handled outside via TimeSeriesSplit
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# -------------------------------------------------------------------
# 9) Hyper-parameter search space
# -------------------------------------------------------------------
param_dist = {
    'module__hidden_dim1': randint(16, 128),
    'module__hidden_dim2': randint(16, 64),
    'batch_size':           randint(16, 256),
    'optimizer__lr':        loguniform(1e-4, 1e-2),
}

# -------------------------------------------------------------------
# 10) RandomizedSearchCV with time-series aware folds
# -------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=3)

rs = RandomizedSearchCV(
    estimator=net,
    param_distributions=param_dist,
    n_iter=200,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=2,
    n_jobs=-1,
)

# skorch/torch need float32
rs.fit(X_train.astype(np.float32), y_train.astype(np.float32))

print("Best parameters:", rs.best_params_)
print("Best CV MSE:    ", -rs.best_score_)

# -------------------------------------------------------------------
# 11) Evaluate on the hold-out set
# -------------------------------------------------------------------
best_model = rs.best_estimator_

y_pred_scaled = best_model.predict(X_test.astype(np.float32))
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test)

mse  = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_orig, y_pred)
r2   = max(0.0, r2_score(y_test_orig, y_pred))

print(f"\nTest MSE : {mse:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE : {mae:.6f}")
print(f"Test R²  : {r2:.4f}")

# -------------------------------------------------------------------
# 12) Plot training loss vs. epoch
# -------------------------------------------------------------------
history      = best_model.history  # list of dicts
train_losses = [row['train_loss'] for row in history]
n_epochs     = len(train_losses)

plt.figure(figsize=(8, 4))
plt.plot(range(1, n_epochs + 1), train_losses, linewidth=2)
plt.title("Training Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 13) Scatter plot: actual vs. predicted (original scale)
# -------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(y_test_orig, y_pred, alpha=0.6, label="Predicted")
axis_min = min(y_test_orig.min(), y_pred.min())
axis_max = max(y_test_orig.max(), y_pred.max())
plt.plot([axis_min, axis_max], [axis_min, axis_max],
         color='red', linestyle='--', label="Ideal")
plt.title("Predicted vs. Actual Volatility")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
