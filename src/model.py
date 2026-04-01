import torch
import torch.nn as nn
import torch.nn.functional as F


class DenguePINN(nn.Module):
    """
    Physics-Informed Neural Network for dengue forecasting.

    Input features with default n_lags=2 (7 total):
      t, Temp_t, Temp_t-1, Temp_t-2, Rain_t, Rain_t-1, Rain_t-2

    Outputs:  S (susceptible), I (infected), beta (transmission rate)

    Architecture change (v2):
      Added a third hidden layer and increased default hidden_size to 32.
      With only 7 inputs and 3 outputs, 2 layers of size 16 (112 params) was
      too small to capture the nonlinear climate → incidence relationship.
      3 layers of size 32 (3,235 params) gives adequate capacity while
      remaining faster to train than large networks.
    """

    def __init__(self, n_lags: int = 2, hidden_size: int = 32, dropout_rate: float = 0.05):
        super(DenguePINN, self).__init__()
        self.n_lags  = n_lags
        n_inputs = 1 + 2 * (1 + n_lags)   # 1 + 2*3 = 7

        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, t, temp_lags, rain_lags):
        out  = self.net(torch.cat([t, temp_lags, rain_lags], dim=1))
        S    = torch.sigmoid(out[:, 0:1])
        I    = torch.sigmoid(out[:, 1:2])
        beta = F.softplus(out[:, 2:3])
        return S, I, beta