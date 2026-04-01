import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def calculate_physics_loss(S_pred, I_pred, beta_pred, gamma, t):
    """SIR ODE residual loss — enforces dS/dt and dI/dt consistency."""
    dS_dt = torch.autograd.grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    res_S = dS_dt - (-beta_pred * S_pred * I_pred)
    res_I = dI_dt - ( beta_pred * S_pred * I_pred - gamma * I_pred)
    return torch.mean(res_S**2) + torch.mean(res_I**2)


def calculate_standard_metrics(y_true, y_pred, split_label=""):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    lbl  = f" ({split_label})" if split_label else ""
    print(f"\n--- Standard Regression Metrics{lbl} ---")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return mse, rmse, mae, r2


def calculate_time_series_metrics(y_true, y_pred, time_array, split_label=""):
    """Peak Timing Error and DTW distance."""
    timing_error = abs(time_array[np.argmax(y_true)] - time_array[np.argmax(y_pred)])
    dtw_dist, _  = fastdtw(y_true.reshape(-1,1), y_pred.reshape(-1,1), dist=euclidean)
    lbl = f" ({split_label})" if split_label else ""
    print(f"--- Epidemiological & Time-Series Metrics{lbl} ---")
    print(f"  Peak Timing Error : {timing_error:.1f} months")
    print(f"  DTW Distance      : {dtw_dist:.4f}\n")
    return timing_error, dtw_dist


def print_overfitting_report(train_r2, test_r2, train_rmse, test_rmse):
    """Prints a plain-language generalisation verdict."""
    r2_gap   = train_r2  - test_r2
    rmse_gap = test_rmse - train_rmse
    print("\n═══════════════════════════════════════")
    print("       OVERFITTING DIAGNOSTIC")
    print("═══════════════════════════════════════")
    print(f"  Train R²  : {train_r2:.4f}   Test R²  : {test_r2:.4f}   Gap: {r2_gap:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}   Test RMSE: {test_rmse:.4f}   Gap: {rmse_gap:.4f}")
    if   r2_gap < 0.05 and test_r2 > 0.80: verdict = "GOOD     — model generalises well"
    elif r2_gap < 0.15 and test_r2 > 0.60: verdict = "MODERATE — mild overfitting"
    elif test_r2 < 0:                       verdict = "SEVERE   — worse than predicting the mean"
    else:                                   verdict = "HIGH     — significant overfitting"
    print(f"  Verdict   : {verdict}")
    print("═══════════════════════════════════════\n")