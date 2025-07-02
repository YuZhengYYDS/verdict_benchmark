import torch
import numpy as np
from typing import Tuple, Optional
from scipy import stats

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def rmse_loss(pred, target):
    """Root Mean Square Error"""
    return torch.sqrt(mse_loss(pred, target))

def mae_loss(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))

def mape_loss(pred, target, epsilon=1e-8):
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100

def r2_score_torch(pred, target):
    """R² coefficient of determination"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)

def pearson_correlation_torch(pred, target):
    """Pearson correlation coefficient"""
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    
    numerator = torch.sum((pred - pred_mean) * (target - target_mean))
    pred_std = torch.sqrt(torch.sum((pred - pred_mean) ** 2))
    target_std = torch.sqrt(torch.sum((target - target_mean) ** 2))
    
    correlation = numerator / (pred_std * target_std + 1e-8)
    return correlation

def normalized_rmse(pred, target):
    """Normalized RMSE by target range"""
    rmse = rmse_loss(pred, target)
    target_range = torch.max(target) - torch.min(target)
    return rmse / target_range

def concordance_correlation_coefficient(pred, target):
    """Concordance Correlation Coefficient (CCC)"""
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    
    pred_var = torch.var(pred)
    target_var = torch.var(target)
    
    covariance = torch.mean((pred - pred_mean) * (target - target_mean))
    
    ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2)
    return ccc

def compute_comprehensive_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        pred: Predictions tensor of shape (N, D)
        target: Ground truth tensor of shape (N, D)
    
    Returns:
        Dictionary of computed metrics
    """
    pred = pred.float()
    target = target.float()
    
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mse_loss(pred, target).item()
    metrics['rmse'] = rmse_loss(pred, target).item()
    metrics['mae'] = mae_loss(pred, target).item()
    metrics['mape'] = mape_loss(pred, target).item()
    metrics['r2'] = r2_score_torch(pred, target).item()
    metrics['nrmse'] = normalized_rmse(pred, target).item()
    metrics['ccc'] = concordance_correlation_coefficient(pred, target).item()
    
    # Per-parameter metrics
    n_params = pred.shape[1] if len(pred.shape) > 1 else 1
    
    if n_params > 1:
        metrics['per_param_mse'] = [mse_loss(pred[:, i], target[:, i]).item() for i in range(n_params)]
        metrics['per_param_rmse'] = [rmse_loss(pred[:, i], target[:, i]).item() for i in range(n_params)]
        metrics['per_param_mae'] = [mae_loss(pred[:, i], target[:, i]).item() for i in range(n_params)]
        metrics['per_param_r2'] = [r2_score_torch(pred[:, i], target[:, i]).item() for i in range(n_params)]
        metrics['per_param_correlation'] = [pearson_correlation_torch(pred[:, i], target[:, i]).item() for i in range(n_params)]
    else:
        metrics['per_param_mse'] = [metrics['mse']]
        metrics['per_param_rmse'] = [metrics['rmse']]
        metrics['per_param_mae'] = [metrics['mae']]
        metrics['per_param_r2'] = [metrics['r2']]
        metrics['per_param_correlation'] = [pearson_correlation_torch(pred, target).item()]
    
    return metrics

def statistical_significance_test(pred1: np.ndarray, pred2: np.ndarray, target: np.ndarray) -> dict:
    """
    Perform statistical significance tests between two models.
    
    Args:
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        target: Ground truth values
    
    Returns:
        Dictionary containing test results
    """
    # Calculate residuals
    residuals1 = np.abs(pred1 - target)
    residuals2 = np.abs(pred2 - target)
    
    # Paired t-test for difference in absolute errors
    t_stat, p_value = stats.ttest_rel(residuals1.flatten(), residuals2.flatten())
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(residuals1.flatten(), residuals2.flatten())
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(residuals1) ** 2 + np.std(residuals2) ** 2) / 2)
    cohens_d = (np.mean(residuals1) - np.mean(residuals2)) / pooled_std
    
    return {
        'paired_t_test': {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'wilcoxon_test': {
            'statistic': wilcoxon_stat,
            'p_value': wilcoxon_p,
            'significant': wilcoxon_p < 0.05
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': interpret_cohens_d(cohens_d)
        }
    }

def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"

def bootstrap_confidence_interval(pred: np.ndarray, target: np.ndarray, 
                                metric_func, n_bootstrap: int = 1000, 
                                confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        pred: Predictions
        target: Ground truth
        metric_func: Function to calculate metric (should take pred, target as args)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (metric_value, lower_bound, upper_bound)
    """
    n_samples = len(pred)
    bootstrap_metrics = []
    
    # Original metric
    original_metric = metric_func(pred, target)
    
    # Bootstrap sampling
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_pred = pred[indices]
        bootstrap_target = target[indices]
        
        # Calculate metric for bootstrap sample
        bootstrap_metric = metric_func(bootstrap_pred, bootstrap_target)
        bootstrap_metrics.append(bootstrap_metric)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_bound = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return original_metric, lower_bound, upper_bound
