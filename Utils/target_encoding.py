import numpy as np
import pandas as pd

def add_noise(series, noise_level):
    """Adds Gaussian noise to avoid the same values from the original different categories"""
    return series * (1 + noise_level * np.random.randn(len(series)))

def compute_averages(data, target_name, min_samples_leaf, smoothing, global_mean):
    """
    The smaller grouped_data["count"] leads to the bigger -(grouped_data["count"] - min_samples_leaf) / smoothing)
    Then the np.exp gets bigger, the 1 / (1 + np.exp) gets smaller.
    Hence, the smaller count leads to the smaller smoothing value.
    In conclusion, the smaller count leads to the bigger (1 - smoothing_value).
    The goal is I wanna use the global weight to contribute to the rare category.
    """
    averages = data.groupby(data.name)[target_name].agg(["mean", "count"])
    smoothing_value = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    averages[target_name] = global_mean * (1 - smoothing_value) + averages["mean"] * smoothing_value
    return averages.drop(["mean", "count"], axis=1)

def merge_data(series, averages, global_mean):
    merged = pd.merge(
        series.to_frame(series.name),
        averages.reset_index().rename(columns={'index': series.name, series.name: 'average'}),
        on=series.name,
        how='left')['average'].rename(series.name + '_mean').fillna(global_mean)
    merged.index = series.index
    return merged

def target_encode(trn_series, val_series, tst_series, target, min_samples_leaf=1, smoothing=1, noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    
    global_mean = target.mean()
    
    # Compute the averages with smoothing
    averages = compute_averages(trn_series, target.name, min_samples_leaf, smoothing, global_mean)
    
    # Merge the original data with the computed averages
    ft_trn_series = merge_data(trn_series, averages, global_mean)
    ft_val_series = merge_data(val_series, averages, global_mean)
    ft_tst_series = merge_data(tst_series, averages, global_mean)
    
    # Apply noise if specified
    return (add_noise(ft_trn_series, noise_level),
            add_noise(ft_val_series, noise_level),
            add_noise(ft_tst_series, noise_level))
