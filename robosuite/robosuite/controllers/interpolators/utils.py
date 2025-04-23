import numpy as np
from scipy.spatial.distance import euclidean

def compute_3d_error(ground_truth, interpolated):
    """Compute error metrics between trajectories"""
    min_len = min(len(ground_truth), len(interpolated))
    errors = np.linalg.norm(ground_truth[:min_len] - interpolated[:min_len], axis=1)
    
    return {
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'total_error': np.sum(errors),
        'error_array': errors
    }