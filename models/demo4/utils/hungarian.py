import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(cost_matrix):
    """
    使用匈牙利算法求解最小代价匹配
    cost_matrix: 形状 (n_obs, n_tar)
    返回: 匹配对列表 [(obs_idx, tar_idx), ...] 以及未匹配的观测索引和目标索引
    """
    n_obs, n_tar = cost_matrix.shape
    # 处理无穷大值，用大数替代
    max_cost = np.max(cost_matrix[np.isfinite(cost_matrix)]) if np.any(np.isfinite(cost_matrix)) else 1e6
    cost_filled = np.nan_to_num(cost_matrix, nan=max_cost*10, posinf=max_cost*10, neginf=0)
    row_ind, col_ind = linear_sum_assignment(cost_filled)
    matches = []
    matched_obs = set()
    matched_tar = set()
    for r, c in zip(row_ind, col_ind):
        if r < n_obs and c < n_tar and np.isfinite(cost_matrix[r, c]):
            matches.append((r, c))
            matched_obs.add(r)
            matched_tar.add(c)
    unmatched_obs = [i for i in range(n_obs) if i not in matched_obs]
    unmatched_tar = [j for j in range(n_tar) if j not in matched_tar]
    return matches, unmatched_obs, unmatched_tar