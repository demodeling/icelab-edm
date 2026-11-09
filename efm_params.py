# from sklearn.metrics import mutual_info_score

from sklearn.feature_selection import mutual_info_regression
import numpy as np

# emb_vec, ind = build_embedding(vector,NN,1)

# MI = [mutual_info_regression(emb_vec[:,0].reshape(1, -1).T,emb_vec[:,j]) for j in range(1,NN) ]#take only up until maxTau to make sure each vector has same number of elements and no looping around

def find_best_tau(vector, NN,E=2, mi_threshold=0.1, mi_fraction=0.5):
    """
    Select tau by first-minimum of MI, but cap it early if MI drops
    below a threshold — indicating sufficient decorrelation already.

    Parameters
    ----------
    mi_threshold : float
        Absolute MI value below which we consider the signal decorrelated.
        Default 0.05 nats. Set to None to disable.
    mi_fraction  : float
        Fraction of MI[tau=1] below which we stop (relative threshold).
        e.g. 0.3 means "stop when MI < 30% of its initial value".
        Set to None to disable.
    """
    MI      = np.zeros(NN + 1)
    emb_len = np.zeros(NN + 1)

    for tau in range(1, NN):
        emb_vec, ind = build_embedding(vector, E, tau)
        if len(emb_vec) < 10:           # too few clean pairs — stop
            MI[tau - 1]      = np.nan
            emb_len[tau - 1] = 0
            continue
        MI[tau - 1]      = mutual_info_regression(
                               emb_vec[:, 0].reshape(-1, 1),
                               emb_vec[:, 1])[0]
        emb_len[tau - 1] = len(emb_vec)

    valid = ~np.isnan(MI[:NN-1])

    # ── threshold rules (applied before searching for a minimum) ──────────
    abs_thresh = mi_threshold if mi_threshold is not None else 0.0
    rel_thresh = (mi_fraction * MI[0]) if (mi_fraction is not None
                                            and not np.isnan(MI[0])
                                            and MI[0] > 0) else 0.0
    effective_thresh = max(abs_thresh, rel_thresh)

    # first tau where MI drops below the effective threshold
    below = np.where(valid & (MI[:NN-1] < effective_thresh))[0]
    if len(below) > 0:
        best_tau = int(below[0]) + 1   # +1 because tau loop starts at 1
        return MI, best_tau, emb_len

    # ── fallback: first local minimum ─────────────────────────────────────
    interior = np.where(valid[1:-1]
                        & (MI[1:NN-2] < MI[:NN-3])
                        & (MI[1:NN-2] < MI[2:NN-1]))[0] + 1
    if len(interior) > 0:
        best_tau = int(interior[0]) + 1
    else:
        best_tau = int(np.nanargmin(MI[:NN-1])) + 1

    return MI, best_tau, emb_len


def build_embedding(x, dim, tau):
    N = len(x) - (dim - 1) * tau
    emb = np.array([x[i:i + dim * tau:tau] for i in range(N)])
    mask = ~np.isnan(emb).any(axis=1)
    return emb[mask], np.where(mask)[0]  # (clean_emb, valid_start_indices)

# def find_best_tau(vector,NN,E=2):
#     MI =np.zeros((NN+1))
#     emb_len =np.zeros((NN+1))

#     for tau in range(1,NN):
#         emb_vec, ind = build_embedding(vector,E,tau)
#         MI[tau-1] = mutual_info_regression(emb_vec[:,0].reshape(1, -1).T,emb_vec[:,1])[0]
#         emb_len[tau-1] = len(emb_vec)
    
#     dips = np.where((MI[1:-1] < MI[:-2]) & (MI[1:-1] < MI[2:]))[0] + 1
#     best_tau = int(dips[0]) if len(dips) > 0 else int(np.argmin(MI[1:-1])) + 1
#     return MI, best_tau+1,emb_len




def run_tau_loop(dataTest,sort_asc,N,E,all_years,nweeks,row_idx, col_idx):
    NN = 8

    MI =np.zeros((N, NN+1))
    tau =np.zeros((N))
    emb_len=np.zeros((N, NN+1))

    for ix_val in range(0,N):
        
        example = dataTest[sort_asc[ix_val],:]
        # np.shape(emb_vec),np.shape(y_data_embedded)
        testData = np.full((len(all_years),nweeks), np.nan) #extra week for nan's
        testData[row_idx, col_idx] = example
        # example
        vector = testData.flatten()
    
        MI[ix_val,:], tau[ix_val], emb_len[ix_val,:] = find_best_tau(vector,NN,E)
    return MI, tau, emb_len

# sort_asc = sorted_vec[::-1]

# MI, tau, emb_len = run_tau_loop(dataTest,sort_asc,100, 2)

import plotly.graph_objects as go

def plot_diagnostic_tau(MI, NN,tau):
    fig = go.Figure()
    
    # MI curve
    fig.add_trace(go.Scatter(x=list(range(1, NN)), y=MI[:NN-1],
    mode='lines+markers', name='MI', line=dict(color='black')))
    
    # threshold lines
    fig.add_hline(y=0.1, line_dash='dash', line_color='#FFD700',
          annotation_text='abs threshold (0.1)', annotation_font_color='#FFD700')
    fig.add_hline(y=0.5 * MI[0], line_dash='dot', line_color='#FF4444',
          annotation_text='rel threshold (50% of MI[1])', annotation_font_color='black')
    
    # chosen tau
    fig.add_vline(x=tau, line_dash='dash', line_color='white',
          annotation_text=f'τ*={tau}', annotation_font_color='white')
    
    fig.update_layout(
    title='MI vs τ with thresholds',
    xaxis_title='τ', yaxis_title='Mutual Information (nats)',
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(color='black'),
    )
    fig.show()


import numpy as np
from sklearn.neighbors import KDTree
def fnn_from_embedding(emb_vec_d, emb_vec_d1, atol_std=2.0, rtol=15.0):
    """
    Run one FNN step given pre-built embedding clouds at dim d and d+1.
    emb_vec_d  : (N, d)   — already NaN-cleaned via build_embedding
    emb_vec_d1 : (M, d+1) — same
    Both must be aligned (same start indices).
    """
    n   = min(len(emb_vec_d), len(emb_vec_d1))
    Xd  = emb_vec_d[:n]
    Xd1 = emb_vec_d1[:n]

    sig_std = Xd[:, 0].std()
    tree    = KDTree(Xd)
    dists, idxs = tree.query(Xd, k=2)
    nn_dist_d  = dists[:, 1]
    nn_idx     = idxs[:, 1]
    nn_dist_d1 = np.linalg.norm(Xd1 - Xd1[nn_idx], axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(nn_dist_d > 1e-10,
                       np.abs(nn_dist_d1 - nn_dist_d) / nn_dist_d, np.inf)

    return ((rel > rtol) | (nn_dist_d1 > atol_std * sig_std)).mean()


def fnn_from_sparse_embeddings(vector, tau, max_dim=8):
    """Use build_embedding directly — handles NaNs natively."""
    dims, fnn_fracs, n_pts = [], [], []
    for d in range(1, max_dim + 1):
        ed,  ind_d  = build_embedding(vector, d,   tau)
        ed1, ind_d1 = build_embedding(vector, d+1, tau)
        # align on common start indices
        common = np.intersect1d(ind_d, ind_d1)
        if len(common) < 10:
            continue
        mask_d  = np.isin(ind_d,  common)
        mask_d1 = np.isin(ind_d1, common)
        frac = fnn_from_embedding(ed[mask_d], ed1[mask_d1])
        dims.append(d); fnn_fracs.append(frac); n_pts.append(len(common))

    dims      = np.array(dims)
    fnn_fracs = np.array(fnn_fracs)
    below     = np.where(fnn_fracs < 0.10)[0]
    best_dim  = int(dims[below[0]]) if len(below) > 0 \
                else int(dims[np.argmin(fnn_fracs)])
    return dims, fnn_fracs, best_dim, np.array(n_pts)
