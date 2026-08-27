import h5py
import numpy as np

def read_nn_weights(nnpath, nn_type="LinNet"):
    with h5py.File(nnpath, "r") as f:

        xmin = f["xmin"][()]
        xmax = f["xmax"][()]

        wavelength = np.ascontiguousarray(f["wavelengths"][()], dtype=np.float64)
        # resolution = np.array(f["resolution"], dtype=float)

        nn_type = nn_type

        # ---- load weights once (NUMPY only) ----
        W = []
        b = []

        if nn_type == "LinNet":
            keys_W = ["model/lin1.weight",
                        "model/lin2.weight",
                        "model/lin3.weight",
                        "model/lin4.weight",
                        "model/lin5.weight",
                        "model/lin6.weight"]

            keys_b = ["model/lin1.bias",
                        "model/lin2.bias",
                        "model/lin3.bias",
                        "model/lin4.bias",
                        "model/lin5.bias",
                        "model/lin6.bias"]

        else:
            raise ValueError("Only LinNet shown here")

        for k in keys_W:
            # W.append(f[k][()])
            W.append(np.ascontiguousarray(f[k][()], dtype=np.float32))

        for k in keys_b:
            # b.append(f[k][()])
            b.append(np.ascontiguousarray(f[k][()], dtype=np.float32))

        f.close()

    return wavelength, W, b, xmin, xmax

def eval_nn__(x, W, b):
    h = np.asarray(x, dtype=np.float64)

    # from IPython import embed;embed();exit()

    # hidden layers
    for Wi, bi in zip(W[:-1], b[:-1]):
        h = h @ Wi.T + bi   # BLAS GEMM

    # output layer
    # W, b = W[-1], b[-1]
    out = h @ W[-1].T + b[-1]

    return out



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def eval_nn(x, W, b):
    h = np.asarray(x, dtype=np.float32)

    for Wi, bi in zip(W[:-1], b[:-1]):
        h = sigmoid(h @ Wi.T + bi)

    Wl, bl = W[-1], b[-1]
    out = h @ Wl.T + bl

    return out

def make_nn(W, b, xmin, xmax):

    def eval_nn(x):
        h = np.asarray(x, dtype=np.float32)
        h = (h - xmin) / (xmax - xmin) - 0.5

        for Wi, bi in zip(W[:-1], b[:-1]):
            h = sigmoid(h @ Wi.T + bi)

        Wl, bl = W[-1], b[-1]
        out = h @ Wl.T + bl
        return out

    return eval_nn

import numpy as np
from scipy.special import expit as sigmoid

def make_nn_batched(W, b, xmin, xmax, norm_factor=1.0):
    """
    Returns a function that evaluates the network on a batch of inputs at once.
    Input:  x of shape (N, d_in)  -- N independent input vectors
    Output: shape (N, d_out)
    """
    xmin = np.asarray(xmin, dtype=np.float32)
    xmax = np.asarray(xmax, dtype=np.float32)
    scale = np.float32(1.0) / (xmax - xmin)

    # Precompute once: transposed, contiguous, correctly-typed weights
    WT = [np.ascontiguousarray(Wi.T, dtype=np.float32) for Wi in W]
    b32 = [np.asarray(bi, dtype=np.float32) for bi in b]

    def eval_nn_batched(x):
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h[np.newaxis, :]          # promote a single vector to (1, d_in)
        h = (h - xmin) * scale - 0.5      # broadcasts over rows

        for Wi, bi in zip(WT[:-1], b32[:-1]):
            h = sigmoid(h @ Wi + bi)       # (N, d_in) @ (d_in, d_out) -> (N, d_out)

        out = h @ WT[-1] + b32[-1]

        return out*norm_factor

    return eval_nn_batched

import numpy as np
from scipy.special import expit as sigmoid

def make_nn_batched_pca(W, b, xmin, xmax, pca_basis, pca_mean, norm_factor=1.0):
    """
    Returns a function that evaluates the network on a batch of inputs at once.
    Input:  x of shape (N, d_in)  -- N independent input vectors
    Output: shape (N, d_out)
    """
    xmin = np.asarray(xmin, dtype=np.float32)
    xmax = np.asarray(xmax, dtype=np.float32)
    scale = np.float32(1.0) / (xmax - xmin)

    # Precompute once: transposed, contiguous, correctly-typed weights
    WT = [np.ascontiguousarray(Wi.T, dtype=np.float32) for Wi in W]
    b32 = [np.asarray(bi, dtype=np.float32) for bi in b]

    def eval_nn_batched(x):
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h[np.newaxis, :]          # promote a single vector to (1, d_in)
        h = (h - xmin) * scale - 0.5      # broadcasts over rows

        for Wi, bi in zip(WT[:-1], b32[:-1]):
            h = sigmoid(h @ Wi + bi)       # (N, d_in) @ (d_in, d_out) -> (N, d_out)

        coeffs = h @ WT[-1] + b32[-1]

        out = coeffs @ pca_basis + pca_mean
        return out*norm_factor

    return eval_nn_batched