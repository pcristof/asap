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