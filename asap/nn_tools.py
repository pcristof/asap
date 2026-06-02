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
                        "model/lin4.weight",
                        "model/lin5.weight",
                        "model/lin6.weight"]

            keys_b = ["model/lin1.bias",
                        "model/lin4.bias",
                        "model/lin5.bias",
                        "model/lin6.bias"]

        else:
            raise ValueError("Only LinNet shown here")

        for k in keys_W:
            # W.append(f[k][()])
            W.append(np.ascontiguousarray(f[k][()], dtype=np.float64))

        for k in keys_b:
            # b.append(f[k][()])
            b.append(np.ascontiguousarray(f[k][()], dtype=np.float64))

        f.close()

    return wavelength, W, b

def eval_nn(x, W, b):
    h = np.asarray(x, dtype=np.float64)

    # from IPython import embed;embed();exit()

    # hidden layers
    for W, b in zip(W[:-1], b[:-1]):
        h = h @ W.T + b   # BLAS GEMM

    # output layer
    W, b = W[-1], b[-1]
    out = h @ W.T + b

    return out


def make_nn(W_list, b_list):
    Ws = tuple(W_list)
    bs = tuple(b_list)

    def eval_nn(x):
        h = np.asarray(x, dtype=np.float64)

        for Wi, bi in zip(Ws[:-1], bs[:-1]):
            h = h @ Wi.T + bi

        return h @ Ws[-1].T + bs[-1]

    return eval_nn