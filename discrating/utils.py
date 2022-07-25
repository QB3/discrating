import numpy as np
from numpy.linalg import norm
from numba import njit
from sklearn.utils import check_random_state
from scipy.sparse import issparse, csc_matrix, lil_matrix


def get_energy(us, vs):
    n_features, n_components = us.shape
    energy = np.zeros((n_features, n_features))
    for k in range(n_components):
        energy += np.outer(us[:, k], vs[:, k]) - np.outer(vs[:, k], us[:, k])
    return energy

def squared_loss(A, us, vs, mask=None):
    if issparse(A):
        return squared_loss_sparse(A, us, vs, mask=mask)
    energy = get_energy(us, vs)
    if mask is None:
        return ((A - 1/ 2 - energy) ** 2).mean() / 2
    else:
        return (((A - 1/ 2 - energy) * mask) ** 2).sum() / (2 * mask.sum())


def squared_loss_sparse(A, us, vs, mask=None):
    energy = get_energy(us, vs)
    nnz = A != 0
    nnz_dense = nnz.todense()
    if mask is None:
        # this is ugly, use lil matrix?
        return ((np.array((A[nnz] - 1/ 2 - energy[nnz_dense])).reshape(-1,)) ** 2).mean() / 2
    else:
        result = (
            (np.array((A[nnz] - 1 / 2 - energy[nnz_dense])).reshape(-1,) * mask[nnz.todense()]) ** 2).sum() / (2 * mask.sum())
        return result

def grad_squared_loss_u(A, us, vs, k, mask=None):
    if issparse(A):
        return grad_squared_loss_u_sparse(A, us, vs, k, mask=mask)
    n_features = A.shape[0]
    energy = get_energy(us, vs)
    grad_L = - (A - 1/ 2 - energy)
    if mask is None:
        grad_u = (grad_L @ vs[:, k] - grad_L.T @ vs[:, k]) / n_features ** 2
    else:
        grad_u = (grad_L * mask @ vs[:, k] - grad_L.T * mask @ vs[:, k]) / mask.sum()
    return grad_u


def grad_squared_loss_u_sparse(A, us, vs, k, mask=None):
    n_features = A.shape[0]
    energy = get_energy(us, vs)
    # energy = np.outer(u, v) - np.outer(v, u)
    nnz = A != 0
    v = vs[:, k]
    grad_L = csc_matrix(A.shape)
    grad_L[nnz] = - (A[nnz] - 1/ 2 - energy[nnz.todense()])
    if mask is None:
        grad_u = (grad_L @ v - grad_L.T @ v) / n_features ** 2
    else:
        grad_u = (grad_L * mask @ v - grad_L.T * mask @ v) / mask.sum()
    return grad_u


def grad_squared_loss_v(A, us, vs, k, mask=None):
    if issparse(A):
        return grad_squared_loss_v_sparse(A, us, vs, k, mask=mask)
    n_features = A.shape[0]
    energy = get_energy(us, vs)
    grad_L = - (A - 1/ 2 - energy)
    if mask is None:
        grad_v = (grad_L.T @ us[:, k] - grad_L @ us[:, k]) / n_features ** 2
    else:
        grad_v = (
            (grad_L.T * mask) @ us[:, k] - (grad_L * mask) @ us[:, k]) / mask.sum()
    return grad_v

def grad_squared_loss_v_sparse(A, us, vs, k, mask=None):
    n_features = A.shape[0]
    energy = get_energy(us, vs)
    # energy = np.outer(u, v) - np.outer(v, u)
    nnz = A != 0
    grad_L = csc_matrix(A.shape)
    grad_L[nnz] = - (A[nnz] - 1/ 2 - energy[nnz.todense()])
    u = us[:, k]
    if mask is None:
        grad_v = (grad_L.T @ u - grad_L @ u) / n_features ** 2
    else:
        grad_v = ((grad_L.T * mask) @ u - (grad_L * mask) @ u) / mask.sum()
    return grad_v


def squared_sigmoid_loss_sparse(payoff, us, vs):
    energy = get_energy(us, vs)
    sigmo = sigmoid_stable(energy)
    # nnz = (A != 0).todense()
    nnz = (payoff != 0)
    value = ((np.array((sigmo - payoff))[nnz.todense()]) ** 2).sum() / 2 / nnz.sum()
    return value

def squared_sigmoid_loss(A, us, vs, mask=None):
    if issparse(A):
        return squared_sigmoid_loss_sparse(A, us, vs)
    d = A.shape[0]
    energy = get_energy(us, vs)
    # energy = np.outer(u, v) - np.outer(v, u)
    sigmo = sigmoid_stable(energy)
    if mask is None:
        value = ((sigmo - A) ** 2).mean() / 2
    else:
        value = (((sigmo - A) * mask) ** 2).sum() / 2 / mask.sum()
    return value


def grad_squared_sigmoid_loss_u(A, us, vs, k, mask=None):
    d = A.shape[0]
    energy = get_energy(us, vs)
    v = vs[:, k]
    sigmo = sigmoid_stable(energy)
    grad_L = sigmo * (1 - sigmo) * (sigmo - A)
    if mask is None:
        grad_u = (grad_L @ v - grad_L.T @ v) / d ** 2
    else:
        grad_u = ((grad_L * mask) @ v - (grad_L.T * mask) @ v) / mask.sum()
    return grad_u


def grad_squared_sigmoid_loss_v(A, us, vs, k, mask=None):
    d = A.shape[0]
    # energy = np.outer(u, v) - np.outer(v, u)
    energy = get_energy(us, vs)
    u = vs[:, k]
    sigmo = sigmoid_stable(energy)
    grad_L = sigmo * (1 - sigmo) * (sigmo - A)
    if mask is None:
        grad_v = (grad_L.T @ u - grad_L @ u) / d ** 2
    else:
        grad_v = ((grad_L.T * mask) @ u - (grad_L * mask) @ u) / mask.sum()
    return grad_v


def grad_log_loss_uv(A, u, v, mask=None):
    if issparse(A):
        grad_u = grad_log_loss_u_sparse(
            A.data, A.indptr, A.indices, A.nnz, u, v)
        grad_v = grad_log_loss_v_sparse(
            A.data, A.indptr, A.indices, A.nnz, u, v)
        return np.concatenate([grad_u, grad_v], axis=0)
    else:
        return


def grad_log_loss_u(A, us, vs, k, mask=None):
    if issparse(A):
        # TODO
        return grad_log_loss_u_sparse(
            A.data, A.indptr, A.indices, A.nnz, us, vs, k)
    else:
        d = A.shape[0]
        # too costly for sparse matrices?
        energy = get_energy(us, vs)
        v = vs[:, k]
        # energy = np.outer(u, v) - np.outer(v, u)
        grad_bce = grad_bce_stable(A, energy)
        if mask is None:
            grad_u = (grad_bce @ v - grad_bce.T @ v) / d ** 2
        else:
            grad_u = (
                (grad_bce * mask) @ v - (grad_bce.T * mask) @ v) / mask.sum()
        return grad_u


@njit
def grad_log_loss_u_sparse(X_data, X_indptr, X_indices, nnz, us, vs, k):
    n_features = X_indptr.shape[0] - 1
    grad_u = np.zeros(n_features)
    v = vs[:, k]
    for j in range(n_features):
        for i in range(X_indptr[j], X_indptr[j+1]):
            idx_i = X_indices[i]
            energy_ij = get_energy_ij(us, vs, idx_i, j)
            # energy_ij = (u[idx_i] * v[j] - v[idx_i] * u[j])
            grad_u[idx_i] += grad_bce_stable_ij(X_data[i], energy_ij) * v[j]

    for j in range(n_features):
        for i in range(X_indptr[j], X_indptr[j+1]):
            idx_i = X_indices[i]
            energy_ij = get_energy_ij(us, vs, idx_i, j)
            # energy_ij = (u[idx_i] * v[j] - v[idx_i] * u[j])
            grad_u[j] -= grad_bce_stable_ij(X_data[i], energy_ij) * v[idx_i]
    grad_u /= nnz
    return grad_u


def grad_log_loss_v(A, us, vs, k, mask=None):
    if issparse(A):
        return grad_log_loss_v_sparse(
            A.data, A.indptr, A.indices, A.nnz, us, vs, k)
    else:
        d = A.shape[0]
        # energy = np.outer(u, v) - np.outer(v, u)
        energy = get_energy(us, vs)
        u = us[:, k]
        grad_bce = grad_bce_stable(A, energy)
        if mask is None:
            grad_v = (grad_bce.T @ u - grad_bce @ u) / d ** 2
        else:
            grad_v = (
                (grad_bce.T * mask) @ u - (grad_bce * mask) @ u) / mask.sum()
        return grad_v


@njit
def grad_log_loss_v_sparse(X_data, X_indptr, X_indices, nnz, us, vs, k):
    n_features = X_indptr.shape[0] - 1
    grad_v = np.zeros(n_features)
    u = us[:, k]
    for j in range(n_features):
        for i in range(X_indptr[j], X_indptr[j+1]):
            idx_i = X_indices[i]
            energy_ij = get_energy_ij(us, vs, idx_i, j)
            # energy_ij = (u[idx_i] * v[j] - v[idx_i] * u[j])
            grad_v[idx_i] -= grad_bce_stable_ij(X_data[i], energy_ij) * u[j]

    for j in range(n_features):
        for i in range(X_indptr[j], X_indptr[j+1]):
            idx_i = X_indices[i]
            energy_ij = get_energy_ij(us, vs, idx_i, j)
            # energy_ij = (u[idx_i] * v[j] - v[idx_i] * u[j])
            grad_v[j] += grad_bce_stable_ij(X_data[i], energy_ij) * u[idx_i]
    grad_v /= nnz
    return grad_v


def log_loss(A, us, vs, mask=None):
    """Logistic loss, numerically stable implementation.
    Taken from https://fa.bianp.net/blog/2019/evaluate_logistic/

    Parameters
    ----------
    A: payoff matrix in the probability space
    x: concatenation of u and v
    Returns
    -------
    loss: float
    """
    if issparse(A):
        return log_loss_sparse(
            A.data, A.indptr, A.indices, A.nnz, us, vs)
    else:
        energy = get_energy(us, vs)
        # energy = np.outer(u, v) - np.outer(v, u)
        if mask is None:
            return np.mean((1 - A) * energy - logsig(energy))
        else:
            return np.sum(
                ((1 - A) * energy - logsig(energy)) * mask) / mask.sum()


@njit
def log_loss_sparse(X_data, X_indptr, X_indices, nnz, us, vs):
    """Logistic loss, numerically stable implementation.
    Taken from https://fa.bianp.net/blog/2019/evaluate_logistic/

    Parameters
    ----------
    A: payoff matrix in the probability space
    x: concatenation of u and v
    Returns
    -------
    loss: float
    """
    n_features = X_indptr.shape[0] - 1
    loss = 0.
    for j in range(n_features):
        for i in range(X_indptr[j], X_indptr[j+1]):
            idx_i = X_indices[i]
            energy_ij = get_energy_ij(us, vs, idx_i, j)
            # energy_ij = (u[idx_i] * v[j] - v[idx_i] * u[j])
            loss += (1 - X_data[i]) * energy_ij - logsig_ij(energy_ij)
    loss /= nnz
    return loss

@njit
def get_energy_ij(us, vs, idx_i, j):
    n_features, n_components = us.shape
    energy_ij = 0
    for k in range(n_components):
        # TODO optimize this step
        energy_ij += (us[idx_i, k] * vs[j, k] - vs[idx_i, k] * us[j, k])
    return energy_ij




def sigmoid(x):
    return 1./(1 + np.exp(-x))

def sigmoid_stable(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def inv_sigmoid(y):
    return np.log(y / (1 - y))

def logsig(x):
    """Compute the log-sigmoid function component-wise."""
    idx0 = x < -33
    out = np.zeros_like(x)
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out



def grad_bce_stable(b, x):
    """Computes the gradient of the logistic loss.
    sigmoid(x) - b component-wise
    Parameters
    ----------
    b : np.array(n_features, n_features)
        payoff matrix
    x : np.array(n_features, n_features)
        energy

    Returns
    -------
    out: array-like, shape (n_features, n_features)
        gradient
    """
    idx = x < 0
    out = np.zeros(x.shape)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


@njit
def grad_bce_stable_ij(b, x):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    if x <0:
        exp_x = np.exp(x)
        b_idx = b
        return ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    else:
        exp_nx = np.exp(-x)
        b_nidx = b
        return ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)

@njit
def logsig_ij(x):
    """Compute the log-sigmoid function component-wise."""
    idx0 = x < -33
    if x < - 33:
        return x
    if (x >= -33) & (x < -18):
        return x - np.exp(x)
    if (x >= -18) & (x < 37):
        return -np.log1p(np.exp(-x))
    if x >= 37:
        return - np.exp(-x)


def generate_mask(n_features, rng=check_random_state(42)):
    mask = np.zeros((n_features, n_features), dtype=bool)
    p = 0.1
    while np.all(mask.sum(axis=1) == 0):
        mask = rng.choice(
            a=[False, True], size=(n_features, n_features), p=[p, 1-p])
        mask[(mask == 0).T] = 0
    return mask


def generate_mask_sparse(
        payoff, n_features, nnz, rng=check_random_state(42), p=0.2):
    mask = np.zeros((n_features, n_features), dtype=bool)
    while np.all(mask.sum(axis=1) == 0):
        mask = generate_mask_sparse_(payoff.shape[0], nnz, p=p, rng=rng)
        mask[(mask == 0).T] = False
        mask[~nnz] = False
    return mask


def generate_mask_sparse_(n_features, nnz, rng=check_random_state(42), p=0.2):
    mask = np.zeros((n_features, n_features), dtype=bool)
    while np.all(mask.sum(axis=1) == 0):
        mask = rng.choice(
            a=[False, True], size=(n_features, n_features), p=[p, 1-p])
        mask[(mask == 0).T] = False
        mask[~nnz] = False
    return mask


def test_train_split(payoff, p, rng):
    """
    payoff: array shape(n_features, n_features)
        Payoff matrix in the probability space.
    p: float
        Percentage of the
    rng: RandomState instance

    Returns
    -------
    payoff_train: array shape(n_features, n_features)
        Payoff matrix in the probability space, to train the model on.
    payoff_train: array shape(n_features, n_features)
        Payoff matrix in the probability space, to test the model on.
    """
    nnz = (payoff != 0).todense()
    payoff_train = payoff.copy()
    payoff_test = payoff.copy()
    payoff_train = lil_matrix(payoff_train)
    payoff_train[payoff_train!=0] = 0
    payoff_train = payoff_train.tocsc()

    while payoff_train.nnz == 0 or payoff_test.nnz == 0:
        mask = generate_mask_sparse(payoff, payoff.shape[0], nnz, rng=rng, p=p)
        payoff_train = payoff.copy()
        payoff_train = lil_matrix(payoff_train)
        payoff_train[~mask] = 0
        payoff_train = payoff_train.tocsc()

        payoff_test = payoff.copy()
        payoff_test = lil_matrix(payoff_test)
        payoff_test[mask] = 0
        payoff_test = payoff_test.tocsc()

    return payoff_train, payoff_test
