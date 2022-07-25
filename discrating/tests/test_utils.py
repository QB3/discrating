import pytest
import numpy as np
from scipy.optimize import check_grad
from scipy.sparse import csc_matrix, issparse

from discrating.utils import (
    log_loss, grad_log_loss_u, grad_log_loss_v,
    squared_sigmoid_loss, grad_squared_sigmoid_loss_u, log_loss_sparse,
    grad_squared_sigmoid_loss_v, sigmoid_stable, grad_log_loss_u_sparse,
    grad_log_loss_v_sparse, squared_loss, grad_squared_loss_u,
    grad_squared_loss_v)
from sklearn.utils import check_random_state

d = 10
rng = check_random_state(42)
A = rng.randn(d, d)
A = sigmoid_stable(A)
A_csc = csc_matrix(A)

mask = np.ones(A.shape, dtype=bool)
mask[-5:d, 0] = False
mask[0, -5:d] = False

@pytest.mark.parametrize('A', [A, A_csc])
@pytest.mark.parametrize('variable', ["u", "v"])
@pytest.mark.parametrize('loss_name', [
    "log_loss",
    # "squared_sigmoid_loss",
    "squared"])
@pytest.mark.parametrize('mask', [None, mask])
def test_gradients(A, variable, loss_name, mask):
    if loss_name == "log_loss":
        loss_fun = log_loss
        grad_u_fun = grad_log_loss_u
        grad_v_fun = grad_log_loss_v
    elif loss_name == "squared_sigmoid_loss":
        loss_fun = squared_sigmoid_loss
        grad_u_fun = grad_squared_sigmoid_loss_u
        grad_v_fun = grad_squared_sigmoid_loss_v
    else:
        loss_fun = squared_loss
        grad_u_fun = grad_squared_loss_u
        grad_v_fun = grad_squared_loss_v

    if mask is not None and issparse(A) and loss_name == "squared":
        pytest.xfail("not implemented")

    vs = sigmoid_stable(rng.randn(d, 2))
    us = sigmoid_stable(rng.randn(d, 2))
    k = 1
    def val_u(u):
        us[:, k] = u.copy()
        return loss_fun(A, us, vs, mask=mask)

    def grad_u(u):
        us[:, k] = u.copy()
        return grad_u_fun(A, us, vs, k, mask=mask)

    def val_v(v):
        vs[:, k] = v.copy()
        return loss_fun(A, us, vs, mask=mask)

    def grad_v(v):
        vs[:, k] = v.copy()
        return grad_v_fun(A, us, vs, k, mask=mask)

    for _ in range(20):
        if variable == "u":
            u = sigmoid_stable(rng.randn(d))
            grad_error = check_grad(val_u, grad_u, u)
        else:
            v = sigmoid_stable(rng.randn(d))
            grad_error = check_grad(val_v, grad_v, v)
        np.testing.assert_array_less(grad_error, 1e-5)


def test_sparse():
    vs = sigmoid_stable(rng.randn(d, 2))
    us = sigmoid_stable(rng.randn(d, 2))
    k= 1
    loss = log_loss(A, us, vs)
    loss_sparse = log_loss_sparse(
        A_csc.data, A_csc.indptr, A_csc.indices, A_csc.nnz, us, vs)
    np.testing.assert_allclose(loss, loss_sparse)

    grad_u = grad_log_loss_u(A, us, vs, k)
    grad_sparse_u = grad_log_loss_u_sparse(
        A_csc.data, A_csc.indptr, A_csc.indices, A_csc.nnz, us, vs, k)

    np.testing.assert_allclose(grad_u, grad_sparse_u)

    grad_v = grad_log_loss_v(A, us, vs, k)
    grad_sparse_v = grad_log_loss_v_sparse(
        A_csc.data, A_csc.indptr, A_csc.indices, A_csc.nnz, us, vs, k)
    np.testing.assert_allclose(grad_v, grad_sparse_v)

if __name__ == "__main__":
    test_sparse()
