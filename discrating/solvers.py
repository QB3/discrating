import numpy as np
from numpy.linalg import norm

from scipy.sparse import issparse

from tqdm import tqdm
from scipy.optimize import fmin_l_bfgs_b  # minimize
from sklearn.utils import check_random_state

from discrating.utils import (
    log_loss, grad_log_loss_u, grad_log_loss_v,
    squared_sigmoid_loss, grad_squared_sigmoid_loss_u,
    grad_squared_sigmoid_loss_v,
    squared_loss, grad_squared_loss_u, grad_squared_loss_v,
    get_energy, test_train_split,
    squared_loss_sparse, squared_sigmoid_loss_sparse)


def solverCV(
        payoff, rng=check_random_state(42), tol=1e-42, max_alt_min=10,
        maxiter=1_000, loss_name="log", model="extended_elo", verbose=False,
        n_components=1, has_bounds=False, alphas_v=0, n_split=5):
    """
    payoff: array shape(n_features, n_features)
        Payoff matrix in the probability space.
    rng: RandomState instance
    tol: float (default=1e-42)
        Tolerance for the inner optimization problems solved with l-BFGS.
    max_alt_min: int (default=10)
        Maximum number of alternate minimization steps.
    maxiter: int (default=1_000)
        Maximum number of iterations for l-BFGS.
    loss_name: str (default="log")
        Name of the loss to use to fit the problem: "log" refers to the binary cross entropy, "squared" to the squared norm in the probability space.
    model: str (default="extended_elo")
        Model to use, the usual "elo", or the "extended elo".
    verbose: bool (default=False)
    n_components: int (default=1)
        Number of *pairs* of components you want to compute in the decomposition.
    has_bounds: bool (default=False)
        Wether or not to add bounds for l-BFGS.
    alphas_v: array shape(default=0)
        Regularization strengths on the component v.
    n_split: int (default=5)
        Number of splits used for the cross-validation.

    Returns
    -------
    us: array, shape(n_features, n_components)
        Components u.
    vs: array, shape(n_features, n_components)
        Components v.
    losses: array shape(max_alt_min)
        Values of the loss along training.
    grad_norms: array shape(max_alt_min)
        Values of the gradients along training.
    """

    nnz = (payoff != 0).todense()
    result = np.zeros((n_split, len(alphas_v)))
    for idx_split in range(n_split):
        # TODO double check that mask are different at each iter
        payoff_train, payoff_test = test_train_split(payoff, p=0.15, rng=rng)
        _, results_test = solver_path(
            payoff_train, payoff_test, rng=rng, tol=tol,  max_alt_min=max_alt_min,
            maxiter=maxiter, loss_name=loss_name, model=model, verbose=verbose,
            n_components=n_components, has_bounds=has_bounds, alphas_v=alphas_v)
        result[idx_split, :] = results_test.copy()

    # refiting
    idx_alpha_v = np.argmin(result.mean(axis=0))
    alpha_v = alphas_v[idx_alpha_v]
    return solver(
        payoff, rng=rng, tol=tol, max_alt_min=max_alt_min,
        maxiter=maxiter, loss_name=loss_name,
        model=model, verbose=verbose, n_components=n_components,
        has_bounds=has_bounds, alpha_v=alpha_v)

def solver_path(
        payoff_train, payoff_val, rng=check_random_state(42), tol=1e-42, max_alt_min=10,
        maxiter=1_000, loss_name="log", model="extended_elo", verbose=False,
        n_components=1, has_bounds=False, alphas_v=0):
    """
    payoff_train: array shape(n_features, n_features)
        Payoff matrix in the probability space to train the model on.
    payoff_test: array shape(n_features, n_features)
        Payoff matrix in the probability space to test the model on.
    rng: RandomState instance
    tol: float (default=1e-42)
        Tolerance for the inner optimization problems solved with l-BFGS.
    max_alt_min: int (default=10)
        Maximum number of alternate minimization steps.
    maxiter: int (default=1_000)
        Maximum number of iterations for l-BFGS.
    loss_name: str (default="log")
        Name of the loss to use to fit the problem: "log" refers to the binary cross entropy, "squared" to the squared norm in the probability space.
    model: str (default="extended_elo")
        Model to use, the usual "elo", or the "extended elo".
    verbose: bool (default=False)
    n_components: int (default=1)
        Number of *pairs* of components you want to compute in the decomposition.
    has_bounds: bool (default=False)
        Wether or not to add bounds for l-BFGS.
    alphas_v: array shape(default=0)
        Regularization strengths on the component v.

    Returns
    -------
    result_test: array shape(len(alphas_v))
        Results on the test set ofr each value of alpha.
    result_train: array shape(len(alphas_v))
        Results on the train set ofr each value of alpha.
    """

    result_train = np.zeros(len(alphas_v))
    result_test = np.zeros(len(alphas_v))
    #TODO use joblib to parallelize this step
    for idx_alpha, alpha_v in enumerate(alphas_v):
        us, vs, losses, grad_norms = solver(
            payoff_train, rng=rng, tol=tol, max_alt_min=max_alt_min,
            maxiter=maxiter, loss_name=loss_name, model=model, verbose=verbose,
            n_components=n_components, has_bounds=has_bounds, alpha_v=alpha_v)
        if loss_name == "squared":
            loss_train = squared_loss_sparse(payoff_train, us, vs)
            loss_val = squared_loss_sparse(payoff_val, us, vs)
        else:
            loss_train = squared_sigmoid_loss_sparse(payoff_train, us, vs)
            loss_val = squared_sigmoid_loss_sparse(payoff_val, us, vs)

        result_train[idx_alpha] = loss_train
        result_test[idx_alpha] = loss_val
    return result_train, result_test

def solver(
        A, mask=None, rng=check_random_state(42), tol=1e-42, max_alt_min=10,
        maxiter=1_000, loss_name="log", model="extended_elo", verbose=False,
        n_components=1, has_bounds=False, alpha_v=0):
    """
    A: array shape(n_features, n_features)
        Payoff matrix in the probability space.
    mask: array shape(n_features, n_features)
        Entries of the payoff matrix you want to hide, in order to evaluate your model on.
    rng: RandomState instance
    tol: float (default=1e-42)
        Tolerance for the inner optimization problems solved with l-BFGS.
    max_alt_min: int (default=10)
        Maximum number of alternate minimization steps.
    maxiter: int (default=1_000)
        Maximum number of iterations for l-BFGS.
    loss_name: str (default="log")
        Name of the loss to use to fit the problem: "log" refers to the binary cross entropy, "squared" to the squared norm in the probability space.
    model: str (default="extended_elo")
        Model to use, the usual "elo", or the "extended elo".
    verbose: bool (default=False)
    n_components: int (default=1)
        Number of *pairs* of components you want to compute in the decomposition.
    has_bounds: bool (default=False)
        Wether or not to add bounds for l-BFGS.
    alpha_v: float (default=0)
        Regularization strength on the component v.

    Returns
    -------
    us: array, shape(n_features, n_components)
        Components u.
    vs: array, shape(n_features, n_components)
        Components v.
    losses: array shape(max_alt_min)
        Values of the loss along training.
    grad_norms: array shape(max_alt_min)
        Values of the gradients along training.
    """
    d = A.shape[0]

    v_elo = np.log(10) / 400
    vs = np.ones((d, n_components)) * np.log(10) / 400
    us = np.zeros((d, n_components))

    losses = []
    grad_norms = []

    if loss_name == "log":
        loss_ = log_loss
        grad_loss_u_ = grad_log_loss_u
        grad_loss_v_ = grad_log_loss_v
    elif loss_name == "squared_sigmoid":
        loss_ = squared_sigmoid_loss
        grad_loss_u_ = grad_squared_sigmoid_loss_u
        grad_loss_v_ = grad_squared_sigmoid_loss_v
    elif loss_name == "squared":
        loss_ = squared_loss
        grad_loss_u_ = grad_squared_loss_u
        grad_loss_v_ = grad_squared_loss_v
    else:
        raise NotImplementedError

    if model == "balduzzi_2018":
        nnz = A != 0
        if issparse(A):
            u0 = A.sum(axis=1).reshape(-1, 1) / nnz.sum()
        else:
            u0 = (A * mask).sum(axis=1).reshape(-1, 1) / (mask.sum())
        v0 = np.ones(A.shape[0]).reshape(-1, 1)

        us, vs, losses, grads = solver(
            A - get_energy(u0, v0), mask=mask, loss_name=loss_name,
            maxiter=maxiter, max_alt_min=max_alt_min,
            n_components=n_components,
            model="extended elo")

        return np.concatenate(
            (u0, us), axis=1), np.concatenate((v0, vs), axis=1), losses, grads


    def sum_square(u, vs):
        # import ipdb; ipdb.set_trace()
        if vs.shape[0] == 0:
            return 0
        if len(vs.shape) > 1:
            res = ((u.T @ vs) ** 2).sum() / (2 * (vs **2).sum())
        return res

    def grad_sum_square(u, vs):
        # import ipdb; ipdb.set_trace()
        if len(vs.shape) > 1:
            res = ((u.T @ vs) * vs / (vs ** 2).sum(axis=0)).sum(axis=1)
        else:
            if vs.shape[0] == 0:
                return 0
            res = ((u.T @ vs) * vs / (vs ** 2).sum()).sum()
        return res


    for k in range(n_components):
        pbar = tqdm(range(max_alt_min))
        for t in pbar:
            if model == "elo" and k == 1:
                def val_u_(u):
                    us_copy = us.copy()
                    us_copy[:, k] = u.copy()
                    res = loss_(A, us_copy, vs, mask=mask)
                    # res += sum_square(u, vs[:, :(k+1)])
                    # if k!=0:
                    #     res += sum_square(u, us_copy[:, :k])
                    return res
                def grad_u_(u):
                    us_copy = us.copy()
                    us_copy[:, k] = u.copy()
                    res = grad_loss_u_(A, us_copy, vs, k, mask=mask)
                    # res += grad_sum_square(u, vs[:, :(k+1)])
                    # if k!=0:
                    #     res += grad_sum_square(u, us_copy[:, :k])
                    return res
            else:
                def val_u_(u):
                    us_copy = us.copy()
                    us_copy[:, k] = u.copy()
                    res = loss_(A, us_copy, vs, mask=mask)
                    res += sum_square(u, vs[:, :(k+1)])
                    if k!=0:
                        res += sum_square(u, us_copy[:, :k])
                    # res += alpha * u ** 2 / 2
                    return res
                def grad_u_(u):
                    us_copy = us.copy()
                    us_copy[:, k] = u.copy()
                    res = grad_loss_u_(A, us_copy, vs, k, mask=mask)
                    res += grad_sum_square(u, vs[:, :(k+1)])
                    if k!=0:
                        res += grad_sum_square(u, us_copy[:, :k])
                    # res += alpha * u
                    return res

            us[:, k] = minimize_u(
                A, us[:, k].copy(), val_u_, grad_u_, mask=mask, maxiter=maxiter,
                tol=1e-42, verbose=verbose, bounds=None)[0]

            if model == "elo" and k == 0:
                break

            def val_v_(v):
                vs_copy = vs.copy()
                vs_copy[:, k] = v.copy()
                res = loss_(A, us, vs_copy, mask=mask)
                res += sum_square(v, us[:, :(k+1)])
                if k!=0:
                    res += sum_square(v, vs_copy[:, :k])
                res += alpha_v / 2 * ((v - v_elo) ** 2).sum()
                return res

            def grad_v_(v):
                # TODO update penalties
                vs_copy = vs.copy()
                vs_copy[:, k] = v.copy()
                res = grad_loss_v_(A, us, vs_copy, k, mask=mask)
                res += grad_sum_square(v, us[:, :(k+1)])
                if k != 0:
                    res += grad_sum_square(v, vs_copy[:, :k])
                res += alpha_v * (v - v_elo)
                return res

            if has_bounds:
                bounds = [(0, np.infty) for k in range(d)]
            else:
                bounds = None
            vs[:, k] = minimize_u(
                A, vs[:, k].copy(), val_v_, grad_v_, mask=mask, maxiter=maxiter,
                tol=1e-42, verbose=verbose, bounds=bounds)[0]

            loss = loss_(A, us, vs, mask=mask)
            losses.append(loss)

            pbar.set_description(
                f"Iteration: {t+1}, loss: {loss}")

    return us, vs, losses, grad_norms



def minimize_u(
        A, u, val_u_, grad_u_, mask=None, maxiter=1000,
        tol=1e-15, verbose=False, bounds=None):
    """
    A: array shape(n_features, n_features)
        Payoff matrix in the probability space.
    u: array shape(n_features,)
        Values for the first component.
    val_u: function
        Function which takes in argument u, and return the value.
    grad_u: function
        Function which takes in argument u, and return the gradient.
    mask: array shape(n_features, n_features)
        Entries of the payoff matrix you want to hide, in order to evaluate your model on.
    maxiter: int (default=1_000)
        Maximum number of iterations for l-BFGS.
    tol: float (default=1e-42)
        Tolerance for the inner optimization problems solved with l-BFGS.
    verbose: bool (default=False)
    bounds:
        List of box bounds for l-BFGS.

    Returns
    -------
    us: array shape(n_features)
        Minimizer of the function val_u.
    losses: array shape(max_alt_min)
        Values of the loss along training.
    grad_norms: array shape(max_alt_min)
        Values of the gradients along training.
    """
    losses = []
    grad_norms = []
    def callback(u):
        loss = val_u_(u)
        losses.append(loss)
        grad_u = grad_u_(u)
        grad_norm = norm(grad_u)
        grad_norms.append(grad_norm)
        if verbose:
            print(grad_norm)

    result_u = fmin_l_bfgs_b(
        val_u_, u.copy(), fprime=grad_u_, pgtol=tol,
        callback=callback, maxls=maxiter, maxiter=maxiter,
        maxfun=maxiter, factr=1e-42, bounds=bounds)
    if verbose:
        print(result_u[2]['task'])
    return result_u[0], losses, grad_norms
