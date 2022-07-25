import numpy as np
import pickle
import scipy
from scipy import sparse

from disc_rating.solvers import solver

min_games_matchups = [80, 100, 150, 175, 200]

min_opponents = 1
dict_uvd = {}

for min_games_matchup in min_games_matchups:
    payoff = scipy.sparse.load_npz(
        "../../data/starcraft/starcraft_payoff_float_%i.npz" %
        min_games_matchup)

    filter = np.array((payoff != 0).sum(axis=0) >= min_opponents)[0]
    payoff = payoff[np.ix_(filter, filter)]

    max_alt_min = 100
    maxiter = 10_000

    d = payoff.shape[0]
    for loss_name in ["squared", "log"]:
        u, v, _,_ = solver(
            payoff, loss_name=loss_name, max_alt_min=max_alt_min,
            maxiter=maxiter, verbose=False)

        dict_uvd[loss_name, min_games_matchup] = (u.copy(), v.copy(), d)

with open('results/starcraft_visu_dict_uv.pkl', 'wb') as f:
    pickle.dump(dict_uvd, f)
