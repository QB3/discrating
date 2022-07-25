import numpy as np
import scipy
import pickle

from disc_rating.solvers import solver

n_games = 1_000_000

min_games_matchups = [40, 50, 60, 70, 80]
months = ["2019-08", "2020-05"]
min_opponents = 1


dict_uvd = {}

for idx, min_games_matchup in enumerate(min_games_matchups):
    payoff = scipy.sparse.load_npz(
        "../../data/lichess/lichess_payoff_float_%i_%i_%s_%s.npz" % (
        n_games, min_games_matchup, months[0], months[-1]))

    filter = np.array((payoff != 0).sum(axis=0) >= min_opponents)[0]
    payoff = payoff[np.ix_(filter, filter)]

    max_alt_min = 500
    maxiter = 10_000

    d = payoff.shape[0]
    for loss_name in ["squared", "log"]:
        u, v, _,_ = solver(
            payoff, loss_name=loss_name, max_alt_min=max_alt_min,
            maxiter=maxiter, verbose=False)

        dict_uvd[loss_name, idx] = (u.copy(), v.copy(), d)

with open('results/lichess_visu_dict_uv.pkl', 'wb') as f:
    pickle.dump(dict_uvd, f)
