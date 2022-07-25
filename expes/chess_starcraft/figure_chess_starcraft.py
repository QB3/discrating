import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from disc_rating.utils_plot import rotate, plot_simplex_projection


fontsize = 30
fontsize_ticks = 20

chess = False
# chess = True
if chess:
    n_games = 1_000_000
    months = ["2019-08", "2020-05"]
    min_games_matchups = [40, 50, 60, 70, 80]

    with open('results/lichess_visu_dict_uv.pkl', 'rb') as f:
        dict_uvd = pickle.load(f)
else:
    with open('results/starcraft_visu_dict_uv.pkl', 'rb') as f:
        dict_uvd = pickle.load(f)
    min_games_matchups = [80, 100, 150, 175, 200]

if chess:
    dict_xticks = {}

    dict_xticks[1, 0] = [0, 2]
    dict_xticks[1, 1] = [0, 2]
    dict_xticks[1, 2] = [0, 2]
    dict_xticks[1, 3] = [0, 2]
    dict_xticks[1, 4] = [0, 2]

    dict_xticks[2, 0] = [-10, 0]
    dict_xticks[2, 1] = [0, 5]
    dict_xticks[2, 2] = [0, 3]
    dict_xticks[2, 3] = [0, 2]
    dict_xticks[2, 4] = [0, 3]

    dict_yticks = {}

    dict_yticks[1, 0] = [0, 4]
    dict_yticks[1, 1] = [0, 4]
    dict_yticks[1, 2] = [0, 2]
    dict_yticks[1, 3] = [0, 2]
    dict_yticks[1, 4] = [0, 2]

    dict_yticks[2, 0] = [-3, 0]
    dict_yticks[2, 1] = [0, 5]
    dict_yticks[2, 2] = [0, 3]
    dict_yticks[2, 3] = [0, 2]
    dict_yticks[2, 4] = [0, 3]
else:
    dict_xticks = {}

    dict_xticks[1, 0] = [0, 2]
    dict_xticks[1, 1] = [0, 2]
    dict_xticks[1, 2] = [0, 2]
    dict_xticks[1, 3] = [0, 2]
    dict_xticks[1, 4] = [0, 2]

    dict_xticks[2, 0] = [0, 2]
    dict_xticks[2, 1] = [0, 5]
    dict_xticks[2, 2] = [0, 3]
    dict_xticks[2, 3] = [0, 2]
    dict_xticks[2, 4] = [0, 3]

    dict_yticks = {}

    dict_yticks[1, 0] = [0, 4]
    dict_yticks[1, 1] = [0, 4]
    dict_yticks[1, 2] = [0, 2]
    dict_yticks[1, 3] = [0, 2]
    dict_yticks[1, 4] = [0, 2]

    dict_yticks[2, 0] = [-3, 0]
    dict_yticks[2, 1] = [0, 5]
    dict_yticks[2, 2] = [0, 3]
    dict_yticks[2, 3] = [0, 2]
    dict_yticks[2, 4] = [0, 3]


fig, axarr = plt.subplots(
    3, len(min_games_matchups), figsize=[12, 6])
for idx, min_games_matchup in enumerate(min_games_matchups):
    if chess:
        payoff = scipy.sparse.load_npz(
            "../../data/lichess/lichess_payoff_float_%i_%i_%s_%s.npz" % (
            n_games, min_games_matchup, months[0], months[-1]))
    else:
        payoff = scipy.sparse.load_npz(
            "../../data/starcraft/starcraft_payoff_float_%i.npz" %
            min_games_matchup)
    axarr[0, idx].spy(payoff, markersize=1)
    axarr[0, idx].set_xticks([])
    axarr[0, idx].set_yticks([])
    axarr[0, idx].set_title(min_games_matchup, fontsize=fontsize)
    for idx_loss, loss_name in enumerate(["squared", "log"]):
        if chess:
            u, v, d = dict_uvd[loss_name, idx]
        else:
            u, v, d = dict_uvd[loss_name, min_games_matchup]
        u_plot, v_plot = rotate(u / np.linalg.norm(u), v / np.linalg.norm(v))
        plot_simplex_projection(
            axarr[1 + idx_loss, idx], d, u_plot + 1e-8 * np.random.randn(d),v_plot, s=40)
    axarr[2, idx].set_xlabel("$\mathbf{v}$", fontsize=fontsize)

    axarr[1, idx].set_xticks(dict_xticks[1, idx])
    axarr[2, idx].set_xticks(dict_xticks[2, idx])

    axarr[1, idx].set_yticks(dict_yticks[1, idx])
    axarr[2, idx].set_yticks(dict_yticks[2, idx])

    axarr[1, idx].tick_params(labelsize=fontsize_ticks)
    axarr[2, idx].tick_params(labelsize=fontsize_ticks)

axarr[0, 0].set_ylabel("payoff", fontsize=fontsize)
axarr[1, 0].set_ylabel('proba. \n $\mathbf{u}$', fontsize=fontsize)
axarr[2, 0].set_ylabel("logit \n $\mathbf{u}$", fontsize=fontsize)

plt.tight_layout()
plt.show(block=False)
