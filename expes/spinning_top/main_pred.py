from itertools import product
import pickle
import numpy as np

from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from disc_rating.solvers import solver, squared_sigmoid_loss
from disc_rating.utils import generate_mask


# download data at https://proceedings.neurips.cc/paper/2020/hash/ca172e964907a97d5ebd876bfdd4adbd-Abstract.html
with open('../../data/spinning_top_payoffs.pkl', 'rb') as f:
    payoffs = pickle.load(f)

keys = [
'10,3-Blotto',
 '10,4-Blotto',
 '10,5-Blotto',
 '3-move parity game 2',
 '5,3-Blotto',
 '5,4-Blotto',
 '5,5-Blotto',
 'AlphaStar',
 'Blotto',
 'Kuhn-poker',
 'Random game of skill',
 'connect_four',
 'go(board_size=3,komi=6.5)',
 'go(board_size=4,komi=6.5)',
 'hex(board_size=3)',
 'misere(game=tic_tac_toe())',
 'quoridor(board_size=3)',
 'quoridor(board_size=4)',
 'tic_tac_toe',
]


all_key = []
all_size = []
for key in keys:
    all_size.append(payoffs[key].shape[0])
    all_key.append(key)

all_size = np.array(all_size)
all_key = np.array(all_key)
idx = np.argsort(all_size)
games_names = all_key[idx]

dict_masks = {}

params = [
    ("log", "elo", 1),
    ("log", "extended_elo", 1),
    ("squared", "balduzzi_2018", 1),
    ("log", "extended_elo", 2),
    ("squared", "balduzzi_2018", 2),
    ("squared", "balduzzi_2018", 3),
    ("log", "extended_elo", 3),
    ]
rng = check_random_state(42)

prototype = False

if prototype:
    res_dir = 'results_proto'
else:
    res_dir = 'results'


for game_name in games_names:
    A = payoffs[game_name]
    if prototype:
        idx = np.arange(min(100, A.shape[0]))
        A = A[np.ix_(idx, idx)]
    mask = generate_mask(A.shape[0], rng)
    dict_masks[game_name] = mask.copy()
    assert mask.shape[0] == A.shape[0]

fig_name = "spinning_top"
with open('%s/%s_mask.pkl' % (res_dir, fig_name), 'wb') as f:
    pickle.dump(dict_masks, f)

def parallel_function(game_name, loss_name, model, n_components):
    mask = dict_masks[game_name]
    A = payoffs[game_name]
    if prototype:
        idx = np.arange(min(100, A.shape[0]))
        A = A[np.ix_(idx, idx)]
    assert mask.shape[0] == A.shape[0]
    us, vs, _, _ = solver(
        (A + 1) / 2, mask=mask, loss_name=loss_name, model=model,
        maxiter=1000, n_components=n_components, max_alt_min=100)
    dict_results = {}
    dict_results[game_name, loss_name, model, n_components] = (
        us.copy(), vs.copy())

    fig_name = game_name.replace(" ", "_") + loss_name + model + str(n_components)
    with open('%s/%s_dict_uv_pred.pkl' % (res_dir, fig_name), 'wb') as f:
        pickle.dump(dict_results, f)


print("enter parallel")
backend = 'loky'
n_jobs = 4
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(parallel_function)(game_name, loss_name, model, n_components)
    for game_name, (loss_name, model, n_components) in product(
        games_names, params))
