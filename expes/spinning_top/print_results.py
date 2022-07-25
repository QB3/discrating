import numpy as np
import sys
import pickle

from disc_rating.utils import squared_loss, squared_sigmoid_loss

with open('../../data/spinning_top_payoffs.pkl', 'rb') as f:
    payoffs = pickle.load(f)

prototype = False
# prototype = True

if prototype:
    res_dir = 'results_proto'
else:
    res_dir = 'results'


# fig_name = "elo_disc"
fig_name = "spinning_top"
with open('%s/%s_mask.pkl' % (res_dir, fig_name), 'rb') as f:
    dict_mask = pickle.load(f)



games_names = [
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
#  'Normal Bernoulli game',
#  'RPS',
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

losse_names_tab = {}
losse_names_tab["log"] = "bce"
losse_names_tab["squared"] = "mse"
losse_names_tab["squared_sigmoid"] = "mse sigmoid"

model_names = {}
model_names["elo"] = "Elo"
model_names["extended_elo"] = "Ours"

number_params = {}
number_params["elo", 1] = 1
number_params["extended_elo", 1] = 2
number_params["balduzzi_2018", 1] = 3

number_params["extended_elo", 2] = 4
number_params["balduzzi_2018", 1] = 3
number_params["balduzzi_2018", 2] = 5
number_params["balduzzi_2018", 3] = 7
number_params["extended_elo", 3] = 6

model_names_table = {}
model_names_table["elo", "log"] = "Elo"
model_names_table["elo", "squared_sigmoid"] = "Elo ++"
model_names_table["elo", "squared"] = r"\citet{Balduzzi2019}"
model_names_table["extended_elo", "log"] = "This work"
model_names_table["extended_elo", "squared_sigmoid"] = "Ours (mse)"
model_names_table["extended_elo", "squared"] = r"\citet{Balduzzi2019}"
model_names_table["balduzzi_2018", "squared"] = r"mElo \citep{Balduzzi2018}"

params = [
    ("log", "elo", 1),
    # ("squared_sigmoid", "elo", 1),
    ("log", "extended_elo", 1),
    # ("squared", "extended_elo", 1),
    ("squared", "balduzzi_2018", 1),
    ("squared", "balduzzi_2018", 2),
    # ("squared", "extended_elo", 2),
    ("log", "extended_elo", 2),
    ("log", "extended_elo", 3),
    ("squared", "balduzzi_2018", 3),
]


dict_results = {}

# sys.stdout = open("results_%s.txt" % fig_name, "w")
for (loss_name, model, n_components) in params:
    if model == "extended_elo" and loss_name == "log":
        print(r"\rowcolor{green!20}")
    print("%s  & %i &" % (
        model_names_table[model, loss_name],
        number_params[model, n_components]))
    for i, game_name in enumerate(games_names):
        mask = dict_mask[game_name].copy()
        fig_name = game_name.replace(" ", "_") + loss_name + model + str(n_components)
        with open('%s/%s_dict_uv_pred.pkl' % (res_dir, fig_name), 'rb') as f:
            dict_uv = pickle.load(f)

        u, v = dict_uv[game_name, loss_name, model, n_components]
        A = payoffs[game_name].copy()
        if prototype:
            idx = np.arange(min(100, A.shape[0]))
            A = A[np.ix_(idx, idx)]
        A = (A + 1) / 2
        if loss_name == "squared":
            loss_train = squared_loss(A, u, v, mask)
            loss_test = squared_loss(A, u, v, ~mask)
        else:
            loss_train = squared_sigmoid_loss(A, u, v, mask)
            loss_test = squared_sigmoid_loss(A, u, v, ~mask)

        dict_results[loss_name, model, n_components, game_name, 'train'] = loss_train
        dict_results[loss_name, model, n_components, game_name, 'test'] = loss_test

        if i != (len(games_names) - 1):
            print(r"\num{%.2e} & \num{%.2e} &" % (loss_train, loss_test))
        else:
            print(r"\num{%.2e} & \num{%.2e}" % (loss_train, loss_test))
    print(r"\\")


with open('%s/all_dict_final_pred_%s.pkl' % (res_dir, prototype), 'wb') as f:
    pickle.dump(dict_results, f)
