import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import pickle

# fig_name = "elo_disc"
# fig_name = "all_others"
fig_dir = "../../../disc_rating/NeurIPS2021/figures/"

prototype = True

if prototype:
    res_dir = 'results_proto'
else:
    res_dir = 'results'

with open('%s/all_dict_final_pred.pkl' % res_dir, 'rb') as f:
    elos = pickle.load(f)


columns = "loss", "method", "dim", "dataset", "split", "value"
rdf = pd.DataFrame([dict(zip(columns, row + tuple([elos[row]]))) for row in elos])
df = rdf



df['method_name'] = df.apply(lambda x: x['loss'] + "_" + x['method']  + "_" + str(x['dim']), axis=1)

df['label'] = 'Elo'
df.loc[df['method_name'] == 'log_elo_1', 'label'] = 'Elo'
df.loc[df['method_name'] == 'squared_sigmoid_elo_1', 'label'] = 'Elo ++'
df.loc[df['method_name'] == 'log_extended_elo_1', 'label'] = 'This work (2)'
df.loc[df['method_name'] == 'log_extended_elo_2', 'label'] = 'This work (4)'
df.loc[df['method_name'] == 'log_extended_elo_1', 'label'] = 'm-Elo'
# df.loc[df['method_name'] == 'log_elo_1', 'label'] = 'Elo'


dict_label = {}
dict_label['log_elo_1'] = 'Elo (1)'
dict_label['squared_sigmoid_elo_1'] = 'Elo ++ (1)'
dict_label['log_extended_elo_1'] = 'This work (2)'
dict_label['log_extended_elo_2'] =  'This work (4)'
dict_label['squared_extended_elo_1'] = 'This work (logit, 2)'
dict_label['squared_extended_elo_2'] =  'This work (logit, 4)'
dict_label['log_extended_elo_3'] =  'This work (6)'
dict_label['squared_balduzzi_2018_1'] = 'm-Elo (3)'
dict_label['squared_balduzzi_2018_2'] = 'm-Elo (5)'
dict_label['squared_balduzzi_2018_3'] = 'm-Elo (7)'


methods = [
    'log_elo_1',
    # 'squared_sigmoid_elo_1',
    'log_extended_elo_1',
    'squared_balduzzi_2018_1',
    'log_extended_elo_2',
    'squared_balduzzi_2018_2',
    'log_extended_elo_3',
    'squared_balduzzi_2018_3',
 ]


list_dataset_label = [
    '10,3-Blotto',
    '10,4-Blotto',
    '10,5-Blotto',
    '3-move parity',
    '5,3-Blotto',
    '5,4-Blotto',
    '5,5-Blotto',
    'AlphaStar',
    'Blotto',
    'Kuhn-poker',
    'Bernoulli',
    # 'Normal Bernoulli game',
    # 'RPS',
    'Random game',
    'connect_four',
    'go(size=3)',
    'go(size=4)',
    'hex(size=3)',
    'misere(tic_tac_toe)',
    'quoridor(size=3)',
    'quoridor(size=4)',
    'tic tac toe']


s_elo = df['method_name'] == 'log_elo_1'
tmp_elo = df[s_elo]
tmp_elo = tmp_elo[tmp_elo.split == 'test']
datasets_elo = tmp_elo['dataset']

fontsize = 20
fontsize_legend = 20
fontsize_x = 18
fontsize_y = 20


plt.figure(figsize=(20, 5))
# plt.figure(figsize=(len(set(df['dataset'])), 5))
sns.set_style('ticks')
# for mid, method_id in enumerate(set(df['method_name'])):
for mid, method_id in enumerate(methods):
  s = df['method_name'] == method_id
  print(method_id)
  tmp = df[s]
  tmp = tmp[tmp.split == 'test']
  datasets = tmp['dataset']
  print(mid)
  print(df['label'][mid])
  plt.bar(
    # np.array(range(len(tmp['dataset']))) + mid,
    np.array(range(len(tmp['dataset']))) + mid * 0.15,
    -np.log10(np.array(tmp['value']) / np.array(tmp_elo['value']) ),
    0.15,
    label=dict_label[method_id])
plt.ylabel(
    '$-\\log_{10}(\\mathrm{MSE}_\\mathrm{test} / \\mathrm{MSE}_\\mathrm{test}^{\\mathrm{Elo}})$', fontsize=fontsize_y)
# plt.legend()
plt.gca().legend(
    loc='center left',
    bbox_to_anchor=(0.92, -0.17),
    fontsize=fontsize_legend)
# plt.gca().legend(
#     loc='center left', bbox_to_anchor=(0.92, -0.3), fontsize=fontsize_legend)
sns.despine(offset=5)
plt.xticks(
    [a+0.25 for a in range(len(datasets))], list_dataset_label, rotation=45, fontsize=fontsize_x)
plt.yticks([0, 0.5], fontsize=fontsize_y)
# plt.xticks([a+0.25 for a in range(len(datasets))], list(datasets), rotation=45, fontsize=fontsize_x)
# plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
save_fig = False
# save_fig = True
if save_fig:
    plt.savefig(
        fig_dir + "prediction_all_payoff_spinning_top.pdf", bbox_inches="tight")
plt.show(block=False)
#   plt.bar(

#   )
# df.groupby(['loss', 'method', 'dim']).value.apply(np.log).plot(kind='bar')
