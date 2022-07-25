import numpy as np
import matplotlib.pyplot as plt
from discrating.solvers import solver
from discrating.utils_plot import configure_plt
tmp = {}
dict_uv = {}
for a in np.linspace(0.51, 0.75, 10):
  for b in np.linspace(0.51, 1, 10):
    # create the probability matrix from Example 3
    payoff = np.ones((3, 3)) / 2
    payoff[0, 1] = a
    payoff[0, 2] = a
    payoff[1, 0] = 1-a
    payoff[2, 0] = 1-a
    payoff[1, 2] = b
    payoff[2, 1] = 1-b
    # compute the disc rating
    u, v, _, _ = solver(payoff, model="extended elo", loss_name="log")
    dict_uv[a, b] = (u.copy(), v.copy())
    # compute the quantity predicting the outcome of the confrontation
    result = u[0] * v[1] - v[0] * u[1]
    print("%.2f %.2f : %5.2f" % (a,b, result))
    tmp[a,b] = result

# Plotting
xs=[]
ys=[]
vs=[]
for x,y in tmp:
  xs.append(x)
  ys.append(y)
  vs.append(tmp[x,y])

xs = np.array(xs)
ys = np.array(ys)
vs = np.array(vs)


configure_plt()
scaling = 1000


fontsize = 50
plt.figure(figsize=(10,10))
scatter = plt.scatter(xs, ys, s=np.abs(vs)*scaling, c='darkgreen')
plt.title("Disc rating works", fontsize=fontsize)
plt.xticks([0.5, 0.75], fontsize=fontsize)
plt.yticks([0.5, 1], fontsize=fontsize)
plt.xlabel('$\gamma$', fontsize=fontsize)
plt.ylabel('$\delta$', fontsize=fontsize)
plt.tight_layout()
plt.show()
