import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from scipy.spatial import ConvexHull


def configure_plt(fontsize=10, poster=True):
    """Configure matplotlib with TeX and seaborn."""
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    usetex = matplotlib.checkdep_usetex(True)
    params = {'axes.labelsize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 2,
              'ytick.labelsize': fontsize - 2,
              'text.usetex': usetex,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette('colorblind')
    sns.set_style("ticks")
    if poster:
        sns.set_context("poster")


def get_main_component(A):
    # maybe to optimize
    _, v = np.linalg.eigh(A @ A)
    # _, v = np.linalg.eigh(A.T @ A)
    return v

def rotate(u,v):
    d = u.shape[0]
    one = np.ones(d)
    theta = np.arctan(np.sum(v*one)/np.sum(u*one))
    u_theta = np.cos(theta)*u + np.sin(theta)*v
    v_theta = -np.sin(theta)*u + np.cos(theta)*v
    if np.sum(u_theta * one) < 0:
        u_theta = - u_theta
    return u_theta, v_theta


def projection_plane(x,u,v):
    return (np.sum(x * u), np.sum(x*v))


def plot_simplex_projection(ax, d, u, v, s=40):
    l_x = []
    l_y = []
    points = np.zeros((d,2))
    for i in range(d):
        x = np.zeros(d)
        x[i] = 1
        points[i,:] = np.array(projection_plane(x,u,v)) * np.sqrt(d)
        l_x.append(points[i,0])
        l_y.append(points[i,1])
    hull = ConvexHull(points)

    sns.scatterplot(x=l_x,y=l_y,alpha=.7, ax=ax)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-')

    ax.scatter([0],[0],marker ="+", s=s, color="b")

    ax.grid()




if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    from discrating.utils import inv_sigmoid
    with open('../data/spinning_top_payoffs_modified.pkl', 'rb') as f:
        payoffs = pickle.load(f)
    game_name = 'Disc game'
    P = payoffs[game_name]

    A = inv_sigmoid((P + 1) / 2)
    v = get_main_component(P)

    plt.imshow(A)
    plt.show()
