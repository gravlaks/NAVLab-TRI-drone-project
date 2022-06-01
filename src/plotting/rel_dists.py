import matplotlib.pyplot as plt

def plot_relative_distances(rel_dists, filepath):

    fig = plt.figure()
    plt.plot(rel_dists[:, 0], rel_dists[:, 1])
    plt.savefig(filepath, dpi=fig.dpi)
    plt.show()

def plot_locations(pos1, pos2):
    plt.plot(pos1[:, 0], pos1[:, 1])
    plt.plot(pos2[:, 0], pos2[:, 1])
    plt.show()