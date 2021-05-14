from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import utils

def dist_pardo(i, j, label_i, label_j, dist_fn):
    return i, j, (dist_fn(label_i, label_j) if i > j else np.nan)

def get_distance_matrix_singlethreaded(all_labels, dist_fn):
    label_distances = [[dist_fn(label_a, label_b) if i < j else np.nan
                        for i, label_a in enumerate(all_labels)]
                        for j, label_b in enumerate(all_labels)]
    return np.array(label_distances)

def get_distance_matrix(all_labels, dist_fn):
    result = np.zeros((len(all_labels), len(all_labels)))

    labelrange = range(len(all_labels))
    args = utils.flatten([[(a, b, all_labels[a], all_labels[b], dist_fn) for a in labelrange] for b in labelrange])

    with Pool() as p:
        reduced = p.starmap(dist_pardo, args)

    for item in reduced:
        result[item[0], item[1]] = item[2]
    return result

class InterAnnotatorAgreement():
    @classmethod
    def create_from_experiment(cls, experiment, distance_fn=None):
        if distance_fn is None:
            distance_fn = experiment.distance_fn
        return cls(experiment.annodf, experiment.item_colname, experiment.label_colname, distance_fn)

    def __init__(self, annodf, item_colname, label_colname, distance_fn):
        self.distance_fn = distance_fn
        self.annodf = annodf.sort_values(item_colname)
        self.items = self.annodf[item_colname].values
        self.all_labels = self.annodf[label_colname].values
    
    def setup(self):
        self.distance_matrix = get_distance_matrix(self.all_labels, self.distance_fn)
        
        same_item = np.array([[np.nan if item_a != item_b else 1 for item_a in self.items] for item_b in self.items])
        different_item = np.array([[np.nan if item_a == item_b else 1 for item_a in self.items] for item_b in self.items])
        self.observed_distances = same_item * self.distance_matrix
        self.expected_distances = different_item * self.distance_matrix
        self.observed_distances = self.observed_distances[~np.isnan(self.observed_distances)]
        self.expected_distances = self.expected_distances[~np.isnan(self.expected_distances)]

    def plot_matrix(self, labels=None, figsize=8, title=None, show_grid=True):
        fix, ax = plt.subplots(figsize=(figsize, figsize))
        if labels is not None:
            for i, label in enumerate(self.all_labels):
                plt.annotate(label, (i, 0.15 + i), color="k")
        plt.imshow(self.distance_matrix, vmin=-0.1, vmax=1.5, cmap="plasma")
        ax = plt.gca()
        if show_grid:
            minor_grids = np.arange(-.5, self.distance_matrix.shape[0], 1)
            ax.set_xticks(minor_grids, minor=True)
            ax.set_yticks(minor_grids, minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_distance_distributions(self, title=None):
        fig, ax = plt.subplots()
        DoH = ax.hist(self.observed_distances, color="b", alpha=0.5)
        DoM, = ax.plot([self.observed_distances.mean(), self.observed_distances.mean()], [0, ax.get_ylim()[1]], "b:")
        ax2 = ax.twinx()
        DeH = ax2.hist(self.expected_distances, color="r", alpha=0.5)
        DeM, = ax2.plot([self.expected_distances.mean(), self.expected_distances.mean()], [0, ax2.get_ylim()[1]], "r:")
        plt.legend([DoM, DeM], ["observed distance", "expected distance"])
        if title is not None:
            plt.title(title)
        plt.show()

    def get_krippendorff_alpha(self):
        return 1 - self.observed_distances.mean() / self.expected_distances.mean()