# Name: Tarun Saxena & Anson Antony
# CS 7180 Advanced Perception
# Date:- 7 December, 2023

# vis_intensity_distribution.py:- defines a function (plt_intensity_distribution) using seaborn and matplotlib to visualize and compare the intensity distributions of multiple datasets, and then displays or saves the plot.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(font_scale=1.5)
sns.set(context="paper", style="white")


def plt_intensity_distribution(list_datasets, labels=None, colors=['tab:blue', 'tab:green'], title='Intensity distribution', save_path=None):
    '''

    :param list_datasets: list of ND array [N*D]
    :param labels: labels of each datasets
    :param colors: color for plot
    :return:
    plt
    '''
    if labels is None:
        labels = ['set {}'.format(i) for i in len(list_datasets)]
    assert len(colors) == len(list_datasets), 'num of colors must match num of inputs'
    f, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True, squeeze=False)
    sns.despine(left=True)

    colors = ['tab:blue', 'green']
    for i, col in enumerate(colors):
        ndarray = list_datasets[i]
        mean_image = np.mean(ndarray, axis=0)
        sns.distplot(mean_image.ravel(), color=col, ax=axes[0, 0], label=labels[i])
    plt.legend()
    plt.title(title, fontsize=18)
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()
