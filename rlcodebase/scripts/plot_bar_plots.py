import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    avg_size = {
        'OptimisticMDQN': 2.81,
        'DiverseMDQN': 3.85,
        'ConsistentMDQN': 4.21,
        'DiverseEMDQN': 1.17,
        'ConsistentEMDQN': 1.21,
        r'IndependentDQN($\epsilon=0$)': 5.18,
        r'IndependentDQN($\epsilon=0.2$)': 2.57,
        r'IndependentDQN($\epsilon=0.3$)': 1.9
    }

    recall = {
        'OptimisticMDQN': 0.521,
        'DiverseMDQN': 0.759,
        'ConsistentMDQN': 0.804,
        'DiverseEMDQN': 0.396,
        'ConsistentEMDQN': 0.415,
        r'IndependentDQN($\epsilon=0$)': 0.97,
        r'IndependentDQN($\epsilon=0.2$)': 0.725,
        r'IndependentDQN($\epsilon=0.3$)': 0.623
    }

    N = len(avg_size)
    A = 6

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, np.array(list(avg_size.values()))/A, width, color='b')

    rects2 = ax.bar(ind + width, list(recall.values()), width, color='r')

    # add some
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(list(avg_size.keys()), rotation=70)

    ax.legend((rects1[0], rects2[0]), ('(Avg. size)/A', 'Recall'))

    plt.tight_layout()

    plt.savefig(os.path.join('..', '..', 'figs', 'size_recall_bar.pdf'))

    plt.show()

    re_sz = np.array(list(recall.values())) / np.array(list(avg_size.values())) * A