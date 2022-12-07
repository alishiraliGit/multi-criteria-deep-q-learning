import os
import numpy as np
from matplotlib import pyplot as plt
import glob
import tensorflow as tf


def get_section_tags(file):
    all_tags = set()
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            all_tags.add(v.tag)

    return all_tags


def get_section_results(file, tags):
    data = {tag: [] for tag in tags}
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            for tag in tags:
                if v.tag == tag:
                    data[tag].append(v.simple_value)

    return data


if __name__ == '__main__':
    load_path_ = os.path.join('..', '..', 'data')

    save_path_ = 'figs'
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    prefix_ = 'p4_eps*_pruned_sparse_LunarLander'
    folder_paths_ = glob.glob(os.path.join(load_path_, prefix_ + '*'))
    file_paths_ = [glob.glob(os.path.join(f, 'events*'))[0] for f in folder_paths_]

    # Print tags
    print(get_section_tags(file_paths_[0]))

    # Extract data
    x_tag_ = 'Train_EnvstepsSoFar'
    y_tag_ = 'Train_AverageReturn'

    xs_ = [get_section_results(f, [x_tag_])[x_tag_][:-1] for f in file_paths_]

    ys_ = [get_section_results(f, [y_tag_])[y_tag_] for f in file_paths_]

    # Plot
    plt.figure(figsize=(5, 4))

    for cnt_ in range(len(xs_)):
        plt.plot(xs_[cnt_], ys_[cnt_])

    plt.legend([f.split('_')[1] for f in folder_paths_])

    plt.xlabel('#time steps')
    plt.ylabel('avg return')

    plt.tight_layout()

    # plt.savefig(os.path.join(save_path_, prefix_ + '_learning-curves.pdf'))

    plt.show()
