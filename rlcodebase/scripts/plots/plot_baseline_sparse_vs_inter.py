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
    do_save = True

    load_path_ = os.path.join('..', '..', '..', 'data')
    save_path_ = os.path.join('..', '..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    prefixes_ = \
        ['v4_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_MIMIC' % (1e-4, 8000, cql_alpha, r)
         for cql_alpha in [0] for r in [0]] + \
        ['v4_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_inter%d_MIMIC' % (1e-4, 8000, cql_alpha, inter)
         for cql_alpha in [0] for inter in [1, 3]] + \
        ['v4_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_MIMIC' % (1e-5, 2000, cql_alpha, r)
         for cql_alpha in [0] for r in [0]] + \
        ['v4_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_inter%d_MIMIC' % (1e-5, 2000, cql_alpha, inter)
         for cql_alpha in [0] for inter in [1, 3]]

    n_color_ = int(len(prefixes_) / 2)
    color_ = lambda cnt: np.array([(cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1)])

    is_dashed_ = lambda cnt: cnt_ >= n_color_

    legends_ = ['sparse (lr=1e-4, TUF=8000)', 'SOFA-1 (lr=1e-4, TUF=8000)', 'LAC-1 (lr=1e-4, TUF=8000)',
                'sparse (lr=1e-5, TUF=2000)', 'SOFA-1 (lr=1e-5, TUF=2000)', 'LAC-1 (lr=1e-5, TUF=2000)']

    folder_paths_ = []
    for prefix_ in prefixes_:
        folder_paths_.append(glob.glob(os.path.join(load_path_, prefix_ + '*')))

    file_paths_ = []
    for folder_path_ in folder_paths_:
        file_paths_.append([glob.glob(os.path.join(f, 'events*'))[0] for f in folder_path_])

    # Print tags
    print(get_section_tags(file_paths_[0][0]))

    # Extract data
    x_tag_ = 'Train_itr'
    y_tag_ = 'Diff_Survival_Quantile_mean'

    xs_ = [get_section_results(f[0], [x_tag_])[x_tag_] for f in file_paths_]
    y_means_ = []
    y_cis_ = []
    for file_path_ in file_paths_:
        y_raw_ = [get_section_results(f, [y_tag_])[y_tag_] for f in file_path_]
        min_len_ = np.min([len(y) for y in y_raw_])
        y_ = np.array([y[:min_len_] for y in y_raw_])
        y_means_.append(np.mean(y_, axis=0))
        y_cis_.append(np.std(y_, axis=0) / np.sqrt(y_.shape[0] - 1))

    # Plot learning curves
    plt.figure(figsize=(5, 4))

    for cnt_ in range(len(xs_)):
        if not is_dashed_(cnt_):
            min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
            plt.plot(xs_[cnt_][:min_len_], y_means_[cnt_][:min_len_], color=color_(cnt_))
        else:
            min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
            plt.plot(xs_[cnt_][:min_len_], y_means_[cnt_][:min_len_], '--', color=color_(cnt_))

    plt.legend(legends_, loc='lower right')

    for cnt_ in range(len(xs_)):
        min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
        plt.fill_between(xs_[cnt_][:min_len_],
                         y_means_[cnt_][:min_len_] - y_cis_[cnt_][:min_len_],
                         y_means_[cnt_][:min_len_] + y_cis_[cnt_][:min_len_],
                         color=color_(cnt_), alpha=0.1)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(r'$\Delta MR$', fontsize=12)

    plt.xlim((0, 60000))

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path_, 'v4_var1c_sparse_vs_inter(curves).pdf'))

    plt.show()

    # Plot bars
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    ind = np.arange(n_color_)
    width = 0.35

    for cnt_ in range(len(xs_)):
        if not is_dashed_(cnt_):
            max_i = np.argmax(y_means_[cnt_])
            ax.bar(ind[cnt_ % n_color_], y_means_[cnt_][max_i], width, yerr=y_cis_[cnt_][max_i],
                   color=color_(cnt_), ecolor='black', capsize=7)
        else:
            max_i = np.argmax(y_means_[cnt_])
            ax.bar(ind[cnt_ % n_color_] + width, y_means_[cnt_][max_i], width, yerr=y_cis_[cnt_][max_i],
                   color=color_(cnt_), ecolor='black', capsize=7, alpha=0.5)

    # Add x-axis labels
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(['sparse', 'SOFA-1', 'LAC-1'], rotation=0)

    plt.ylabel(r'Best $\Delta MR$', fontsize=12)

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path_, 'v4_var1c_sparse_vs_inter(bar).pdf'))

    plt.show()
