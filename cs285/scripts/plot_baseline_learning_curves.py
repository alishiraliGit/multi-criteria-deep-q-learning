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

    load_path_ = os.path.join('..', '..', 'data')

    save_path_ = os.path.join('..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    prefixes_ = ['expvar1clr-4_*_offline_baseline_cql%g_r%g_tuf%d_MIMIC' % (cql_alpha, r, tuf) for cql_alpha in [0.001] for r in [0, 0.001, 0.01, 0.1, 1] for tuf in [8000]] \
        # + ['expvar1inter%dfast_*_offline_baseline_cql%g_MIMIC' % (cord, 0.001) for cord in range(1)]

    n_color_ = np.maximum(2, len(prefixes_))
    color_ = lambda cnt: ((cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1))

    # legends_ = [r'$\alpha_{CQL}$ = ' + s[s.find('cql') + 3: s.find('sparse') - 1] for s in prefixes_[:n_color_]]
    legends_ = [r'$r=$%s' % s[s.rfind('_r') + 2: s.find('tuf') - 1] for s in prefixes_[:n_color_]]

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
    y_tag_ = 'Rho_mort'

    xs_ = [get_section_results(f[0], [x_tag_])[x_tag_] for f in file_paths_]
    y_means_ = []
    y_cis_ = []
    for file_path_ in file_paths_:
        y_raw_ = [get_section_results(f, [y_tag_])[y_tag_] for f in file_path_]
        min_len_ = np.min([len(y) for y in y_raw_])
        y_ = np.array([y[:min_len_] for y in y_raw_])
        y_means_.append(np.mean(y_, axis=0))
        y_cis_.append(np.std(y_, axis=0) / np.sqrt(y_.shape[0] - 1))

    # Plot
    plt.figure(figsize=(5, 4))

    for cnt_ in range(len(xs_)):
        if 'sparse' in prefixes_[cnt_] or cnt_ < n_color_:
            min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
            plt.plot(xs_[cnt_][:min_len_], y_means_[cnt_][:min_len_], color=color_(cnt_))
        else:
            min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
            plt.plot(xs_[cnt_][:min_len_], y_means_[cnt_][:min_len_], '--', color=color_(cnt_))

    plt.legend(legends_)

    for cnt_ in range(len(xs_)):
        min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
        plt.fill_between(xs_[cnt_][:min_len_],
                         y_means_[cnt_][:min_len_] - y_cis_[cnt_][:min_len_],
                         y_means_[cnt_][:min_len_] + y_cis_[cnt_][:min_len_],
                         color=color_(cnt_), alpha=0.1)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(r'$\rho_{Mortality}$', fontsize=12)
    plt.title(r'Effect of intermediate-to-sparse reward weight ($r$)')

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path_, 'expvar1c_lr-4_effect_of_r.pdf'))

    plt.show()
