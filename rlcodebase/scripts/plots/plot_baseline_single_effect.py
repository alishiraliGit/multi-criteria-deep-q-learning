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
    do_save = False

    load_path_ = os.path.join('..', '..', '..', 'data')
    save_path_ = os.path.join('..', '..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    effect_of = 'cql'  # 'cql' or 'r'

    # Find relevant files
    if effect_of == 'r':
        cql_alphas = [0]
        rs = [0, 0.01, 0.1, 1]
    elif effect_of == 'cql':
        cql_alphas = [0.001, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3, 1]  # [0, 0.001, 0.003, 0.01]
        rs = [0]
    else:
        raise NotImplementedError

    prefixes_ = \
        ['v4_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_MIMIC' % (1e-4, 8000, cql_alpha, r)
         for cql_alpha in cql_alphas for r in rs]

    n_color_ = int(len(prefixes_))
    color_ = lambda cnt: np.array([(cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1)])

    is_dashed_ = lambda cnt: cnt_ >= n_color_

    if effect_of == 'r':
        legends_ = [r'$r$ = ' + s[s.rfind('r') + 1: s.rfind('MIMIC') - 1] for s in prefixes_]
    elif effect_of == 'cql':
        legends_ = [r'$\alpha$ = ' + s[s.rfind('cql') + 3: s.rfind('r') - 1] for s in prefixes_]
    else:
        raise NotImplementedError

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
    y_tag_ = 'Diff_Survival_Quantile_mean'  # 'T_stat'

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
    # plt.ylabel(r'$t_{Mortality}$', fontsize=12)
    plt.ylabel(r'$\Delta$ Survival Rates', fontsize=12)

    # plt.xlim((0, 60000))

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path_, 'v3_var1c_effect_of_%s.pdf' % effect_of))

    plt.show()
