import os
import numpy as np
from matplotlib import pyplot as plt
import glob

from rlcodebase.scripts.plots.plot_baseline_sparse_vs_inter import get_section_tags, get_section_results
from rlcodebase.infrastructure.utils.general_utils import escape_bracket_globe


def max_so_far(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    y = x.copy()
    for idx in range(1, len(x)):
        y[idx] = np.maximum(y[idx], y[idx - 1])

    return y


if __name__ == '__main__':
    do_save = False

    get_max_so_far = False

    load_path_ = os.path.join('..', '..', '..', 'data')
    save_path_ = os.path.join('..', '..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    lr1 = 1e-5
    tuf1 = 1000
    r1 = 5

    lr2 = 1e-4
    tuf2 = 8000

    cql_alpha = 0.001

    prefixes_ = \
        ['v6_var1c_*_offline_pruned_cmdqn_lr[1]%.0e_tuf[1]%g_r[1]%g_lr[2]%.0e_tuf[2]%g_cql%g_alpha%g_sparse_MIMIC'
         % (lr1, tuf1, r1, lr2, tuf2, cql_alpha, alpha) for alpha in [20, 40, 80, 160]] + \
        ['v6_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_MIMIC' % (1e-4, 8000, 0.001, 0)]

    prefixes_ = [escape_bracket_globe(prefix) for prefix in prefixes_]

    n_color_ = len(prefixes_) - 1
    color_ = lambda cnt: ((cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1)) if cnt_ < n_color_ else 'k'
    line_type_ = lambda cnt: '-' if cnt_ < n_color_ else '--'

    legends_ = [r'Pruned CQL($\alpha$=%g, $\beta$=' % cql_alpha + s[s.find('alpha') + 5: s.find('sparse') - 1] + ')'
                for s in prefixes_[:-1]]
    legends_.append('Best CQL baseline')

    folder_paths_ = []
    for prefix_ in prefixes_:
        folder_paths_.append(glob.glob(os.path.join(load_path_, prefix_ + '*')))

    print(folder_paths_)

    file_paths_ = []
    for folder_path_ in folder_paths_:
        file_paths_.append([glob.glob(os.path.join(glob.escape(f), 'events*'))[0] for f in folder_path_])

    print(file_paths_)

    # Print tags
    print(get_section_tags(file_paths_[0][0]))

    # Extract data
    x_tag_ = 'Train_itr'
    y_tag_ = 'Diff_Survival_Quantile_mean'  # 'T_stat'

    xs_ = [get_section_results(f[0], [x_tag_])[x_tag_] for f in file_paths_]
    y_means_ = []
    y_cis_ = []
    for file_path_ in file_paths_:
        if get_max_so_far:
            y_raw_ = [max_so_far(get_section_results(f, [y_tag_])[y_tag_]) for f in file_path_]
        else:
            y_raw_ = [get_section_results(f, [y_tag_])[y_tag_] for f in file_path_]
        min_len_ = np.min([len(y) for y in y_raw_])
        y_ = np.array([y[:min_len_] for y in y_raw_])
        y_means_.append(np.mean(y_, axis=0))
        y_cis_.append(np.std(y_, axis=0) / np.sqrt(y_.shape[0] - 1))

    # Plot
    plt.figure(figsize=(5, 4))

    for cnt_ in range(len(xs_)):
        min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))

        plt.plot(xs_[cnt_][:min_len_], y_means_[cnt_][:min_len_], line_type_(cnt_), color=color_(cnt_))

    plt.legend(legends_)

    for cnt_ in range(len(xs_)):
        min_len_ = np.minimum(len(xs_[cnt_]), len(y_means_[cnt_]))
        plt.fill_between(xs_[cnt_][:min_len_],
                         y_means_[cnt_][:min_len_] - y_cis_[cnt_][:min_len_],
                         y_means_[cnt_][:min_len_] + y_cis_[cnt_][:min_len_],
                         color=color_(cnt_), alpha=0.1)

    plt.xlabel('Iterations', fontsize=12)
    # plt.ylabel(r'$t_{Mortality}$', fontsize=12)
    plt.ylabel(r'$\Delta MR$', fontsize=12)
    # plt.ylabel(r'Best Return', fontsize=12)

    # plt.xlim(left=10000)
    # plt.ylim(bottom=0.15)

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(
            save_path_,
            'v3_var1c_offline_pruned_cmdqn_lr[1]%.0e_tuf[1]%g_r[1]%g_lr[2]%.0e_tuf[2]%g_cql%g_alpha[varies]_sparse'
            % (lr1, tuf1, r1, lr2, tuf2, cql_alpha)
        ))
        # plt.savefig(os.path.join(save_path_, 'expll_mdqn_phase_2_LunarLander.pdf'))

    plt.show()
