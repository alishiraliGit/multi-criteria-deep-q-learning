import os
import numpy as np
from matplotlib import pyplot as plt
import glob

from rlcodebase.scripts.plots.plot_baseline_sparse_vs_inter import get_section_tags, get_section_results


if __name__ == '__main__':
    do_save = False

    load_path_ = os.path.join('..', '..', '..', 'data')

    save_path_ = os.path.join('..', '..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    lr = 1e-5
    tuf = 1000
    cql_alpha = 0.001
    r = 10

    prefixes_ = ['v6_var1c_*_offline_cmdqn_lr%.0e_tuf%g_cql%g_alpha%g_r%g_MIMIC'
                 % (lr, tuf, cql_alpha, alpha, r) for alpha in [20, 40, 80, 160]]

    n_color_ = np.maximum(int(len(prefixes_)), 2)
    color_ = lambda cnt: ((cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1))
    line_type_ = lambda cnt: '-' if cnt_ < n_color_ else '--'

    legends_ = [r'$\alpha$ = ' + s[s.find('alpha') + 5: s.rfind('r') - 1] + ', '
                + r'tuf = ' + s[s.rfind('tuf') + 3: s.find('cql') - 1]
                for s in prefixes_]

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
    xs_ = [get_section_results(f[0], [x_tag_])[x_tag_] for f in file_paths_]

    y_tags_ = ['Recall', 'Precision', 'F1', 'Avg_Num_Available_Actions', 'RTG_Available_Actions']
    y_labels_ = ['Recall', 'Precision', 'F1', 'Avg Num of Available Actions', 'Avg Reward-To-Go of Available Actions']

    for i_y_, y_tag_ in enumerate(y_tags_):
        y_means_ = []
        y_cis_ = []
        for file_path_ in file_paths_:
            y_raw_ = [get_section_results(f, [y_tag_])[y_tag_] for f in file_path_]
            min_len_ = np.min([len(y) for y in y_raw_])
            y_ = np.array([y[:min_len_] for y in y_raw_])
            y_means_.append(np.mean(y_, axis=0))
            y_cis_.append(np.std(y_, axis=0)/np.sqrt(y_.shape[0] - 1))

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
        plt.ylabel(y_labels_[i_y_], fontsize=12)
        plt.title(r'Offline MDQN ($\alpha_{CQL}$ = %g)' % cql_alpha)

        plt.tight_layout()

        if do_save:
            plt.savefig(os.path.join(save_path_,
                                     'v3_var1c_offline_cmdqn_lr%.0e_tuf%g_cql%g_alpha[varies]_r%g[%s].pdf'
                                     % (lr, tuf, cql_alpha, r, y_tag_)))

    plt.show()
