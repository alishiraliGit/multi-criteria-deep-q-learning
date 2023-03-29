import os
import numpy as np
from matplotlib import pyplot as plt
import glob

from rlcodebase.scripts.plots.plot_baseline_sparse_vs_inter import get_section_tags, get_section_results


if __name__ == '__main__':
    do_save = True

    load_path_ = os.path.join('..', '..', 'data')

    save_path_ = os.path.join('..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    prefixes_ = ['arch_*_offline_baseline_cql0.001_d%g_sparse_MIMIC' % d for d in [4, 8, 16, 32, 64, 128]]

    n_color_ = len(prefixes_)
    color_ = lambda cnt: ((cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1))

    legends_ = [r'$d$ = ' + s[s.find('d') + 1: s.find('sparse') - 1] for s in prefixes_]

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
        y_ = np.array([get_section_results(f, [y_tag_])[y_tag_] for f in file_path_])
        y_means_.append(np.mean(y_, axis=0))
        y_cis_.append(np.std(y_, axis=0)/np.sqrt(y_.shape[0] - 1))

    # Plot
    plt.figure(figsize=(5, 4))

    for cnt_ in range(len(xs_)):
        plt.plot(xs_[cnt_], y_means_[cnt_], color=color_(cnt_))

    plt.legend(legends_)

    for cnt_ in range(len(xs_)):
        plt.fill_between(xs_[cnt_], y_means_[cnt_] - y_cis_[cnt_], y_means_[cnt_] + y_cis_[cnt_],
                         color=color_(cnt_), alpha=0.1)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(r'$\rho_{Mortality}$', fontsize=12)
    plt.title('Offline CQL Baselines')

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path_, 'arch_offline_baseline.pdf'))

    plt.show()
