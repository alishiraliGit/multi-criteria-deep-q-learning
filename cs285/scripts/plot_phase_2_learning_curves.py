import os
import numpy as np
from matplotlib import pyplot as plt
import glob

from plot_baseline_learning_curves import get_section_tags, get_section_results


if __name__ == '__main__':
    do_save = False

    load_path_ = os.path.join('..', '..', 'data')

    save_path_ = os.path.join('..', '..', 'figs')
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    cql_alpha = 0.001
    r = 1
    tuf1 = 1000
    tuf2 = 8000
    prefixes_ = ['expvar1clr1-5lr2-4_*_offline_pruned_cmdqn_alpha%g_cql%g_r%g_tuf1%d_tuf2%d_sparse' % (alpha, cql_alpha, r, tuf1, tuf2) for alpha in [5, 10, 20]] \
        + ['expvar1clr-4_*_offline_baseline_cql%g_r%g_tuf%d_MIMIC' % (cql_alpha, 0, tuf2)]

    n_color_ = len(prefixes_) - 1
    color_ = lambda cnt: ((cnt % n_color_)/(n_color_ - 1), 0, 1 - (cnt % n_color_)/(n_color_ - 1)) if cnt_ < n_color_ else 'k'
    line_type_ = lambda cnt: '-' if cnt_ < n_color_ else '--'

    legends_ = [r'$\alpha$ = ' + s[s.find('alpha') + 5: s.find('cql') - 1] + r', $r$ = ' + s[s.rfind('_r') + 2: s.find('tuf') - 1] for s in prefixes_[:-1]]
    legends_.append('best baseline')

    folder_paths_ = []
    for prefix_ in prefixes_:
        folder_paths_.append(glob.glob(os.path.join(load_path_, prefix_ + '*')))

    file_paths_ = []
    for folder_path_ in folder_paths_:
        file_paths_.append([glob.glob(os.path.join(f, 'events*'))[0] for f in folder_path_])
    
    print(file_paths_)

    # Print tags
    print(get_section_tags(file_paths_[0][0]))

    # Extract data
    x_tag_ = 'Train_itr'
    y_tag_ = 'Rho'

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
    plt.ylabel(r'$\rho_{Mortality}$', fontsize=12)
    plt.title('Offline MDQN')

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path_, 'expvar1clr1-5lr2-4_mdqn_phase_2(r%g_tuf1%d_tuf2%d).pdf' % (r, tuf1, tuf2)))

    plt.show()
