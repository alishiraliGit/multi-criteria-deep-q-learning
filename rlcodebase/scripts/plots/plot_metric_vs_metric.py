import os
import pickle
import glob
from functools import reduce
import operator

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import combine_pvalues
from sklearn.linear_model import LinearRegression

from rlcodebase.infrastructure.utils.general_utils import escape_bracket_globe


def read_files_with_prefix_search(loadpath, folder_prefixes, file_prefix, verbose=True):
    folder_prefixes = [escape_bracket_globe(prefix) for prefix in folder_prefixes]

    folder_paths = []
    for folder_prefix in folder_prefixes:
        folder_paths.append(glob.glob(os.path.join(loadpath, folder_prefix + '*')))

    file_paths = []
    for folder_path in folder_paths:
        file_paths.append([glob.glob(os.path.join(glob.escape(f), file_prefix))[0] for f in folder_path])

    if verbose:
        print('Found %d files' % sum([len(file_path) for file_path in file_paths]))

    results = []
    for file_path in file_paths:
        res = []
        for fi in file_path:
            with open(fi, 'rb') as f:
                res.append(pickle.load(f))
        results.append(res)

    return results


def read_eval_files_with_prefix_search(loadpath, folder_prefixes, verbose=True):
    metrics_dicts = read_files_with_prefix_search(loadpath, folder_prefixes, 'metrics*', verbose)
    actions_dicts = read_files_with_prefix_search(loadpath, folder_prefixes, 'actions*', verbose)

    return metrics_dicts, actions_dicts


def flatten_list_of_lists(list_of_lists):
    return reduce(operator.concat, list_of_lists)


def combine_estimated_metrics_weighted(metrics_dicts, metric_name):
    m = []  # Means
    w = []  # 1/Var
    for metrics_dict in metrics_dicts:
        m.append(metrics_dict[metric_name]['mean'])
        std = metrics_dict[metric_name]['std']/np.sqrt(metrics_dict[metric_name]['n'])
        w.append(1 / std**2)
    m = np.array(m)
    w = np.array(w)

    weighted_mean = np.sum(w*m)/np.sum(w)
    std_err = np.sqrt(1/np.sum(w))

    return weighted_mean, std_err


def combine_estimated_metrics(metrics_dicts, metric_name):
    m = []  # Means
    for metrics_dict in metrics_dicts:
        m.append(metrics_dict[metric_name]['mean'])
    m = np.array(m)

    mean = np.mean(m)
    std_err = np.std(m)/np.sqrt(len(m) - 1)

    # TODO
    if metric_name == 'WIS' or metric_name == 'FQE':
        mean *= 100/1.155
        std_err *= 100/1.155
        pass
    if metric_name == 'Accuracy':
        mean *= 100
        std_err *= 100

    return mean, std_err


def combine_tests(metrics_dicts, metric_name):
    pvalues = []
    for metrics_dict in metrics_dicts:
        pvalues.append(metrics_dict[metric_name]['pvalue'])

    statistic, _pvalue = combine_pvalues(pvalues, method='stouffer')

    # noinspection PyTypeChecker, PyUnresolvedReferences
    return statistic, 0


def combine_metrics_dicts(metrics_dicts, metric_name):
    sample_res_dict = metrics_dicts[0][metric_name]
    if 'mean' in sample_res_dict:
        return combine_estimated_metrics(metrics_dicts, metric_name)
    elif 'statistic' in sample_res_dict:
        return combine_tests(metrics_dicts, metric_name)
    else:
        raise NotImplementedError


def combine_list_of_metrics_dicts(list_of_metrics_dicts, metric_name):
    vals = []
    errs = []
    for metrics_dicts in list_of_metrics_dicts:
        val, err = combine_metrics_dicts(metrics_dicts, metric_name)

        vals.append(val)
        errs.append(err)

    return np.array(vals), np.array(errs)


def plot_linear_fit(x, y, cl, x_range=None, alpha=1.):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]

    reg = LinearRegression()
    reg.fit(x, y)

    if x_range is None:
        x_range = np.array([np.min(x), np.max(x)])
    if not isinstance(x_range, np.ndarray):
        x_range = np.array(x_range)

    y_pr = reg.predict(x_range[:, np.newaxis])

    plt.plot(x_range, y_pr, '--', color=cl, alpha=alpha)


if __name__ == '__main__':
    # ----- Settings ------
    do_save = True

    load_path = 'data'
    save_path = 'figs'

    metric_1 = 'Accuracy'
    # metric_1 = 'Recall'

    # metric_2 = 'Diff_Survival_Quantiles'
    metric_2 = 'WIS'
    # metric_2 = 'FQE'
    # metric_2 = 'Num_of_Available_Actions'

    x_label = '% of actions similar to physicians'
    # y_label = r'$\Delta MR$'
    y_label = 'Value (WIS)'
    # y_label = 'Value (FQE)'

    # ----- To be evaluated models -----
    # Eval fixed params
    v = 6

    lr1 = 1e-5
    tuf1 = 1000
    r1 = 10

    lr2 = 1e-4
    tuf2 = 8000

    cql_alpha = 0.001

    # Eval changing param
    consistency_alphas = [20, 40, 160]

    # Eval prefixes
    eval_prefixes = \
        ['v%d_var1c_*_offline_pruned_cmdqn_lr[1]%.0e_tuf[1]%g_r[1]%g_lr[2]%.0e_tuf[2]%g_cql%g_alpha%g_sparse_eval'
         % (v, lr1, tuf1, r1, lr2, tuf2, cql_alpha, alpha) for alpha in consistency_alphas]

    # ----- Baseline models -----
    # Baseline fixed params
    v_b = 6

    lr_b = 1e-4
    tuf_b = 8000

    r_b = 0

    # Baseline changing params
    cql_alphas_b = [0.001, 0.005, 0.01]  # [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]  # [0.001, 0.003, 0.01]

    # Baseline prefixes
    baseline_prefixes = \
        ['v%d_var1c_*_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_eval'
         % (v_b, lr_b, tuf_b, cql_alpha, r_b) for cql_alpha in cql_alphas_b]

    # ----- Load results -----
    eval_metrics_dicts, _ = read_eval_files_with_prefix_search(load_path, eval_prefixes)
    baseline_metrics_dicts, _ = read_eval_files_with_prefix_search(load_path, baseline_prefixes)

    # ----- Extract desired metrics -----
    eval_avg_metric_1, eval_err_metric_1 = combine_list_of_metrics_dicts(eval_metrics_dicts, metric_1)
    eval_avg_metric_2, eval_err_metric_2 = combine_list_of_metrics_dicts(eval_metrics_dicts, metric_2)

    baseline_avg_metric_1, baseline_err_metric_1 = combine_list_of_metrics_dicts(baseline_metrics_dicts, metric_1)
    baseline_avg_metric_2, baseline_err_metric_2 = combine_list_of_metrics_dicts(baseline_metrics_dicts, metric_2)

    # ----- Log -----
    print('Eval:')
    print(metric_1)
    for idx in range(len(eval_avg_metric_1)):
        print('%g: %g +- %g' % (consistency_alphas[idx], eval_avg_metric_1[idx], eval_err_metric_1[idx]))
    print(metric_2)
    for idx in range(len(eval_avg_metric_2)):
        print('%g: %g +- %g' % (consistency_alphas[idx], eval_avg_metric_2[idx], eval_err_metric_2[idx]))

    print('\nBaseline:')
    print(metric_1)
    for idx in range(len(baseline_avg_metric_1)):
        print('%g: %g +- %g' % (cql_alphas_b[idx], baseline_avg_metric_1[idx], baseline_err_metric_1[idx]))
    print(metric_2)
    for idx in range(len(baseline_avg_metric_2)):
        print('%g: %g +- %g' % (cql_alphas_b[idx], baseline_avg_metric_2[idx], baseline_err_metric_2[idx]))

    # ----- Plot -----
    plt.figure(figsize=(5, 4))

    plt.plot(baseline_avg_metric_1, baseline_avg_metric_2, 'k')
    plt.plot(eval_avg_metric_1, eval_avg_metric_2, 'r')

    plt.errorbar(
        x=baseline_avg_metric_1,
        y=baseline_avg_metric_2,
        xerr=baseline_err_metric_1,
        yerr=baseline_err_metric_2,
        fmt='o',
        color='k',
        ecolor=[0.5, 0.5, 0.5],
        capsize=7
    )

    plt.errorbar(
        x=eval_avg_metric_1,
        y=eval_avg_metric_2,
        xerr=eval_err_metric_1,
        yerr=eval_err_metric_2,
        fmt='o',
        color='r',
        ecolor=[1, 0.5, 0.5],
        capsize=7
    )

    plot_linear_fit(baseline_avg_metric_1, baseline_avg_metric_2, 'k', x_range=[7, 30], alpha=0.6)
    plot_linear_fit(eval_avg_metric_1, eval_avg_metric_2, 'r', x_range=[7, 25], alpha=0.6)

    plt.legend(['CQL', 'Pruned CQL'])

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # plt.xscale('log')
    # plt.yscale('log')

    plt.tight_layout()

    if do_save:
        plt.savefig(os.path.join(save_path, f'{metric_2}_vs_{metric_1}.pdf'))

    plt.show()
