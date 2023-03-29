import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    save_path = os.path.join('..', '..', 'figs')
    os.makedirs(save_path, exist_ok=True)

    model_rho = {
        'baseline (not conservative)': {'mean': 0.226, 'err': 0.006},
        r'CQL($\allpha=0.001$)': {'mean': 0.22, 'err': 0.013},
        r'CQL($\allpha=0.01$)': {'mean': 0.204, 'err': 0.006},
        r'Pruned CQL($\allpha=0.001$, $\beta=5$)': {'mean': 0.228, 'err': 0.011},
        r'Pruned CQL($\allpha=0.001$, $\beta=10$)': {'mean': 0.238, 'err': 0.010},
        r'Pruned CQL($\allpha=0.001$, $\beta=20$)': {'mean': 0.238, 'err': 0.014},
    }

    model_acc = {
        'baseline (not conservative)': {'mean': 7.73, 'err': 2.36},
        r'CQL($\allpha=0.001$)': {'mean': 10.4, 'err': 3.0},
        r'CQL($\allpha=0.01$)': {'mean': 27.3, 'err': 1.5},
        r'Pruned CQL($\allpha=0.001$, $\beta=5$)': {'mean': 15.3, 'err': 1.5},
        r'Pruned CQL($\allpha=0.001$, $\beta=10$)': {'mean': 17.9, 'err': 1.6},
        r'Pruned CQL($\allpha=0.001$, $\beta=20$)': {'mean': 23.7, 'err': 2.6},
    }

    baselines = list(model_rho.keys())[:3]
    others = list(model_rho.keys())[3:]

    plt.figure(figsize=(5, 4))

    plt.plot(
        [model_acc[mdl]['mean'] for mdl in baselines],
        [model_rho[mdl]['mean'] for mdl in baselines],
        color='k'
    )
    plt.plot(
        [model_acc[mdl]['mean'] for mdl in others],
        [model_rho[mdl]['mean'] for mdl in others],
        color='r'
    )

    plt.errorbar(
        x=[model_acc[mdl]['mean'] for mdl in baselines],
        y=[model_rho[mdl]['mean'] for mdl in baselines],
        xerr=[model_acc[mdl]['err'] for mdl in baselines],
        yerr=[model_rho[mdl]['err'] for mdl in baselines],
        fmt='o',
        color='k',
        ecolor=[0.5, 0.5, 0.5],
        capsize=7
    )
    plt.errorbar(
        x=[model_acc[mdl]['mean'] for mdl in others],
        y=[model_rho[mdl]['mean'] for mdl in others],
        xerr=[model_acc[mdl]['err'] for mdl in others],
        yerr=[model_rho[mdl]['err'] for mdl in others],
        fmt='o',
        color='r',
        ecolor=[1, 0.5, 0.5],
        capsize=7
    )

    plt.legend(['CQL', 'Pruned CQL'])

    plt.xlabel('% of actions similar to physicians', fontsize=12)
    plt.ylabel(r'$\rho_{Mortality}$', fontsize=12)

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'rho_vs_physician_accuracy.pdf'))
