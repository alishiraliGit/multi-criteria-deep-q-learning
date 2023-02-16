import os
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import argparse
from collections import Counter

import numpy as np

"""
eps [0.1 0.2 0.3]
Pareto set accuacies for some models [82.18469727, 72.467776, 62.38149807]
Mean Pareto set size [ 3.47977536, 2.56945326, 1.90034 ]
Action space dim is 6
Chance level [0.57996256, 0.42824221, 0.31672333]

eps [0.1 0.2 0.3]
Pareto set accuacies for some models [74.23434419 62.85043658 55.17603981]
Mean Pareto set size [3.28187964, 2.22081495, 1.67816167] 
Action space dim is 6
Chance level [0.54697994, 0.37013582, 0.27969361]
"""

if __name__ == "__main__":
    
    X = ['PrunedMultiDQN (e=0.1)','PrunedMultiDQN (e=0.2)','PrunedMultiDQN (e=0.3)','PrunedMultiCQL (e=0.1)','PrunedMultiCQL (e=0.2)','PrunedMultiCQL (e=0.3)']
    pareto_acc = [82.18469727, 72.467776, 62.38149807, 74.23434419, 62.85043658, 55.17603981]
    chance = [57.996256, 42.824221, 31.672333,54.697994, 37.013582, 27.969361]

    cur_path = os.getcwd()
    fig_path_ = os.path.join(cur_path,'figs')

    
    X_axis = np.arange(len(X))
    
    plt.bar(X_axis - 0.2, pareto_acc, 0.4, label = 'Recall, %', color='b')
    plt.bar(X_axis + 0.2, chance, 0.4, label = '(Avg. size)/A, %', color='r')
    
    #plt.xticks(X_axis, X, rotation=60)
    plt.xticks(X_axis, X, rotation=70)
    plt.xlabel("")
    #plt.ylabel("Number of Students")
    #plt.title("Number of Students in each group")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(fig_path_, 'DQN_CQL_acc_vs_chance.pdf'))
    plt.show()

    #plt.figure.autofmt_xdate()

    

    #if params['save']:
            #plt.savefig(os.path.join(fig_path, folder_paths_short[i] + '_counts.jpg'))