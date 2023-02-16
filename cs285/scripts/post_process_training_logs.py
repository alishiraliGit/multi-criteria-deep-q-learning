import os
import numpy as np
from matplotlib import pyplot as plt
import glob
import argparse
import tensorflow as tf


def get_section_tags(file):
    all_tags = set()
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            all_tags.add(v.tag)

    return all_tags


def get_section_results(file, tags):
    data = {tag: [] for tag in tags}
    print(data.keys())
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            for tag in tags:
                if v.tag == tag:
                    data[tag].append(v.simple_value)

    return data


if __name__ == '__main__':

    ###############################
    ####### Parse arguments #######
    ###############################

    parser = argparse.ArgumentParser()
    #get prefix information for models
    parser.add_argument('--prefix', type=str)

    #specify learning curves to display
    parser.add_argument('--x_tag', type=str, default="Train_itr")
    parser.add_argument('--y_tag', type=str, default="Rho")

    #add baseline model if needed
    parser.add_argument('--baseline_model', default=None) #should be prefix of model

    #Check whether plot should be saved
    parser.add_argument('--save', action='store_true')

    #Check whether I should print the positive correlation labels
    parser.add_argument('--print_positive', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.05)

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    ##############################
    #### Plot Learning Curves ####
    ##############################

    #load_path_ = os.path.join('..', '..', 'data')
    load_path_ = os.path.join('data')

    #save_path_ = os.path.join('..', '..', 'figs')
    save_path_ = os.path.join('figs')
    
    os.makedirs(save_path_, exist_ok=True)

    # Find relevant files
    #prefix_ = 'p4_eps*_pruned_sparse_LunarLander' #Saving Ali's results 
    #prefix_ = 'MIMIC_[lSb][aO]'
    #prefix_ = 'MIMIC_s_'
    #prefix_ = 'off_pDQN'

    prefix_ = params['prefix']
    
    folder_paths_ = sorted(glob.glob(os.path.join(load_path_, params['prefix'] + '*')), key=os.path.getmtime)

    print(folder_paths_)

    file_paths_ = [glob.glob(os.path.join(f, 'events*'))[0] for f in folder_paths_]

    print([f.split(os.sep)[-1] for f in folder_paths_])

    #Adding result of baseline sparse DQN if needed
    if params['baseline_model'] is not None:
        prefix_b = params['baseline_model']
        folder_paths_b = glob.glob(os.path.join(load_path_, prefix_b + '*'))
        file_paths_b = [glob.glob(os.path.join(f, 'events*'))[0] for f in folder_paths_b]

        file_paths_ = file_paths_ + file_paths_b
        folder_paths_ = folder_paths_ + folder_paths_b
    
    # Print possible variables tags
    print(get_section_tags(file_paths_[0]))

    # Extract data

    #x_tag_ = 'Train_EnvstepsSoFar'
    #y_tag_ = 'Train_AverageReturn' #'Training_Loss'

    x_tag_ = params['x_tag']
    y_tag_ = params['y_tag']

    #y_tag_ = 'Training_Loss'
    #xs_ = [get_section_results(f, [x_tag_])[x_tag_] for f in file_paths_]
    xs_ = [get_section_results(f, [x_tag_])[x_tag_] for f in file_paths_]
    ys_ = [get_section_results(f, [y_tag_])[y_tag_] for f in file_paths_]

    print([np.max(y) for y in ys_])
    y_mat_ = np.array(ys_)
    y_mean_ = np.mean(y_mat_, axis=0)
    y_std_ = np.std(y_mat_, axis=0)
    max_idx_ = np.argmax(y_mean_)
    print(y_mean_[max_idx_], y_std_[max_idx_]/np.sqrt(2))

    y_max_ = np.max(ys_, axis=1)
    print(np.mean(y_max_), np.std(y_max_)/np.sqrt(2))

    #xs_l = [len(b) for b in xs_]
    #ys_l = [len(b) for b in ys_]

    #print(xs_l)
    #print(ys_l)

    #For offline learning we initially did not log train steps so we will need to recreate an iteration folder
    #xs_ = [list(range(len(data))) for data in ys_]

    # Plot
    plt.figure(figsize=(5, 4))

    for cnt_ in range(len(xs_)):
        try:
            plt.plot(xs_[cnt_], ys_[cnt_])
        except ValueError:
            plt.plot(xs_[cnt_][1:], ys_[cnt_])

    plt.legend([f.split(os.sep)[-1].split('-')[0] for f in folder_paths_]) #join 2 elements of string

    #mylist = [f"eps = {fold.split('_')[1]}" for fold in folder_paths_[:-1]]
    #mylist.append('baseline')
    #mylist = ['PrunedMultiDQN (e=0.3)','Baseline DQN']
    #mylist = ['PrunedMultiCQL (e=0.3)','Baseline CQL']
    #plt.legend(mylist)

    #plt.xlabel('#time steps')
    #plt.ylabel('avg return')

    plt.xlabel(params['x_tag'])
    plt.ylabel(params['y_tag'])
    #plt.ylabel('Training Loss')

    plt.tight_layout()

    if params['save']:
        plt.savefig(os.path.join(save_path_, prefix_ + f'{params["y_tag"]}_learning-curves.jpg'))

    plt.show()

    #Check which models perform above threshold at last iteration
    if params['print_positive']:
        last_ys = [y[-1] for y in ys_]
        last_y_above_t = [y>params['threshold'] for y in last_ys] 
        labels = ["".join(f.split('_')[1:2]) for f in folder_paths_]
        labels_above_t = [print(label) for i, label in enumerate(labels) if last_y_above_t[i]==True]
