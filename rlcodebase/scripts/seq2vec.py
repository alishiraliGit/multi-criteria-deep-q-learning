import os
import pickle
import numpy as np
from gensim.models import Word2Vec


def load_data(data_path):
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)

    return paths


def get_sentences_from_paths(paths):
    sentences = []
    for path in paths:
        sentence = [str(int(ob)) for ob in path['observation']]
        sentences.append(sentence)

    return sentences


if __name__ == '__main__':
    # Load data
    # data_folder_ = os.path.join('..', '..', 'Replay_buffer_extraction')
    data_folder_ = os.path.join('Replay_buffer_extraction')
    data_name_ = 'Paths_all_rewards_biomarkers.pkl' 

    print(os.path.join(data_folder_, data_name_))

    paths_ = load_data(os.path.join(data_folder_, data_name_).replace("\\", "/"))

    # Get sentences
    sentences_ = get_sentences_from_paths(paths_)

    # Train word2vec
    model_ = Word2Vec(sentences=sentences_, vector_size=13, window=3, min_count=10, epochs=5)
    #model_ = Word2Vec(sentences=sentences_, vector_size=3, window=3, min_count=10, epochs=5) #original

    # Extract vectors
    new_paths_ = []
    for path_ in paths_:
        ob_no_ = []
        try:
            for ob_ in path_['observation']:
                ob_no_.append(model_.wv[str(int(ob_))])

            new_path_ = path_.copy()
            new_path_['observation'] = np.array(ob_no_)

            new_paths_.append(new_path_)
        except KeyError as e:
            print(e)

    print('%.2f%% of paths reconstructed.' % (100*len(new_paths_)/len(paths_)))

    # Save new paths
    new_data_name_ = data_name_.replace('Paths', 'Encoded_paths13')

    with open(os.path.join(data_folder_, new_data_name_), 'wb') as f_:
        pickle.dump(new_paths_, f_)
