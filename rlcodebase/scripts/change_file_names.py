import os
import glob

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')

    folders = glob.glob(os.path.join(data_path, 'p4_*'))

    for f in folders:
        os.rename(f, f.replace('|', '-'))
