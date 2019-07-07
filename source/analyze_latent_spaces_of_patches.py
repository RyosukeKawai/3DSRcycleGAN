"""
* @auther tozawa
* @date 2018-8-8
* Analyze patches using t-sne. If I can observe some class, it will provide motivation
* that I seek good latent spaces.
* References
* https://qiita.com/fist0/items/d0779ff861356dafaf95
"""
import os, sys, time
import argparse
import numpy as np
import sklearn.base
import bhtsne
import matplotlib.pyplot as plt
import shutil
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import util.ioFunction_version_4_3 as IO

def _load_datasets(root, path, patch_side):
    """
    @param root: Path to input image directory
    @param path: Path to list file text
    @param patch_side: Patch side per 1 side
    """
    print(' Initilaze datasets ')
    patch_size = int(patch_side**3)

    # Read path to patches
    path_pairs = []
    with open(path) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line : continue
            path_pairs.append(line[:])

    datasets = np.empty((0, patch_size), dtype=float)
    for i in path_pairs:
        print('   Data from: {}'.format(i[0]))
        Mat = IO.read_raw_to_numpy_ColMajor(root+i[0], 'float', patch_size) # (patchsize, number_of_data)
        Mat = Mat.transpose() # (number_of_data, patchsize)
        datasets = np.append(datasets, Mat, axis=0)

    print(' Initilazation done ')

    return datasets

class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    * http://iwiwi.hatenadiary.jp/entry/2016/09/24/230640
    """

    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, x):
        return bhtsne.tsne(
            x.astype(np.float64), dimensions=self.dimensions, perplexity=self.perplexity, theta=self.theta,
            rand_seed=self.rand_seed)

def copy_to_result_dir(fn, result_dir):
    bfn = os.path.basename(fn)
    shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-R', default='',
                        help='Root directory path of input')
    parser.add_argument('--patch_list', '-p', default='../work/analyze_latent_spaces_of_patches/patch_list.txt',
                        help='Path to patch list file')
    parser.add_argument('--output_dir', '-o',  default='../work/analyze_latent_spaces_of_patches/2018-8-10',
                        help='Output directory')

    parser.add_argument('--random_state', '-rs', type=int, default=20180808,
                        help='Random state')
    parser.add_argument('--patch_side', '-ps', type=int, default=32,
                        help='Patch side')
    parser.add_argument('--number_of_trials', '-nt', type=int, default=10,
                        help='Number of trials')
    args = parser.parse_args()

    """
    * Read data
    """
    base = os.path.dirname(os.path.abspath(__file__))
    list_path = os.path.normpath(os.path.join(base, args.patch_list))
    X = _load_datasets(args.root, list_path, args.patch_side)

    """
    * Aplly t-sne to dataset
    """
    result_dir = os.path.normpath(os.path.join(base, args.output_dir))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    copy_to_result_dir(list_path, result_dir)
    np.random.seed(args.random_state)
    for i in range(args.number_of_trials):
        # Extract 3000 datas for time reductions
        idx = np.random.choice(X.shape[0], 50000, replace=False)
        x = X[idx, :]
        # Apply t-sne
        start = time.time()
        patches_proj = BHTSNE(rand_seed=args.random_state).fit_transform(x)
        end = time.time()
        print('  Number of trials: {}, Time: {:.3f} [s] '.format(i, end-start))
        # Plot result
        plt.figure(figsize=(8,8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c='navy')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        tsne_generated_fig = '{}/tsne-generated-{}.png'.format(result_dir, i)
        plt.savefig(tsne_generated_fig, dpi=120)

        # Save datas
        input_data_filename = '{}/input-data-index-{}.csv'.format(result_dir, i)
        np.savetxt(input_data_filename, idx, delimiter=',')
        tsne_generated_data_filename = '{}/tsne-generated-data-{}.csv'.format(result_dir, i)
        np.savetxt(tsne_generated_data_filename, patches_proj, delimiter=',')


if __name__ == '__main__':
    main()
