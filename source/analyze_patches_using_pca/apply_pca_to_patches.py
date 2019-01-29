# coding:utf-8
import os, sys, time, random
import numpy as np
import argparse, pickle, yaml, glob
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO


def load_dataset(fnames, labels, minV, maxV):
    dataset = []
    for f in fnames:
        nda = ((IO.read_mhd_and_raw_withoutSitk(f)[0]-minV)/(maxV-minV)*2.)-1.
        dataset.append(nda)
    dataset = np.asarray(dataset, dtype=np.float)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDirList', '-i', help='input dir list file')
    parser.add_argument('--outputDir', '-o', help='output directory')

    parser.add_argument('--dataset', help='path to dataset pickle')
    parser.add_argument('--labels', help='path to label pickle')
    parser.add_argument('--pca', help='path to pca pickle')

    parser.add_argument('--min', default=0.,
                        help='Minimum value in LR image')
    parser.add_argument('--max', default=255.,
                        help='Maxmum value in LR image')

    args = parser.parse_args()

    print('----- Save configs -----')
    result_dir = args.outputDir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open('{}/args_configs.yml'.format(result_dir), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))

    print('----- Load patches -----')
    if not args.dataset:
        dnames = []
        with open(args.inputDirList) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line : continue
                dnames.append(line[:])
        "http://monmon.hateblo.jp/entry/20110218/1298002354"
        dnames = [i for d in dnames for i in d]
        fnames = [glob.glob('{}/*.mhd'.format(d)) for d in dnames]
        fnames = [i for d in fnames for i in d]
        "https://github.com/zEttOn86/classify-anime-characters-with-fine-tuning-model/blob/master/classify-anime-characters-with-fine-tuning-model.ipynb"
        labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
        dnames = [os.path.basename(d) for d in dnames]
        labels = [dnames.index(l) for l in labels]
        labels = np.asarray(labels, dtype=int)
        dataset = load_dataset(fnames, labels, args.min, args.max)
        with open('{}/dataset.pickle'.format(result_dir), 'wb') as f:
            pickle.dump(dataset, f)
        with open('{}/labels.pickle'.format(result_dir), 'wb') as f:
            pickle.dump(labels, f)
    else:
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
        with open(args.labels, 'rb') as f:
            labels = pickle.load(f)

    print('# labels: {}'.format(len(np.unique(labels))))

    print('----- Apply pca to patches -----')
    if not args.pca:
        pca = PCA(n_components=53649)
        start = time.time()
        X_r2 = pca.fit(dataset)
        transformed=pca.fit_transform(X_r2)
        end = time.time()
        print('  Time: {:.3f} [s] '.format(end-start))
        with open('{}/pca.pickle'.format(result_dir), 'wb') as f:
            pickle.dump(pca, f)
    else:
        with open(args.pca, 'rb') as f:
            pca = pickle.load(f)
        X_r2 = pca.transform(dataset)

    np.savetxt('{}/transform.csv'.format(result_dir), transformed, delimiter=',')

    # print('----- Plot results -----')
    # colors = ['navy', 'turquoise', 'darkorange']
    # target_names = ['gt', 'with-SN', 'without-SN']
    # alphas = [0.3, 0.5, 0.5]
    # figure = plt.figure()
    # ax = figure.add_subplot(111, aspect='equal', projection='3d')
    # for color, i, target_name, alpha in zip(colors, [0,1,2], target_names, alphas):
    #     ax.scatter(X_r2[labels==i,0], X_r2[labels==i,1], X_r2[labels==i,2], lw=0, c=color, alpha=alpha, label=target_name)
    # ax.axis('tight')
    # ax.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.savefig('{}/plot.png'.format(result_dir), dpi=120)
    # with open('{}/plot.pickle'.format(result_dir), 'wb') as f:
    #     pickle.dump(figure, f)
    # plt.show()

if __name__ == '__main__':
    main()
