#coding:utf-8
import os, sys, time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import util.ioFunction_version_4_3 as IO

def main():
    parser = argparse.ArgumentParser(description='Make histogram')
    parser.add_argument('--inputImageFile1', '-i1', help='input image file1')
    parser.add_argument('--inputImageFile2', '-i2', help='input image file2')
    parser.add_argument('--outputFile', '-o', help='output file')
    parser.add_argument('--bins', type=int, default=100, help='bins of histogram')
    args = parser.parse_args()

    # Read image
    img1, dict = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile1)
    img2, _ = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile2)

    # Calc histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    H1, xedges1 = np.histogram(img1, bins=args.bins)
    H2, xedges2 = np.histogram(img2, bins=args.bins)

    # Normalization
    H1, H2 = H1.astype(np.float), H2.astype(np.float)
    H1 /= len(img1)
    H2 /= len(img2)

    # Plot
    ax.bar(xedges1[:-1], H1, width=abs(xedges1[1]-xedges1[0]), color='red', alpha=0.5)
    ax.bar(xedges2[:-1], H2, width=abs(xedges2[1]-xedges2[0]), color='blue', alpha=0.5)
    ax.set_xlim(0, 2)
    plt.xlabel('Intensity')
    plt.ylabel('Relative Frequency')
    filename1, _ = os.path.splitext(os.path.basename(args.inputImageFile1))
    filename2, _ = os.path.splitext(os.path.basename(args.inputImageFile2))
    plt.title('{}(red)-{}(blue)'.format(filename1, filename2))
    plt.savefig(args.outputFile)

    #df = pd.DataFrame({'Intensity':xedges[0:-1], 'Relative Frequency':H})
    #df.to_csv('{}/{}.csv'.format(os.path.dirname(args.outputFile), filename), index=False, encoding="utf-8", mode='a')

if __name__ == '__main__':
    main()
