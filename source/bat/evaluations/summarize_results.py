# coding:utf-8
import sys, os, time
import argparse, glob
import numpy as np
import pandas as pd
from itertools import chain

parser = argparse.ArgumentParser(description='Summarize evaluations log')
parser.add_argument('--root', '-R', default='',
                    help='Input directory (This directory has each iteration model result)')
args = parser.parse_args()

dir_pathes = glob.glob('{}/*'.format(args.root))
dir_list = sorted([int(os.path.basename(d)) for d in dir_pathes if os.path.isdir(d)])

data_df = pd.read_csv('{}/{}/results.csv'.format(args.root, dir_list[0]))
data_df['iteraton'] = dir_list[0]

for dir in dir_list[1:]:
    df = pd.read_csv('{}/{}/results.csv'.format(args.root, dir))
    df['iteraton'] = dir

    data_df = pd.concat([data_df, df])

"""
https://code-examples.net/ja/q/c8a10d
https://code.i-harness.com/ja/q/cca4d8
"""
cols = data_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_df = data_df[cols]
data_df.to_csv('{}/results.csv'.format(args.root),index=False, encoding='utf-8', mode='w')
