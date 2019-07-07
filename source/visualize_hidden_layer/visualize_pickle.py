#coding:utf-8
import os, sys, time
import argparse, yaml, shutil, math
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', '-p',
                    help='Path to pickle file')
args = parser.parse_args()

with open(args.pickle, 'rb') as f:
    fig = pickle.load(f)

plt.show()
