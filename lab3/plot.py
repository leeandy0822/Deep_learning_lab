#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt

def read(file):
    episode = []
    score = []
    with open(file) as f:
        Lines = f.readlines()
        for line in Lines:
            ep, sc = line.split(",")
            episode.append((int)(ep.split(":")[1]))
            score.append((int)(sc.split(":")[1]))

    return np.array(episode), np.array(score)        
    
def plt_result(epoches, score, file):

    plt.title("Score vs Episode", fontsize = 18)

    plt.plot(episode[0:250000:1000], score[0:250000:1000])

    plt.xlim(0,250000)
    plt.ylim(0,250000)

    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.savefig('{0}.png'.format(file))
    plt.show()

def create_argparse():
    parser = argparse.ArgumentParser(prog="DLP homework 2", description='This code will show output result with plt')

    parser.add_argument("read_file", type=str,  nargs='?',default="record.txt", help="input file path, default is none")
    parser.add_argument("save_file", type=str,  nargs='?',default="img", help="save result with this save_file naem, default is none")

    return parser

if __name__ == "__main__":

    parser = create_argparse()
    args = parser.parse_args()

    episode, score = read(args.read_file)
    episode2, score2 = read("record_final.txt")
    episode2 += 200000
    episode = np.concatenate([episode, episode2])
    score = np.concatenate([score, score2])

    plt_result(episode, score, args.save_file)