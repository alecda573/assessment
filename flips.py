import seaborn
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
from numpy.random import choice


'''
The flip function simulates a Binomical RV. If we assume in the context of the problem that the marketing 
executive is tracking whether or not a potential client clicks on some HTML link, the flip is simulating
that there is a 70% chance a given client will click while 30% chance they do not.
So for N potential clients who visit the webpage, flip(N) will return the number of occurrences where
the potential client clicks the aforementioned hyperlink

X ~ Bin(n, p=.7) which has expected value E[X] = np.
In the context of 100 sample flips, this is not unexpected. We would expect though asymptotically with respect to n that flip(n) would return np.

'''


def flip(flips):
    r = [0,1]
    trials = 0
    for i in range(flips):
        trials += choice(r, p = [0.3,0.7], replace = True)
    return trials

def faster_flip(flips):
    r = [0, 1]
    trials = sum([choice(r, p = [0.3, 0.7], replace=True) for i in range(flips)])
    return trials


def speedFlipTest(trials=10000, num=1000):
    t1 = time.time()
    for i in range(trials):
        f = flip(num)
    print("base flip took {}".format(time.time() - t1))

    t1 = time.time()
    for i in range(trials):
        f = faster_flip(num)
    print("faster_flip took {}".format(time.time() - t1))

    return


def main(args):
    if args.profile:
        speedFlipTest()

    if args.run_experiment:
        results = np.array([faster_flip(args.num_flips) for i in range(args.num_trials)])
        seaborn.histplot(results)
        plt.savefig(args.save + "flipsHistogram.png")
        if args.show_fig:
            plt.show()

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default= os.getcwd() + "/plots/")
    parser.add_argument("--run_experiment", default=True)
    parser.add_argument("--num_trials", default=1000)
    parser.add_argument("--num_flips", default=10000)
    parser.add_argument("--profile", default=True)
    parser.add_argument("--show_fig", default=True)

    args = parser.parse_args()

    main(args)