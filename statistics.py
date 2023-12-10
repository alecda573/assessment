import csv
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy
import os
import pandas as pd

import scipy.stats as stats
import pylab

import argparse


def t_test(x, y, alternative='less', equal_var=False):
    _, double_p = stats.ttest_ind(x, y, equal_var=equal_var)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p / 2.
        else:
            pval = 1.0 - double_p / 2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p / 2.
        else:
            pval = 1.0 - double_p / 2.
    return pval


def main(args):
    os.makedirs(args.save, exist_ok=True)
    pre, post =[], []

    with open(args.data, newline="") as c:
        data = csv.reader(c, delimiter=' ')
        for row in data:
            split = [int(x) for x in row[0].split(",") if x != ""]
            try:
                pre.append(split[0])
            except:
                pre.append(-1)
                pass
            try:
                post.append(split[1])
            except:
                post.append(-1)
                pass

    post_filtered = [x for x in post if x != -1]
    pre, post, post_filtered = np.array(pre), np.array(post), np.array(post_filtered)

    pre_stats = {"mean": np.mean(pre),
                 "median": np.median(pre),
                 "var": np.var(pre),
                 "std": np.std(pre),
                 "num_samples": len(pre)}
    post_stats = {"mean": np.mean(post_filtered),
                  "median": np.median(post_filtered),
                  "var": np.var(post_filtered),
                  "std": np.std(post_filtered),
                  "num_samples": len(post_filtered)}

    if args.gen:
        seaborn.histplot(pre, kde=False)
        plt.title("Pre Course Score Histogram")
        plt.savefig(args.save + "preHist.png")
        if args.showPlots:
            plt.show()

        seaborn.histplot(post_filtered, kde=False)
        plt.title("Post Course Score Histogram")
        plt.savefig(args.save + "postHist.png")
        if args.showPlots:
            plt.show()

        df = pd.DataFrame({"pre": pre, "post": post})
        seaborn.boxplot(df.loc[df['post'] >= 0])
        plt.title("Box-Plot for Pre/Post Course Scores")
        plt.savefig(args.save + "boxPlot.png")
        if args.showPlots:
            plt.show()

        stats.probplot(pre, dist="norm", plot=pylab)
        pylab.title("Pre Course QQ Plot")
        pylab.savefig(args.save + "preQQ.png")
        if args.showPlots:
            pylab.show()

        stats.probplot(post_filtered, dist="norm", plot=pylab)
        pylab.title("Post Course QQ Plot")
        pylab.savefig(args.save + "postQQ.png")
        if args.showPlots:
            pylab.show()

    # Kolmogorov-Smirnovv test for normality
    # Technically requires more than 50 samples which we dont have
    # All three distributions for the KS test say normality although its questionable whether this test is appropriate

    preNormal = stats.kstest(pre, 'norm')
    postNormal = stats.kstest(post_filtered, 'norm')
    print("KS test results for the pre course distribution: ", preNormal)
    print("KS test results for the post course distribution: ", postNormal)

    # Instaed we will use the Shapiro-Wilk test which is appropriate for N <= 50
    # The Shapiro test is more appropriate given the smaller sample size
    # Shapiro gives the post and difference distributions are normal at a 90% alpha while the pre distribution is not considered normal

    preShap = stats.shapiro(pre)
    postShap = stats.shapiro(post_filtered)
    print("Shapiro test results for the pre course distribution: ", preShap)
    print("Shapiro test results for the post course distribution: ", postShap)

    # Need to assess if variances of the pre and post distribution are equal or not
    levene = stats.levene(pre, [x for x in post if x != -1])
    # Up to alpha = .05 we can assume the variances are drawn from the same population so traditional t-test is appropriate
    print("Levene test for equal variances: ", levene)

    # Calculate 90% confidence intervals for each data
    # This approach is technically only appropriate if the normality assumption is satisfied which is unclear for the pre dist.
    significance_levels = [0.9, 0.95, 0.99]
    for sig in significance_levels:
        pre_confidence_interval = stats.t.interval(confidence=sig, df=len(pre) - 1, loc=pre_stats['mean'],
                                                   scale=stats.sem(pre))
        shifted_pre_confidence_interval = stats.t.interval(confidence=sig, df=len(pre) - 1, loc=pre_stats["mean"] + 10,
                                                           scale=stats.sem(pre))
        post_confidence_interval = stats.t.interval(confidence=sig, df=len(post) - 1, loc=post_stats['mean'],
                                                    scale=stats.sem(np.array([x for x in post if x != -1])))
        print(
            "At significance level {} the pre CI was {} the 10 point shifted pre CI was {} the post CI was {} ".
            format(sig, pre_confidence_interval, shifted_pre_confidence_interval, post_confidence_interval))

    for i in range(0, 11):
        print("The p-value for equal means between pre + {} pts and post assuming equal variances are: ".format(i),
              t_test(pre + i, post_filtered, equal_var=True))


    return pre_stats, post_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', default=True, help="Generate plots boolean type")
    parser.add_argument('--save', default=os.getcwd() + "/plots/", help="rel path to save graphics")
    parser.add_argument('--data', default="data/VUdata.csv", help="path to csv file containing test data")
    parser.add_argument("--showPlots", default=False)

    args = parser.parse_args()

    pre_stats, post_stats = main(args)

    print("summary stats for the pre course distribution are: ", pre_stats)
    print("summary stats for the post course distribution are: ", post_stats)