

import argparse
import scipy
import numpy as np
#import statsmodels.stats.contingency_tables
from statsmodels.stats.contingency_tables import mcnemar
#from statsmodels.stats.contingecy_tables import mcneamr

parser = argparse.ArgumentParser()

parser.add_argument('--first', type=str)
parser.add_argument('--second', type=str)
parser.add_argument('--right', type=str)


args = parser.parse_args()

def get_lables(path):
    with open(path) as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    return np.array(lines)



first = get_lables(args.first)
second = get_lables(args.second)
right = get_lables(args.right)



tt = np.sum((first == right)&(second == right))
ff = np.sum((first != right)&(second != right))
tf = np.sum((first == right)&(second != right))
ft = np.sum((first != right)&(second == right))

print('first right, second not right:', tf, ' ' , 'first not right, second right:', ft)

print()

table = [[tt, tf], [ft, ff]]

result = mcnemar(table, exact=True)
print('Exact mcNemar statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

result = mcnemar(table, exact=False)
print('mcNemar statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

print()

p = scipy.stats.binom_test(tf, tf+ft, 0.5, alternative = 'two-sided')
print('binomial two-sided statistic = (%d, %d) p-value=%.3f' % (tf, tf+ft, p))

p = scipy.stats.binom_test(tf, tf+ft, 0.5, alternative = 'greater')
print('binomial greater statistic = (%d, %d) p-value=%.3f' % (tf, tf+ft, p))

p = scipy.stats.binom_test(tf, tf+ft, 0.5, alternative = 'less')
print('binomial less statistic = (%d, %d) p-value=%.3f' % (tf, tf+ft, p))

nab = (first == right).astype(np.float32) - (second == right).astype(np.float32)

print()
print('wilcoxon')
for zero_method in ["pratt", "wilcox", "zsplit"]:
    for alternative in ["two-sided", "greater", "less"]:
        print(zero_method, alternative, scipy.stats.wilcoxon(nab, zero_method = zero_method, alternative = alternative))
        



