import pdb

from torch import mean, log, exp

def gausian_prob(input, target):
    # pdb.set_trace()
    skip = 10
    if len(input[0])<=skip:
        skip = 0
    mu = input[0][skip:]
    sigma = input[1][skip:]
    return mean(log(sigma) + (1.0/2.0*(target[skip:] - mu)**2)/sigma)