import matplotlib.pyplot as plt
import numpy as np
import random

d = 2

def dataFromGMM(N, priors, means, covs):
    components = priors.size
    x = np.zeros((d, N))
    labels = np.zeros(N)
    u = np.random.rand(N)
    csum = np.cumsum(priors)
    # go through all u and assign the labels to x
    for l in range(0, components):
        # find all u that are less than the threshold
        indl = np.where(u <= csum[l])[0]
        # track the number of samples below this threshold
        Nl = len(indl)
        # remove these samples from u
        u[indl] = 1.1
        # set all labels 0 to nl to i
        labels[indl] = l
        # generate values for x
        x[:, indl] = np.random.multivariate_normal(means[l], covs[l], Nl).T

    return (x, labels)

def writeData(C, N, fname):
    """Generate the real dataset for X and write to file"""
    # C is the number of distinct covarience matrices
    # create means by evenly spacing on a line
    means = np.array([np.linspace(-(C),C,C), np.zeros(C)]).T
    # evenly distribute them
    priors = np.ones(C)/C
    # C cov arrays that are all 
    covs = np.zeros((C, d, d))
    for i in range(C):
        covs[i,:,:] = np.eye(d) * random.uniform(0.1, 0.5) + (np.ones((d,d)) - np.eye(d)) * random.uniform(-1, 1)


    x, labels = dataFromGMM(N, priors, means, covs)

    plt.scatter(x[0], x[1], c=labels)
    plt.show()

def readData():
    """Read the generated datasets from file"""
    pass

writeData(6, 500, "")
