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

def writeSamples(C, N, E):
    """Generate the real dataset for X and write to file"""
    x = np.zeros((E, d, N))
    labels = np.zeros((E, N))
    for e in range(E):
        # C is the number of distinct covarience matrices
        # create means by evenly spacing on a line
        means = np.array([np.linspace(-(C),C,C), np.zeros(C)]).T
        # evenly distribute them
        priors = np.ones(C)/C
        # C cov arrays that are all 
        covs = np.zeros((C, d, d))
        for i in range(C):
            covs[i,:,:] = np.eye(d) * random.uniform(0.1, 0.5) + (np.ones((d,d)) - np.eye(d)) * random.uniform(-1, 1)

        x[e, :, :], labels[e, :] = dataFromGMM(N, priors, means, covs)

    np.save(f"data/{N}samples_set", x)
    np.save(f"data/{N}samples_labels", labels)

def readSamples(N):
    """Read the generated datasets from file"""
    x = np.load(f"data/{N}samples_set.npy")
    labels = np.load(f"data/{N}samples_labels.npy")
    return x, labels

def createData(E, P):
    """Creates all datasets and writes them to the data folder
        E -- number of experiments
        P -- range of powers to apply to 10 for the number of samples
    """
    for p in P:
        writeSamples(6, 10**p, E)

def readAll(P):
    ret = []
    for p in P:
        ret.append(readSamples(10**p))
    return ret

#createData(100, range(2,7))
allData = readAll(range(2,7))
