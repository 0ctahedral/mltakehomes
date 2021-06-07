import matplotlib.pyplot as plt
import numpy as np

# cube from https://stackoverflow.com/questions/52229300/creating-numpy-array-with-coordinates-of-vertices-of-n-dimensional-cube
cube = 2*((np.arange(2**3)[:,None] & (1 << np.arange(3))) > 0) - 1

# generate data
# four classes
C = 4
# 3 dimensional
d = 3

means = np.array(
        [
            [cube[0], cube[1]], # class 0
            [cube[3], cube[5]], # class 1
            [cube[4], cube[2]], # class 2
            [cube[6], cube[7]], # class 3
        ]
)

covs = np.array(
        [
            [0.5 * np.eye(3), 0.45 * np.eye(3)], # class 0
            [0.8 * np.eye(3), 0.67 * np.eye(3)], # class 1
            [0.6 * np.eye(3), 0.72 * np.eye(3)], # class 2
            [0.7 * np.eye(3), 0.81 * np.eye(3)], # class 3
            
        ]
)

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

def genData(N):
    # uniform priors so we can skip that and just make a uniform
    # cumulative sum
    cumsum = np.array([0.25, 0.5, 0.75, 1.0])

    # random vector for assigning labels from
    u = np.random.rand(N)

    # empty label and data matrices
    labels = np.zeros(N)
    x = np.zeros((d, N))

    # even 50/50 mixture in the gaussian
    weights = np.array([0.5, 0.5])

    # do the same thing for each class
    for l in range(C):
        indl = np.where(u <= cumsum[l])[0]
        # set the labels
        labels[indl] = l
        # make those values in u too high so we don't reuse them
        u[indl] = 1.1
        z, zlabel = dataFromGMM(indl.size, weights, means[l], covs[l])
        x[:, indl] = z


    return x, labels

def writeData():
    """Generate datasets and write them to files"""
    # sets
    train = {}
    trainLabels = {}
    train[100],  trainLabels[100] = genData(100)
    train[200],  trainLabels[200] = genData(200)
    train[500],  trainLabels[500] = genData(500)
    train[1000], trainLabels[1000]= genData(1000)
    train[2000], trainLabels[2000]= genData(2000)
    train[5000], trainLabels[5000]= genData(5000)
    test, testLabels = genData(100000)
    #dataset = np.c_[labels, x.T]
    #numpy.savetxt("500.csv", dataset, delimiter=",")
    return train, trainLabels, test, testLabels

def loadData():
    pass

train, trainLabels, test, testLabels = writeData()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(train[2000][0], train[2000][1], train[2000][2], c=trainLabels[2000])
plt.show()
