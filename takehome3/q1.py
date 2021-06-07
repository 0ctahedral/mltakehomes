import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as ml

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

def evalGaussianPDF(x, mu, Sigma):
    N = x[1].size # number of items in x
    # normalization constant (const with respect to x)
    C = (2*np.pi)**(-2/2)*np.linalg.det(Sigma)**(1/2)
    #E = -0.5 * np.sum((x-ml.repmat(mu,1,N)).T * (np.linagl.inv(Sigma)* (x-ml.repmat(mu,1,N))),1)
    like = np.zeros(N)
    for i in range(N):
        like[i]  = C * np.exp(-0.5 * np.sum(((x[:, i]-mu).T * np.linalg.inv(Sigma)) * (x[:, i]-mu)))
    return like

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

def optimalClassifier(x, labels):
    N = x.shape[1]
    pxgivenl = np.zeros((C, N))
    priors = np.array([[0.25, 0.25, 0.25, 0.25]])

    for l in range(C):
        pxgivenl[l,:] = 0.5 * evalGaussianPDF(x,means[l][0], covs[l][0]) + 0.5 * evalGaussianPDF(x,means[l][1], covs[l][1]) 

    px = (priors * pxgivenl[:].T).T.sum(axis=0)
    classPosteriors = pxgivenl * ml.repmat(priors.T, 1, N) / ml.repmat(px, C, 1)
    expectedRisks = np.matmul(np.ones(C)-np.eye(C), classPosteriors)

    decisions = np.argmin(expectedRisks, axis=0)
    pErr = np.not_equal(decisions, testLabels).nonzero()[0].size/testLabels.size

    return decisions, pErr

train, trainLabels, test, testLabels = writeData()
#decisions, optimalPerr = optimalClassifier(test, testLabels)
