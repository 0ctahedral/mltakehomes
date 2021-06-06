import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
from scipy.optimize import minimize

priors = np.array([[0.65, 0.35]])

m01 = np.array([3, 0])
C01 = np.array([
    [2,0],
    [0,1]
])
m02 = np.array([0, 3])
C02 = np.array([
    [1,0],
    [0,2]
])

m1 = np.array([2, 2])
C1 = np.array([
    [1,0],
    [0,1]
])

def dataFromGMM(N, priors, means, covs):
    components = priors.size
    x = np.zeros((2, N))
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

def genData(N, priors):
    u = np.random.rand(N)
    cumsum = np.cumsum(priors)
    x = np.zeros((2, N))
    labels = np.zeros(N)
    for l in range(2):
        indl = np.where(u <= cumsum[l])[0]
        # set the labels
        labels[indl] = l
        # make those values in u too high
        u[indl] = 1.1

        if l == 1:
            N1 = indl.size
            y = np.random.multivariate_normal(m1, C1, N1).T
            x[:, indl] = y
        elif l == 0:
            N2 = indl.size
            weights = np.array([0.5, 0.5])
            means = np.array([m01, m02])
            covs = np.array([C01, C02])
            # do the gmm
            z, zlabel = dataFromGMM(N2, weights, means, covs)
            x[:, indl] = z

    return x, labels

def evalGaussianPDF(x, mu, Sigma):
    N = x[1].size # number of items in x
    # normalization constant (const with respect to x)
    C = (2*np.pi)**(-2/2)*np.linalg.det(Sigma)**(1/2)
    #E = -0.5 * np.sum((x-ml.repmat(mu,1,N)).T * (np.linagl.inv(Sigma)* (x-ml.repmat(mu,1,N))),1)
    like = np.zeros(N)
    for i in range(N):
        like[i]  = C * np.exp(-0.5 * np.sum(((x[:, i]-mu).T * np.linalg.inv(Sigma)) * (x[:, i]-mu)))

    return like
    
def rocCurve(scores,labels):
    # thresholds start with the smallest ratio - eps, all in between, the max + eps
    thresholds = np.array(np.sort(scores))
    nt = thresholds.size
    pfp = np.zeros(nt)
    ptp = np.zeros(nt)
    perr = np.zeros(nt)

    for i in range(nt):
        tau = thresholds[i]
        # map all decisions to labels given the current tau
        #d = np.where(scores >= tau, 0, 1)
        d = np.greater_equal(scores, tau).astype(int)
        true_positive = np.equal(d, 1) & np.equal(labels, 1)
        true_negative = np.equal(d, 0) & np.equal(labels, 0)
        false_positive = np.equal(d, 1) & np.equal(labels, 0)
        false_negative = np.equal(d, 0) & np.equal(labels, 1)
        ptp[i] = true_positive.sum() / (true_positive.sum() + false_negative.sum())
        pfp[i] = false_positive.sum() / (false_positive.sum() + true_negative.sum())
        perr[i] = np.not_equal(d, labels).nonzero()[0].size/labels.size

    return pfp, ptp, perr, thresholds


# get the training and validation data
d20, labels20 = genData(20, priors)
d200, labels200 = genData(200, priors)
d2000, labels2000 = genData(2000, priors)
validate, validate_labels = genData(10000, priors)

# classify and show ROC, and min perror
def part1(v, v_labels):
    pxgivenl = np.zeros((2, 10000))
    pxgivenl[0,:] = 0.5 * evalGaussianPDF(v,m01, C01) + 0.5 * evalGaussianPDF(v,m02, C02)
    pxgivenl[1,:] = evalGaussianPDF(v,m1, C1)
    px = (priors * pxgivenl[:].T).T.sum(axis=0)
    classPosteriors = pxgivenl * ml.repmat(priors.T, 1, 10000) / ml.repmat(px, 2, 1)
    expectedRisks = np.matmul(np.ones(2)-np.eye(2), classPosteriors)
    ratios = (classPosteriors[1,:]/classPosteriors[0,:])

    pfp, ptp, perr, thresholds = rocCurve(ratios, v_labels)

    part1 = plt.figure()
    p1plots = part1.add_subplot()
    p1plots.set_title("roc curve")
    plt.setp(p1plots, xlabel="p(false positive)", ylabel="p(true positive)")
    p1plots.scatter(pfp, ptp)
    # find the lowest p error
    minErrIdx = np.argmin(perr)
    p1plots.scatter(pfp[minErrIdx], ptp[minErrIdx], color="#ff0000")
    p1plots.annotate(f"Min P(Error): {perr[minErrIdx]}, threshold: {thresholds[minErrIdx]}", xy=(pfp[minErrIdx], ptp[minErrIdx]))

    comparison, cplots = plt.subplots(1,2)
    # the actual data
    cplots[0].scatter(validate[0], validate[1], c=v_labels)
    cplots[0].set_title("Actual Data")
    # data with the threshold we just used
    decisions = np.where(ratios >= thresholds[minErrIdx], 1, 0)
    cplots[1].scatter(validate[0], validate[1], c=decisions)
    cplots[1].set_title("Data With min P(error) threshold")

    plt.show()

def part2_linear():
    fig, lall = plt.subplots(1, 3)

    # train three log near linear
    w20 = minimize(linCost, np.zeros(3), d20, method='Nelder-Mead').x
    decisions20 = np.zeros((1,10000))
    z = np.c_[np.ones((validate.shape[1])), validate.T].T
    h = 1/(1+np.exp(-(np.dot(w20.T,z))))
    decisions20[0,:] = (h[:]>=0.5).astype(int)
    lall[0].set_title("Linear With Training Set of 20")
    lall[0].scatter(validate[0], validate[1], c=decisions20)

    w200 = minimize(linCost, np.zeros(3), d200, method='Nelder-Mead').x
    decisions200 = np.zeros((1,10000))
    z = np.c_[np.ones((validate.shape[1])), validate.T].T
    h = 1/(1+np.exp(-(np.dot(w200.T,z))))
    decisions200[0,:] = (h[:]>=0.5).astype(int)
    lall[1].set_title("Linear With Training Set of 200")
    lall[1].scatter(validate[0], validate[1], c=decisions200)

    w2000 = minimize(linCost, np.zeros(3), d2000, method='Nelder-Mead').x
    decisions2000 = np.zeros((1,10000))
    z = np.c_[np.ones((validate.shape[1])), validate.T].T
    h = 1/(1+np.exp(-(np.dot(w2000.T,z))))
    decisions2000[0,:] = (h[:]>=0.5).astype(int)
    lall[2].set_title("Linear With Training Set of 2000")
    lall[2].scatter(validate[0], validate[1], c=decisions2000)


    plt.show()

def part2_quad():
    fig, lall = plt.subplots(1, 3)

    w20 = minimize(quadCost, np.zeros(6), d20, method='Nelder-Mead').x
    decisions20 = np.zeros((1,10000))
    z = np.c_[np.ones((validate.shape[1])), validate[0], validate[1], validate[0]*validate[0], validate[0]*validate[1], validate[1]*validate[1]].T
    h = 1/(1+np.exp(-(np.dot(w20.T,z))))
    decisions20[0,:] = (h[:]>=0.5).astype(int)
    lall[0].set_title("Quadratic With Training Set of 20")
    lall[0].scatter(validate[0], validate[1], c=decisions20)

    w200 = minimize(quadCost, np.zeros(6), d200, method='Nelder-Mead').x
    decisions200 = np.zeros((1,10000))
    z = np.c_[np.ones((validate.shape[1])), validate[0], validate[1], validate[0]*validate[0], validate[0]*validate[1], validate[1]*validate[1]].T
    h = 1/(1+np.exp(-(np.dot(w200.T,z))))
    decisions200[0,:] = (h[:]>=0.5).astype(int)
    lall[1].set_title("Quadratic With Training Set of 200")
    lall[1].scatter(validate[0], validate[1], c=decisions200)

    w2000 = minimize(quadCost, np.zeros(6), d2000, method='Nelder-Mead').x
    decisions2000 = np.zeros((1,10000))
    z = np.c_[np.ones((validate.shape[1])), validate[0], validate[1], validate[0]*validate[0], validate[0]*validate[1], validate[1]*validate[1]].T
    h = 1/(1+np.exp(-(np.dot(w2000.T,z))))
    decisions2000[0,:] = (h[:]>=0.5).astype(int)
    lall[2].set_title("Quadratic With Training Set of 2000")
    lall[2].scatter(validate[0], validate[1], c=decisions2000)

    plt.show()

def linCost(w, x):
    z = np.c_[np.ones((x.shape[1])), x.T].T
    denom = 1 + np.exp(-np.dot(w.T , z)).sum()
    return 1/denom

def quadCost(w, x):
    z = np.c_[np.ones((x.shape[1])), x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[1]].T
    denom = 1 + np.exp(-np.dot(w.T , z)).sum()
    return 1/denom

part1(validate, validate_labels)

part2_linear()

part2_quad()
