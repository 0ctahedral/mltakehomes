import numpy as np
import numpy.matlib as ml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

m01 = np.array([3, 0]).T
C01 = np.array( [[2, 0],
                [0, 1]])

m02 = np.array([0, 3]).T
C02 = np.array( [[1, 0],
                [0, 2]])

m1 = np.array([2, 2])
C1 = np.eye(2)

# class priors
P_L0 = 0.65
P_L1 = 0.35

# consts
w1 = 0.5
w2 = 0.5
n = 2
C = 2

#N = 10000
N = 1000
lossmat = np.ones(C)-np.eye(C)

def genData(N):
    # make labels random
    labels = np.random.rand(1,N)
    x = np.zeros((n, N))
    # remap labels to be either 0 or 1
    labels = np.where(labels <= P_L0, 0, 1)
    # loop through and assign value using the label
    for l in range(0, 2):
        indl = np.where(labels == l)[1]
        # find all labels matching this one
        if l == 0:
            N0 = len(indl)
            weights = np.array([0.5, 0.5])
            means = np.array([m01, m02])
            covs = np.array([C01, C02])
            # do the gmm
            z, zlabel = dataFromGMM(N0, weights, means, covs)
            x[:, indl] = z
            #p1.scatter(z[0], z[1], c=zlabel, marker='o')
        elif l == 1:
            N1 = len(indl)
            y = np.random.multivariate_normal(m1, C1, N1).T
            x[:, indl] = y
            #p1.scatter(y[0], y[1], marker='+')

    return x, labels

def dataFromGMM(N, priors, means, covs):
    """ Generates N vector saamples from mixture of Gaussians """
    components = len(priors)
    x = np.zeros((n, N))
    labels = np.zeros(N)
    u = np.random.rand(N)
    thresholds = np.cumsum(priors)
    # go through all u and assign the labels to x
    for l in range(0, components):
        # find all u that are less than the threshold
        indl = np.where(u <= thresholds[l])[0]
        # track the number of samples below this threshold
        Nl = len(indl)
        # remove these samples from u
        u[indl] = 1.1
        # set all labels 0 to nl to i
        labels[indl] = l
        # generate values for x
        x[:, indl] = np.random.multivariate_normal(means[l], covs[l], Nl).T

    return (x, labels)


def evalGaussianPDF(x, mu, Sigma):
    N = x[1].size # number of items in x
    # normalization constant (const with respect to x)
    C = (2*np.pi)**(-n/2)*np.linalg.det(Sigma)**(1/2)
    #E = -0.5 * np.sum((x-ml.repmat(mu,1,N)).T * (np.linagl.inv(Sigma)* (x-ml.repmat(mu,1,N))),1)
    like = np.zeros(N)
    for i in range(N):
        like[i]  = C * np.exp(-0.5 * np.sum(((x[:, i]-mu).T * np.linalg.inv(Sigma)) * (x[:, i]-mu)))

    return like

def lda(x, labels):
    # separate the data by label
    x1 = x[:, np.flatnonzero(labels == 0)]
    x2 = x[:, np.flatnonzero(labels == 1)]
    # estimate the means and covariences
    mu1 = np.mean(x1)
    mu2 = np.mean(x2)
    S1 = np.cov(x1)
    S2 = np.cov(x2)
    # Sb and Sw
    Sb = (mu1-mu2)*(mu1-mu2).T
    Sw = (S1+S2)
    # get eigen values and vectors
    values,vectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    # find index of largest value
    idx = np.argmax(values)
    # w is the vector of that index
    w = vectors[idx]
    # 
    y1 = np.matmul(w.T, x1)
    y2 = np.matmul(w.T, x2)

    return y1, y2


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
        d = np.where(scores >= tau, 0, 1)
        pfp[i] = np.flatnonzero(np.logical_and(d == 1, labels == 0)).size / np.flatnonzero(labels==0).size
        ptp[i] = np.flatnonzero(np.logical_and(d == 1, labels == 1)).size / np.flatnonzero(labels==0).size
        perr[i] = np.flatnonzero(d != labels).size/labels.size

    return pfp, ptp, perr, thresholds

# generate data
x, labels = genData(N)


# evaluate gaussians!
pxgivenl = np.zeros((C, N))
pxgivenl[0,:] = w1 * evalGaussianPDF(x,m01, C01) + w2 * evalGaussianPDF(x,m02, C02)
pxgivenl[1,:] = evalGaussianPDF(x,m1, C1)


priors = np.array([[P_L0, P_L1]])
px = (priors * pxgivenl[:].T).T.sum(axis=0)
classPosteriors = pxgivenl * ml.repmat(priors.T, 1, N) / ml.repmat(px, C, 1)
threshold = 0.1;
descriminant_ratios = (classPosteriors[1,:]/classPosteriors[0,:])

expectedRisks = np.matmul(lossmat, classPosteriors)
ERMdecisions = np.array([np.argmin(e) for e in expectedRisks.T])

# confusion matrix
confusionMat = np.zeros((C,C))
for d in range(C):
    for l in range(C):
        dl = np.flatnonzero(np.logical_and(ERMdecisions == d, labels == l)).size
        confusionMat[d, l] = dl/np.flatnonzero(labels==l).size


y1, y2 = lda(x, labels)
ally = np.concatenate((y1, y2))
# get roc curves
ERMpfp, ERMptp, ERMperr, ERMthresholds = rocCurve(descriminant_ratios, labels)

# get index of min perr and use it for threshold

LDApfp, LDAptp, LDAperr, LDAthresholds = rocCurve(ally, labels)

LDAdecisions = np.where(ally >= -4, 0, 1)
ldaconfusionMat = np.zeros((C,C))
for d in range(C):
    for l in range(C):
        dl = np.flatnonzero(np.logical_and(LDAdecisions == d, labels == l)).size
        ldaconfusionMat[d, l] = dl/np.flatnonzero(labels==l).size

def plotAll():
    # figure showing the real and estimated data
#    fig, (p1, p2) = plt.subplots(1, 2)
#    fig.suptitle("Comparison of real class labels and best estimate:")
#    p1.set_title("Real classification")
#    p1.scatter(x[0], x[1], c=labels, marker='+')
#    p2.set_title("Estimate using ERM")
#    p2.scatter(x[0], x[1], c=ERMdecisions, marker='+')

#    # figure with the ROC curve
#    fig1 = plt.figure()
#    f1plots = fig1.add_subplot()
#    f1plots.scatter(ERMpfp, ERMptp)
#    f1plots.set_title("ROC cuve for ERM")
#    # add confusion matrix shit
#    f1plots.scatter(confusionMat[0,0], confusionMat[0,1], marker='+')
#    plt.setp(f1plots, xlabel="P(false positive)")
#    plt.setp(f1plots, ylabel="P(true positive)")

    # add perror curve
  #  f1plots[1].set_title("P(error) for ERM by threshold")
  #  f1plots[1].scatter(ERMthresholds, ERMperr, marker='+')
  #  plt.setp(f1plots[1], xlabel="Threshold")
  #  plt.setp(f1plots[1], ylabel="P(error) for Descriminant ratios")

    # lda plots
    ldafig, ldaplots = plt.subplots(1,2)
    #ldaplots[0,0].set_title("Actual data")
    #ldaplots[0,0].scatter(x[0], x[1], c=labels, marker='+')
    ## lda projection 
    #ldaplots[0,0].axis('equal')
    #ldaplots[0,0].scatter(y1, np.zeros(y1.size))
    #ldaplots[0,0].scatter(y2, np.zeros(y2.size))

    #ldaplots[0,1].set_title("Reclassification with ideal threshold")
    #ldaplots[0,1].scatter(x[0], x[1], c=LDAdecisions, marker='+')

    ldaplots[0].set_title("LDA ROC curve")
    ldaplots[0].scatter(LDApfp, LDAptp)
    ldaplots[0].scatter(ldaconfusionMat[1,0], ldaconfusionMat[1,1])
    plt.setp(ldaplots[0], xlabel="P(false positive)")
    plt.setp(ldaplots[0], ylabel="P(true positive)")

    ldaplots[1].set_title("LDA Perror by threshold")
    ldaplots[1].scatter(LDAthresholds, LDAperr)
    plt.setp(ldaplots[1], xlabel="Threshold")
    plt.setp(ldaplots[1], ylabel="P(error)")

    plt.show()

plotAll()
