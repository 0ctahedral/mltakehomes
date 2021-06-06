import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
n = 3
markers = "+x*"
N = 10000
fig1 = plt.figure()
p1 = fig1.add_subplot(projection='3d')
p1.set_title("Actual distribution of samples")

m0 = np.array([2, 2, 0])
C0 = np.array( [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

m1 = np.array([-2, 2, 3])
C1 = np.eye(3)

m21 = np.array([3, 0, 2])
C21 = np.array( [[2, 0, 0],
                 [0, 2, 0],
                 [0, 0, 1]])

m22 = np.array([0, 3, 1])
C22 = np.array( [[2, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

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


def genData(N, priors):
    C = priors.size
    # make labels random
    u = np.random.rand(N)
    cumsum = np.cumsum(priors)
    x = np.zeros((n, N))
    labels = np.zeros(N)

    for l in range(C):
        # get idx of all u that are less than sum of probs at label l
        indl = np.where(u <= cumsum[l])[0]
        # set the labels
        labels[indl] = l
        # make those values in u too high
        u[indl] = 1.1
        if l == 0:
            N0 = indl.size
            y = np.random.multivariate_normal(m0, C0, N0).T
            x[:, indl] = y
            p1.scatter(y[0], y[1], y[2], marker=markers[l])
        elif l == 1:
            N1 = indl.size
            y = np.random.multivariate_normal(m1, C1, N1).T
            x[:, indl] = y
            p1.scatter(y[0], y[1], y[2], marker=markers[l])
        elif l == 2:
            N2 = indl.size
            weights = np.array([0.5, 0.5])
            means = np.array([m21, m22])
            covs = np.array([C21, C22])
            # do the gmm
            z, zlabel = dataFromGMM(N2, weights, means, covs)
            x[:, indl] = z
            p1.scatter(z[0], z[1], z[2], marker=markers[l])

    return x, labels

def evalGaussianPDF(x, mu, Sigma):
    #N = x[1].size # number of items in x
    # normalization constant (const with respect to x)
    C = (2*np.pi)**(-n/2)*np.linalg.det(Sigma)**(1/2)
    like = np.zeros(N)
    for i in range(N):
        like[i]  = C * np.exp(-0.5 * np.sum(((x[:, i]-mu).T * np.linalg.inv(Sigma)) * (x[:, i]-mu)))

    return like

priors = np.array([[0.3, 0.3, 0.4]])
x, labels = genData(N, priors)
# divide x into the three labels

# okay estimate it with MAP
pxgivenl = np.zeros((3, N))
# first two are single gaussians 
pxgivenl[0,:] = evalGaussianPDF(x, m0, C0)
pxgivenl[1,:] = evalGaussianPDF(x, m1, C1)
pxgivenl[2,:] = 0.5 * evalGaussianPDF(x, m21, C21) + 0.5 * evalGaussianPDF(x, m22, C22)

px = (priors * pxgivenl[:].T).T.sum(axis=0)
classPosteriors = pxgivenl * ml.repmat(priors.T, 1, N) / ml.repmat(px, 3, 1)

# loss matrix for MAP
lossMAP = np.eye(3)
MAPexpectedRisks = np.matmul(lossMAP, classPosteriors)
MAPdecisions = np.array([np.argmin(e) for e in MAPexpectedRisks.T])
MAPcorrect = np.where(np.equal(labels, MAPdecisions), "#00ff00", "#ff0000")
MAPconfusion = confusion_matrix(labels, MAPdecisions, normalize='true')
print(f"map:\n{MAPconfusion}")

# Decisions with
loss10 = np.array([
    [0,   1,   10],
    [1,   0,   10],
    [1,   1,   0],
    ])
expectedRisks10 = np.matmul(loss10, classPosteriors)
decisions10 = np.array([np.argmin(e) for e in expectedRisks10.T])
correct10 = np.where(np.equal(labels, decisions10), "#00ff00", "#ff0000")
loss10confusion = confusion_matrix(labels, decisions10, normalize='true')
print(f"10:\n{loss10confusion}")

loss100 = np.array([
    [0,   1,   100],
    [1,   0,   100],
    [1,   1,   0],
    ])
expectedRisks100 = np.matmul(loss100, classPosteriors)
decisions100 = np.array([np.argmin(e) for e in expectedRisks100.T])
correct100 = np.where(np.equal(labels, decisions100), "#00ff00", "#ff0000")
loss100confusion = confusion_matrix(labels, decisions100, normalize='true')
print(f"100:\n{loss100confusion}")

fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
erm1 = fig2.add_subplot(projection='3d')
erm1.set_title("MAP")
erm2 = fig3.add_subplot(projection='3d')
erm2.set_title("Loss with 10")
erm3 = fig4.add_subplot(projection='3d')
erm3.set_title("Loss with 100")

for l in range(3):
    # break up into true labels
    indl = np.where(labels == l)[0]
    # plot shape and color
    erm1.scatter(x[0,indl], x[1,indl], x[2,indl], c=MAPcorrect[indl], marker=markers[l])
    erm2.scatter(x[0,indl], x[1,indl], x[2,indl], c=correct10[indl], marker=markers[l])
    erm3.scatter(x[0,indl], x[1,indl], x[2,indl], c=correct100[indl], marker=markers[l])


# get array of if the label is correct
plt.show()
