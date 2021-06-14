import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool

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
    ret = {}
    for p in P:
        ret[10**p] = (readSamples(10**p))
    return ret

#createData(100, range(2,7))
#allData = readAll(range(2,7))

def evalGaussianPDF(x, mu, Sigma):
    N = x[1].size # number of items in x
    # normalization constant (const with respect to x)
    C = (2*np.pi)**(-2/2)*np.linalg.det(Sigma)**(1/2)
    #E = -0.5 * np.sum((x-ml.repmat(mu,1,N)).T * (np.linagl.inv(Sigma)* (x-ml.repmat(mu,1,N))),1)
    like = np.zeros(N)
    for i in range(N):
        like[i]  = C * np.exp(-0.5 * np.sum(((x[:, i]-mu).T * np.linalg.inv(Sigma)) * (x[:, i]-mu)))

    return like

def bic(D, M):
    """Given a dataset (D) with many experiments and an order (M) return the BIC for the M-component GMM"""
    X, Labels = D

    results = []
    # do all 100 or whatever experiments
    for i in range(X.shape[0]):
        x = X[i]
        l = Labels[i]
        # test gaussian on all orders M
        scores = []
        for m in range(1, M):
            # create model with m components
            model = GaussianMixture(n_components=m, init_params='random').fit(x.T)

            # evaluate gaussians so we can add to the score
            L = 0
            for i in range(m):
                # get the probability of x in each and average
                L += (1/m * evalGaussianPDF(x, model.means_[i], model.covariances_[i]))

            # add score
            scores.append(m * np.log(x.shape[1]) - (2 * np.log(L.sum() / x.shape[1])))

        results.append(scores)
    return results

def splitData(data, labels, nFolds):
    """splits the given dataset into the number of folds required, with the label as the first array"""

    # combine to make easier
    dset = list(data.T)
    #ll = list(labels)
    # size of each split
    fsize = int(len(dset)/nFolds)

    # list of list of values
    splits = []
    slabels = []

    # segment by taking randomly from the dset and poping them
    for i in range(nFolds):
        fold = []
        #flabels = []
        while len(fold) < fsize:
            # random index
            idx = random.randrange(len(dset))
            # apend to fold
            fold.append(dset.pop(idx))
            #flabels.append(ll.pop(idx))
        # append fold to splits
        splits.append(fold)
        #slabels.append(flabels)

    # return as separate lists again
    #return splits, slabels
    return splits

def kfold(D, M, K):
    """Maximize the cross validation log likelihood and return the best M"""
    # partition dataset with K
    X, Labels = D
    #results = []
    #for j in range(X.shape[0]):
    pool = Pool(processes=100)
    results = pool.map(kfold_sub, [X[i] for i in range(X.shape[0])])
    pool.close()

    #    results.append(scores)
    return results

def kfold_sub(X):
    M = 10
    K = 5
    scores = []
    for m in range(1, M):
        subscores = []
        for i in range(K):
            fdata = splitData(X, np.zeros(100), K)
            # take the ith for tests
            ftest = np.asarray(fdata.pop(i))
            ftrain = np.asarray(fdata).T.reshape(d, (K-1)*ftest.shape[0])
            model = GaussianMixture(n_components=5, init_params='random', n_init=5).fit(ftrain.T)
            subscores.append(model.score(ftest))
        scores.append(sum(subscores)/K)
    return scores

# time to plot
def plot():
    kfoldResults = {}
    bicResults = {}
    mins =  {}
    mins['kfold'] = []
    mins['bic'] = []
    maxs =  {}
    maxs['kfold'] = []
    maxs['bic'] = []
    means =  {}
    means['kfold'] = []
    means['bic'] = []
    per = {}
    per['bic'] = np.zeros((6,5))
    per['kfold'] = np.zeros((6,5))
    percentiles = [10, 25, 50, 75, 90]
    # collect results
    for p in range(2,6):
        i = 10**p
        # get maximum
        kfoldResults[i] = np.argmax(np.loadtxt(f"results/kfold{i}.csv", delimiter=' '), axis=1) + 1
        mins["kfold"].append(np.min(kfoldResults[i]))
        maxs["kfold"].append(np.max(kfoldResults[i]))
        means["kfold"].append(np.mean(kfoldResults[i]))
        per["kfold"][p,:] = np.percentile(kfoldResults[i], [10, 25, 50, 75, 90])

        bicResults[i] = np.loadtxt(f"results/kfold{i}.csv", delimiter=' ')
        # change all negative inf to zero
        bicResults[i][bicResults[i] == -np.inf] = 0
        # get minimum
        bicResults[i] = np.argmin(bicResults[i], axis=1) + 1
        mins["bic"].append(np.min(bicResults[i]))
        maxs["bic"].append(np.max(bicResults[i]))
        means["bic"].append(np.mean(bicResults[i]))
        per["bic"][p,:] = np.percentile(bicResults[i], percentiles)

    # min median max M for group of N samples
    bicfig, bicplots = plt.subplots(1,1)
    bicplots.set_title("Percential and mean M for each N samples in BIC")
    bicplots.plot(list(bicResults.keys()), means["bic"], label='mean')

    kfoldfig, kfoldplots = plt.subplots(1,1)
    kfoldplots.set_title("Percential and mean M for each N samples in kfold")
    kfoldplots.plot(list(kfoldResults.keys()), means["kfold"], label='mean')

    for i in range(len(percentiles)):
        bicplots.plot(list(bicResults.keys()), per["bic"][i, 0:4], label=percentiles[i])
        kfoldplots.plot(list(kfoldResults.keys()), per["kfold"][i, 0:4], label=percentiles[i])

    kfoldplots.legend(loc='right')
    bicplots.legend(loc='right')
    
    plt.show()
plot()
#for p in range(3,7):
#    i = 10**p
#    print(f"kfold for {i}")
#    np.savetxt(f"results/kfold{i}.csv", np.asarray(kfold(allData[i], 10, 5)))

#for p in range(2,7):
#    i = 10**p
#    print(f"bic for {i}")
#    np.savetxt(f"results/bic{i}.csv", np.asarray(bic(allData[i], 11)))
