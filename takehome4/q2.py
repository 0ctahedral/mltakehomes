import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import KFold
#
#
#if 1
#    colors = rand(C,3);
#    figure(1), clf,
#    for l = 1:C
#        ind_l = find(labels==l);
#        plot(data(1,ind_l),data(2,ind_l),'.','MarkerFaceColor',colors(l,:)); axis equal, hold on,
#    end
#end
def generateMultiring(C, N):
    """Generate C classes with a total of N samples"""
    # Randomly determine class labels for each sample
    thr = np.linspace(0,1,C+1); # split [0,1] into C equal length intervals
    u = np.random.rand(1,N); # generate N samples uniformly random in [0,1]
    labels = np.zeros((1,N));
    for l in range(1,C):
        ind_l = np.logical_and(thr[l]<u, u<=thr[l+1]);
        labels[ind_l] = l

    # parameters of the Gamma pdf needed later
    a = np.array([range(1,C+1)]).T * 2.5
    b = np.matlib.repmat(1.7,1,C).T
    # Generate data from appropriate rings
    # radius is drawn from Gamma(a,b), angle is uniform in [0,2pi]
    angle = 2*np.pi*np.random.rand(1,N)
    # reserve space
    radius = np.zeros((1,N))
    for l in range(C):
        ind_l = np.where(labels==l)[1]
        radius[:, ind_l] = np.random.gamma(a[l],b[l],(1,ind_l.shape[0]))

    data = np.array([radius * np.cos(angle), radius * np.sin(angle)]).reshape(C,N);
    
    return data, labels
traind, trainl = generateMultiring(2, 1000)
testd, testl = generateMultiring(2, 10000)

# setup possible values for C and sigma
C = np.logspace(-2, 2, 5)
sigma = np.logspace(-2, 2, 5)
# setup parameter dictionary and kfold
params = dict(C=C, gamma=sigma)
cv = KFold(n_splits=10)

# grid search using the kfold
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=params, cv=cv, scoring='accuracy')
grid.fit(traind.T, trainl.T)
params = grid.cv_results_['params']
means = grid.cv_results_['mean_test_score']
best = grid.best_params_

crossfig, crossplot = plt.subplots(1,1)
crossplot.set_title("Accuracy vs. Hyperparameter Values")
crossplot.plot(sigma, means[0:5])
crossplot.plot(sigma, means[5:10])
crossplot.plot(sigma, means[10:15])
crossplot.plot(sigma, means[15:20])
crossplot.plot(sigma, means[20:25])
crossplot.legend([f"C = {i}" for i in C], loc='right')
crossplot.set_xscale('log')
crossplot.set_xlabel('Value of Hyperparameter Sigma')
crossplot.set_ylabel('Validation Accuracy')

# train on whole set
final = SVC(kernel='rbf', C=best['C'], gamma=best['gamma'])
final.fit(traind.T, trainl.T)
# classify on test
pred = final.predict(testd.T)
cidx = np.where(pred==testl)[1]
widx = np.where(pred!=testl)[1]
# plot it all
finalfig, finalplot = plt.subplots(1,2)
finalplot[0].set_title("Class Predictions")
finalplot[0].scatter(testd[0,cidx], testd[1,cidx], marker='+', c=testl[:,cidx], label='correct')
finalplot[0].scatter(testd[0,widx], testd[1,widx], marker='x', c=testl[:,widx], label='incorrect')
finalplot[0].legend(loc='right')

finalplot[1].set_title("Samples by Correctness")
colors = np.where(np.equal(testl, pred), "#00ff00", "#ff0000").reshape(10000)
finalplot[1].scatter(testd[0], testd[1], c=colors, marker='+')

plt.show()
