import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

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

    data = np.array([radius * np.cos(angle), radius * np.sin(angle)]);
    
    return data, labels
traind, trainl = generateMultiring(2, 1000)
testd, testl = generateMultiring(2, 1000)
