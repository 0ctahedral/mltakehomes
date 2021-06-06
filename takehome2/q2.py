import numpy as np
import matplotlib.pyplot as plt

def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    #plot3(data[0,:],data[1,:],data[2,:])
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    
    Ntrain = 1000
    data = generateData(Ntrain)
    #plot3(data[0,:],data[1,:],data[2,:])
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    
    return xTrain,yTrain,xValidate,yValidate

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3] # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0] # Data dimensionality
    C = len(priors) # Number of components
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))
        
    return x,labels

def plot3(a,b,c,mark="o",col="b"):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(a, b, c,marker=mark,color=col)
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("y")
  ax.set_title('Training Dataset')

# rate each w by using the validation set
def create_y(x, w):
    return w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[0]*x[0] + w[4]*x[0]*x[1] + w[5]*x[1]*x[1]

xTrain,yTrain,xValidate,yValidate = hw2q2()

# assume that varience is 0.1, lambda i


# make z
Z = np.c_[np.ones((xTrain.shape[1])), xTrain[0], xTrain[1], xTrain[0]*xTrain[0], xTrain[0]*xTrain[1], xTrain[1]*xTrain[1]].T
R = np.zeros((6,6))
for i in range(Z.shape[1]):
    R += Z[:, i]*Z[:,i].T
R = R/Z.shape[1]

q = np.zeros((1,6))
for i in range(Z.shape[1]):
    q += yTrain[i]*Z[:,i]
q = q/Z.shape[1]
# range of gamma values
gammaList = np.logspace(-7, 4, num=50)

# estimate w
w_map = np.zeros((gammaList.size, 6))
w_ml = np.zeros((gammaList.size, 6,))
# average l2 norm of this model for each gamma
l2_ml = np.zeros((1, gammaList.size))
l2_map = np.zeros((1, gammaList.size))

for i in range(gammaList.size):
    gamma = gammaList[i]
    w_map[i] = np.dot(np.linalg.inv(R + np.eye(6)), q.T).T
    w_ml[i] = np.dot(np.linalg.inv(R + gamma*np.eye(6)), q.T).T

    # for all gamma, create y and evaluate the average squared error btwn generated and validation y
    mly = create_y(xValidate, w_ml[i])
    l2_ml[:,i] = pow(np.linalg.norm(yValidate - mly), 2)

    mapy = create_y(xValidate, w_map[i])
    l2_map[:,i] = pow(np.linalg.norm(yValidate - mapy), 2)


mlfig = plt.figure()
ml_plot = mlfig.add_subplot()
ml_plot.set_title("Results with Maximum Likelihood Estimator")
ml_plot.plot(gammaList, l2_ml[0])
ml_plot.set_yscale('log')
ml_plot.set_xscale('log')
plt.setp(ml_plot, xlabel="Gamma", ylabel="Average L2 norm squared error")

mapfig = plt.figure()
map_plot = mapfig.add_subplot()
map_plot.set_title("Results with MAP Estimator")
map_plot.plot(gammaList, l2_map[0])
map_plot.set_yscale('log')
map_plot.set_xscale('log')
plt.setp(map_plot, xlabel="Gamma", ylabel="Average L2 norm squared error")

allfig = plt.figure()
actual = allfig.add_subplot(1,3,1, projection='3d')
ml_data = allfig.add_subplot(1,3,2, projection='3d')
map_data = allfig.add_subplot(1,3,3, projection='3d')

actual.set_title("Actual data")
actual.scatter(xValidate[0], xValidate[1], yValidate)
ml_data.set_title("Using ML Estimator")
ml_data.scatter(xValidate[0], xValidate[1], mly)
map_data.set_title("Using MAP Estimator")
map_data.scatter(xValidate[0], xValidate[1], mapy)

plt.show()
