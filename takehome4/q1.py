from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

# setup kfold
kf = KFold(n_splits=10)

def generateData(N):
    """Generate the dataset of size N used for this experiment"""
    # a, b, dims
    x = np.random.gamma(3, 2, (1,N))
    # 
    z = np.exp((x**2) * np.exp(-x/2));
    # mu, sigma, dims
    v = np.random.lognormal(0,0.1,(1,N));
    y = v*z;
    return x, y

def makeModel(P):
    """Creates and returns a model with P percpetrons in the hidden layer"""
    model = keras.Sequential(
            [
                #layers.Dense(units = P, activation = 'softplus', input_dim = 1),
                layers.Dense(units = P, activation = 'sigmoid', input_dim = 1),
                layers.Dense(units = 1)
            ]
    )
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

    return model

def tenFold(x, y, r):
    """
    Perform ten fold cross-validation with the given dataset
    and return the MSE for each run of the model.
    r -- number of perceptrons
    """
    # make this go in separate threads
    pool = Pool(processes=len(r))
    results = pool.map(subfold, [(n, x, y) for n in r])
    pool.close()

    return np.mean(np.array(results), axis=1)

def subfold(arg):
    # get the variables
    n, x, y = arg
    errors = []
    # make model
    model = makeModel(n)
    
    # split dataset
    for train_idx, test_idx in kf.split(x.T):
        # get test and train
        trainx = x.T[train_idx]
        trainy = y.T[train_idx]
        testx = x.T[test_idx]
        testy = y.T[test_idx]
        # train model on this split
        model.fit(trainx, trainy, batch_size = 10, epochs = 100, verbose=0)
        # get results on test split
        res = model.predict(testx)
        # calculate mse
        # add to list of mse
        mse = np.mean(pow((testy - res), 2))
        errors.append(mse)

    return errors

def applyModel(x, y, M):
    """Applys the given model M to a validation set x and y"""
    pass

# generate data
trainx, trainy = generateData(1000)
testx, testy = generateData(1000)

# optimize params with 10fold cross validation
perceptrons = np.arange(1,16)
results = tenFold(testx, testy, perceptrons)

# find the best performing one number of perceptrons
minidx = np.argmin(results)
# train mlp with optimal set of perceptrons
model = makeModel(perceptrons[minidx])
model.fit(trainx.T, trainy.T, batch_size = 10, epochs = 100, verbose=0)
# apply trained mlp to test set and visualize
pred_y = model.predict(testx.T)

fig, plots = plt.subplots(1,1)
plots.scatter(testx, testy, label="test")
plots.scatter(testx, pred_y, label=f"model with {perceptrons[minidx]} perceptrons")
plots.legend(loc='right')
plt.show()
