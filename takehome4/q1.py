from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

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
                layers.Dense(units = P, activation='sigmoid', kernel_initializer = 'random_uniform', input_dim = 1),
                #layers.Dense(units = 1, activation='softmax', kernel_initializer = 'random_uniform', name = 'soft')
            ]
    )
    model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

def tenFold(x, y):
    """
    Perform ten fold cross-validation with the given dataset
    and return the MSE for each run of the model
    """
    pass

def applyModel(x, y, M):
    """Applys the given model M to a validation set x and y"""
    pass

# generate data
trainx, trainy = generateData(1000)
testx, testy = generateData(1000)

# optimize params with 10fold cross validation
model = makeModel(3)
model.fit(trainx.T, trainy.T, batch_size = 10, epochs = 100, verbose=0)

results = model.predict(testx.T)

# train mlp with optimal set of perceptrons

# apply trained mlp to tes set and visualize
