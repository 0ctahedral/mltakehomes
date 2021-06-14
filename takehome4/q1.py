import matplotlib.pyplot as plt
import numpy as np

def generateData(N):
    # a, b, dims
    x = np.random.gamma(3, 2, (1,N))
    # 
    z = np.exp((x**2) * np.exp(-x/2));
    # mu, sigma, dims
    v = np.random.lognormal(0,0.1,(1,N));
    y = v*z;
    return x, y

x, y = generateData(100)

plt.scatter(x, y)
plt.show()
