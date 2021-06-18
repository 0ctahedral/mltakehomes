import numpy as np
from tensorflow import keras
import keras.preprocessing.image as image
from sklearn.preprocessing import normalize

def img_to5d(path):
    """
    generate 5-D vector for pixels
      row
      col
      red
      green
      blue
    then normalize to 0-1
    """

    img = image.load_img(path)
    arr = image.img_to_array(img)
    ret = np.zeros((arr.shape[0] * arr.shape[1], 5))

    # index into the hyperarray
    i = 0
    # fill the hyperarray
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            ret[i, :] = np.array(
                    [
                        row, col,
                        arr[row,col,0],
                        arr[row,col,1],
                        arr[row,col,2]
                    ]
            )
            i += 1
    return normalize(ret, axis=1, norm='max')

bird = img_to5d("42049_color.jpg")
plane = img_to5d("3096_color.jpg")


# fit 2 component gmm using ml param estimation

# use 10fold cross val to find best number of components
# based on max aberage validation log likelihood

# use this number of components to classify again
