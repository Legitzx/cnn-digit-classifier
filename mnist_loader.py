import numpy as np
from urllib import request
import gzip
import pickle

"""
Ref:
https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
"""

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    init()

    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def get_mnist_data():
  X_train, y_train, x_test, y_test = load()

  trainingInput = [x.reshape(1, 28, 28) / 255.0 for x in X_train]
  trainingOutput = [vectorized_result(y) for y in y_train]

  testingInput = [x.reshape(1, 28, 28) / 255.0 for x in x_test]
  testingOutput = [vectorized_result(y) for y in y_test]

  return trainingInput, trainingOutput, testingInput, testingOutput

def vectorized_result(j):
  """Return a 10-dimensional unit vector with a 1.0 in the jth
  position and zeroes elsewhere.  This is used to convert a digit
  (0...9) into a corresponding desired output from the neural
  network."""
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

