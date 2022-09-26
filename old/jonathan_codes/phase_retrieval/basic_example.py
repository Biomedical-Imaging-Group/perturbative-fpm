import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from time import time
from tqdm import tqdm

from models.random_model import random_sampling
from algos.gradient_descent import gradient_descent
from utils.eval import eval_vector, eval_matrix

n = 2000
d = 100
(y, A, x) = random_sampling(n=n, d=d, verbose=1)

x0 = np.random.randn(d, 1) / np.sqrt(100)
x1 = gradient_descent(y, A, x0, verbose=1)

print("The first number is the correlation (close to 1 is good, it means that our estimate is correlated x1 with the solution x).")
print("The second number is the loss function (close to 0 is good, it means that the loss is very low).")
print(eval_vector(A, x1, x))