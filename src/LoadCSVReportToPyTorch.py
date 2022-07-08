import csv
import numpy as np
import torch

path = 'results_only_buy_to_open.csv'

#results_numpy = np.genfromtxt(path, dtype = [('col{}'.format(i+1), float) for i in range(20)], delimiter=",", missing_values="", filling_values=0)

results_numpy = np.genfromtxt(path, dtype=np.float32, delimiter=",", missing_values="", filling_values=0)


#print(results_numpy)

results_pytorch = torch.from_numpy(results_numpy)

print(results_pytorch)

print(type(results_pytorch))
print(results_pytorch.dtype)
print(results_pytorch.shape)

quarter = results_pytorch.shape[0]/4;
print(quarter)

print(torch.split(results_pytorch, int(quarter)))


