import os

for line in open("list_eval_partition.txt", 'r'):
    train, val, test = list(),list(),list()
    sample = line.split()
    if sample[1] == 0:
        train.append()