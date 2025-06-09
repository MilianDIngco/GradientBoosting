import dectree
import csv

'''
Gradient Boosting Regresssion Algorithm
Author: Milian Ingco
Date: 6/9/2025

Implementation of gradient boosting algorithm for summer research
Takes some 1-d input and returns a 1-d prediction

'''


# Gradient Boosting

# Decision Tree algorithm

# Optimization criteria
# 1\N * sum [ y_i * ln(f_ID3(x_i) + (1 - y_i) * ln(1 - f_ID3(x_i)) ]

tree = dectree.DecisionTree()
print(tree.predict(1))

# Read in csv
filename = "project_data.csv"
whichColumn = 1

time = []
x = []

with open(filename, mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        time.append(lines[0])
        x.append(lines[whichColumn])

# Omit first entry
time.pop(0)
x.pop(0)

time = [float(val) for val in time]
x = [float(val) for val in x]
