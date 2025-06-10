import csv
import gradboost as gb
import matplotlib.pyplot as plt
import numpy as np

# Read in csv
filename = "project_data.csv"
whichColumn = 1

time = []
data = []

with open(filename, mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        time.append(lines[0])
        data.append(lines[whichColumn])

# Omit first entry
time.pop(0)
data.pop(0)

time_l = [float(val) for val in time]
data_l  = [float(val) for val in data]

time = np.array(time_l)
data = np.array(data_l)

# Create gradient booster
alpha = 1
epsilon = 0 
max_depth = 5
n_tests = 10

split_time = []
divisions = 4
for i in range(len(time) - 1):
    for n in range(1, divisions + 1):
        split_time.append(time[i] + ((time[i + 1] - time[i]) * (n / divisions)))
split_time = np.array(split_time)

test_M = [1, 5, 10]
for i, M in enumerate(test_M):
    model = gb.GradientBooster(time_l, data_l, M, alpha, epsilon, max_depth, n_tests)
    model.start_train()
    predicted = np.array([model.predict(x) for x in split_time])
    
    # Plot model against actual data
    plt.subplot(1,len(test_M),i + 1)
    plt.plot(time, data, 'o-b', markersize=5)
    plt.plot(split_time, predicted, 'o-r', markersize=1)

    plt.title(f"M = {M}")
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Cancer Cell Count")
    plt.grid(axis='y')

plt.suptitle("Growth of Untreated Cancer Cells")
plt.show()
