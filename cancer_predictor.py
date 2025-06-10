import csv
import gradboost as gb
import matplotlib.pyplot as plt
import numpy as np

whichColumn = 1

# Default Gradient Booster Params
M = 5
alpha = 1
epsilon = 0 
max_depth = 5
n_tests = 10

param_names = ["M", "Alpha", "Epsilon", "Max Depth", "# of Samples"]
params = [M, alpha, epsilon, max_depth, n_tests]

# Gradient Booster Param values to test
parameter = 1
test = [0.1, 0.5, 1]

# Read in csv
filename = "project_data.csv"

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
split_time = []
divisions = 10
for i in range(len(time) - 1):
    for n in range(1, divisions + 1):
        split_time.append(time[i] + ((time[i + 1] - time[i]) * (n / divisions)))
split_time = np.array(split_time)

for i, p in enumerate(test):
    params[parameter] = p
    model = gb.GradientBooster(time_l, data_l, params[0], params[1], params[2], params[3], params[4])
    model.start_train()
    predicted = np.array([model.predict(x) for x in split_time])
    
    # Plot model against actual data
    plt.subplot(1,len(test),i + 1)
    plt.plot(time, data, 'o-b', markersize=5)
    plt.plot(split_time, predicted, 'o-r', markersize=1)

    plt.title(f"{param_names[parameter]} = {p}")
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Cancer Cell Count")
    plt.grid(axis='y')

plt.suptitle("Growth of Untreated Cancer Cells")
plt.show()
