import dectree
import random

print("--------------------Decision Tree Prediction-----------------")
tree = dectree.DecisionTree()
root = dectree.Split(0, 10, 10)
root.left = dectree.Split(0, 5, 5)
root.right = dectree.Split(0, 15, 15)
root.left.left = dectree.Split(0, 2.5, 2.5)
root.left.right = dectree.Split(0, 7.5, 7.5)
root.right.left = dectree.Split(0, 12.5, 12.5)
root.right.right = dectree.Split(0, 17.5, 17.5)

tree.set_root(root)

for i in range(-10, 30):
    predicted = tree.predict(i)
    print(f"In: {i} Out: {predicted}")

'''
         10
    5          15
2.5   7.5  12.5   17.5
    '''

print("----------------------Calculate variance-------------------------")

# Sets with high variance
high_variance_sets = []
difference = 200
for i in range(20):
    high_set = []
    current = random.random() * 100
    for n in range(20):
        high_set.append(current)
        current += difference + difference * random.random()
    high_variance_sets.append(high_set)

# Sets with low variance
low_variance_sets = []
difference = 5
for i in range(20):
    low_set = []
    current = random.random() * 100
    for n in range(20):
        low_set.append(current)
        current *= random.random() + 0.25
    low_variance_sets.append(low_set)

# Random sets
random_variance_sets = []
for i in range(20):
    random_set = []
    current = random.random() * 100 - 50
    for n in range(20):
        random_set.append(current)
        current = random.random() * 1000 - 50
    random_variance_sets.append(random_set)

x = [float(x) for x in range(0, 20)]

average_variance = 0
for i in range(len(high_variance_sets)):
    tree.set_data(x, high_variance_sets[i])
    average_variance += tree._calc_variance(0, len(high_variance_sets[i]))
average_variance /= len(high_variance_sets)
print(f"Average variance for high sets was {average_variance}")

average_variance = 0
for i in range(len(low_variance_sets)):
    tree.set_data(x, low_variance_sets[i])
    average_variance += tree._calc_variance(0, len(low_variance_sets[i]))
average_variance /= len(low_variance_sets)
print(f"Average variance for low sets was {average_variance}")

average_variance = 0
for i in range(len(random_variance_sets)):
    tree.set_data(x, random_variance_sets[i])
    average_variance += tree._calc_variance(0, len(random_variance_sets[i]))
average_variance /= len(random_variance_sets)
print(f"Average variance for random sets was {average_variance}")

print("------------------Split evaluation---------------------")

# Sets with very good splits
good_set = []
for n in range(10):
    good_set.append(5)
for n in range(10):
    good_set.append(100)

# Sets that are uniformly distributed
uniform_set = []
for n in range(20):
    uniform_set.append(n * 5)

x = [float(x) for x in range(20)]

print("Good split set")
tree.set_data(x, good_set)
good_initial_variance = tree._calc_variance(0, len(x))
print(f"Good initial variance: {good_initial_variance}")
for i in range(20):
    split = dectree.Split(feature_value=x[i])
    variance_reduction = tree._eval_split(split, 0, len(x), good_initial_variance)
    print(f"Good set evaluated at {i}\n Variance reduction of {variance_reduction}")

print("Uniform split set")
tree.set_data(x, uniform_set)
uniform_initial_variance = tree._calc_variance(0, len(x))
print(f"Uniform initial variance: {uniform_initial_variance}")
for i in range(20):
    split = dectree.Split(feature_value=i)
    variance_reduction = tree._eval_split(split, 0, len(x), uniform_initial_variance)
    print(f"Uniform set evaluated at {i}\n Variance reduction of {variance_reduction}")

print("-------------------Find best split--------------------------")

# Set with a very good split
good_set = []
for n in range(20):
    good_set.append(5)
for n in range(10):
    good_set.append(100)

# Set with completely uniform data
uniform_set = [5.0 for _ in range(30)]

x = [float(x) for x in range(30)]

tree.n_tests = 10

print("Best split for good set")
tree.set_data(x, good_set)
good_initial_variance = tree._calc_variance(0, len(x))
results = tree._find_best_split(0, len(x))
print(f"Good sets best value: {results[0]}, variance reduction: {results[1]}, current variance: {good_initial_variance}")

print("Best split for completely uniform set")
tree.set_data(x, uniform_set)
uniform_initial_variance = tree._calc_variance(0, len(x))
results = tree._find_best_split(0, len(x) - 1)
print(f"Uniform sets best value: {results[0]}, varianced reduction: {results[1]}, current variance: {uniform_initial_variance}")

print("------------------finally... start training--------------------")
print("Simple set")
N = 30
epsilon = 0
max_depth = 100
n_tests = 200
x = [float(x) for x in range(N)]
y = []
for i in range(5):
    y.append(1)
for i in range(5):
    y.append(2)
for i in range(5):
    y.append(10)
for i in range(5):
    y.append(1)
for i in range(5):
    y.append(100)
for i in range(5):
    y.append(200)

finalboss = dectree.DecisionTree(x, y, epsilon, max_depth, n_tests)
finalboss.start_train()
for i in range(N):
    prediction = finalboss.predict(x[i])
    print(f"{i}: Actual={y[i]} Predicted={prediction}")

print("Quadratic")
N = 30
epsilon = 0
max_depth = 100
n_tests = 20
x = [float(i) for i in range(N)]
y = [val ** 2 for val in x]
finalboss = dectree.DecisionTree(x, y, epsilon, max_depth, n_tests)
finalboss.start_train()
for i in range(N):
    prediction = finalboss.predict(x[i])
    print(f"{i}: Actual={y[i]} Predicted={prediction}")

print("Random inputs (should be the input squared)")
for i in range(N):
    input = random.random() * N
    prediction = finalboss.predict(input)
    print(f"{input}: Actual={input ** 2} Predicted={prediction}")
