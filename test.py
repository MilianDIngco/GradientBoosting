import dectree as dt
import gradboost as gb
import random
import sys

def list_to_data(arr):
    pass


def dt_prediction():
    print("--------------------Decision Tree Prediction-----------------")
    tree = dt.DecisionTree()
    root = dt.Split(0, 10, 10)
    root.left = dt.Split(0, 5, 5)
    root.right = dt.Split(0, 15, 15)
    root.left.left = dt.Split(0, 2.5, 2.5)
    root.left.right = dt.Split(0, 7.5, 7.5)
    root.right.left = dt.Split(0, 12.5, 12.5)
    root.right.right = dt.Split(0, 17.5, 17.5)

    tree.set_root(root)

    for i in range(-10, 30):
        predicted = tree.predict(i)
        print(f"In: {i} Out: {predicted}")

    '''
             10
        5          15
    2.5   7.5  12.5   17.5
        '''
def dt_calc_variance():
    print("----------------------Calculate variance-------------------------")

    # Sets with high variance
    high_variance_sets = []
    difference = 200
    for i in range(20):
        high_set = []
        current = random.random() * 100
        for _ in range(20):
            high_set.append(current)
            current += difference + difference * random.random()
        high_variance_sets.append(high_set)

    # Sets with low variance
    low_variance_sets = []
    difference = 5
    for i in range(20):
        low_set = []
        current = random.random() * 100
        for _ in range(20):
            low_set.append(current)
            current *= random.random() + 0.25
        low_variance_sets.append(low_set)

    # Random sets
    random_variance_sets = []
    for i in range(20):
        random_set = []
        current = random.random() * 100 - 50
        for _ in range(20):
            random_set.append(current)
            current = random.random() * 1000 - 50
        random_variance_sets.append(random_set)

    x = [float(x) for x in range(0, 20)]

    tree = dt.DecisionTree()

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

def dt_split_evaluation():
    print("------------------Split evaluation---------------------")

    tree = dt.DecisionTree()
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
        split = dt.Split(feature_value=x[i])
        variance_reduction = tree._eval_split(split, 0, len(x), good_initial_variance)
        print(f"Good set evaluated at {i}\n Variance reduction of {variance_reduction}")

    print("Uniform split set")
    tree.set_data(x, uniform_set)
    uniform_initial_variance = tree._calc_variance(0, len(x))
    print(f"Uniform initial variance: {uniform_initial_variance}")
    for i in range(20):
        split = dt.Split(feature_value=i)
        variance_reduction = tree._eval_split(split, 0, len(x), uniform_initial_variance)
        print(f"Uniform set evaluated at {i}\n Variance reduction of {variance_reduction}")

def dt_best_split():
    print("-------------------Find best split--------------------------")

    # Set with a very good split
    good_set = []
    for _ in range(20):
        good_set.append(5)
    for _ in range(10):
        good_set.append(100)

    # Set with completely uniform data
    uniform_set = [5.0 for _ in range(30)]

    x = [float(x) for x in range(30)]

    tree = dt.DecisionTree()
    tree.n_split_samples = 30

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

def dt_training():
    print("------------------finally... start training--------------------")
    print("Simple set")
    N = 30
    epsilon = 0
    max_depth = 20
    n_split_samples = 20
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

    finalboss = dt.DecisionTree(x, y, epsilon, max_depth, n_split_samples)
    finalboss.start_train()
    for leaf in finalboss.leafs:
        print(leaf.datapoints)
    for i in range(N):
        prediction = finalboss.predict(x[i])
        print(f"{i}: Actual={y[i]} Predicted={prediction}")

    print("Quadratic")
    N = 30
    epsilon = 0
    max_depth = 100
    n_split_samples = 20
    x = [float(i) for i in range(N)]
    y = [val ** 2 for val in x]
    finalboss = dt.DecisionTree(x, y, epsilon, max_depth, n_split_samples)
    finalboss.start_train()
    for i in range(N):
        prediction = finalboss.predict(x[i])
        print(f"{i}: Actual={y[i]} Predicted={prediction}")

    print("Random inputs (should be the input squared)")
    for i in range(N):
        input = random.random() * N
        prediction = finalboss.predict(input)
        print(f"{input}: Actual={input ** 2} Predicted={prediction}")

def dt_constant_tree():
    # Testing making a constant model with decision tree
    print("--------------------Constant tree--------------------")
    val = 9.0
    constree = dt.DecisionTree([0.0], [val], 1, 1, 1)
    constree.start_train()
    
    for i in range(5):
        print(f"{i - 2}: predicted {constree.predict(i - 2)}")

def gb_residuals():
    # Test finding residuals
    print("---------------------Finding residuals-------------------")
    N = 10
    x = [float(x) for x in range(N)]
    y = [float(x) for x in range(N)]
    boost = gb.GradientBooster(x, y, M=0, alpha=1, epsilon=0, max_depth=1, n_split_samples=1)
    boost.constant_model = 4.5
    boost.start_train()
    print(boost.find_gradient())

def gb_prediction():
    print("--------------------Test prediction-------------------")
    val1 = 1
    val2 = 2
    tree_1 = dt.DecisionTree([0.0], [val1], 1, 1, 1)
    tree_2 = dt.DecisionTree([0.0], [val2], 1, 1, 1)
    tree_1.start_train()
    tree_2.start_train()
    pred_gb = gb.GradientBooster(M=3)
    pred_gb.trees.append(tree_1)
    pred_gb.trees.append(tree_2)

    print(f"Tree 1 predicts: {tree_1.predict(10)}")
    print(f"Tree 2 predicts: {tree_2.predict(10)}")

    print(f"Actual: {val1 + val2} Predicted: {pred_gb.predict(10)}")

def gb_training():
    print("-------------------Test training gradient boost------------------")

    print("==============Testing y = x (1 = 1 etc)=================")
    N = 30
    x = [float(x) for x in range(N)]
    y = [float(x) for x in range(N)]

    #       M, alpha, epsilon, max_depth, n_split_samples
    args = [50, 1, 0, 20, 20]
    ffb = gb.GradientBooster(x, y, args[0], args[1], args[2], args[3], args[4])
    ffb.start_train()
    for i, xi in enumerate(x):
        actual = y[i]
        predicted = ffb.predict(xi)
        print(f"Percent error: {(((actual + 1) - (predicted + 1)) / (actual + 1)):.3f}% Actual: {actual} Predicted: {predicted}")

    print("===============Testing y = x^2===================")
    N = 30
    x = [float(x) for x in range(N)]
    y = [float(xi ** 2) for xi in x]

    #       M, alpha, epsilon, max_depth, n_split_samples
    args = [20, 1, 0, 20, 20]
    ffb = gb.GradientBooster(x, y, args[0], args[1], args[2], args[3], args[4])
    ffb.start_train()
    for i, xi in enumerate(x):
        actual = y[i]
        predicted = ffb.predict(xi)
        print(f"Percent error: {(((actual + 1) - (predicted + 1)) / (actual + 1)):.3f}% Actual: {actual} Predicted: {predicted}")

def dt_calc_entropy():
    N = 20
    x = [float(i) for i in range(N)]

    # Testing high entropy sets
    y = [float(i % 2) for i in range(N)]
    tree = dt.DecisionTree(x, y, is_regression=False)
    print(y)
    print(f"High entropy: {tree._calc_entropy(0, N)}")

    # Testing low entropy sets
    y = [1 for _ in range(N)]
    tree.set_data(x, y)
    print(y)
    print(f"Low entropy: {tree._calc_entropy(0, N)}")

    # Testing random sets (should be pretty high entropy)
    y = [float(random.randint(0, 1)) for _ in range(N)]
    tree.set_data(x, y)
    print(y)
    print(f"Random set entropy: {tree._calc_entropy(0, N)}")

def dt_eval_entropy_split():
    N = 20
    x = [float(i) for i in range(N)]
    tree = dt.DecisionTree(is_regression=False)

    # Testing set with entropy thats easy to split
    y = []
    for _ in range(N // 2):
        y.append(0.0)
    for _ in range(N // 2, N):
        y.append(1.0)

    tree.set_data(x, y)
    original_entropy = tree._calc_entropy(0, N)
    print("Easy to split set")
    print(y)
    print(f"Ideal split is around {N // 2}? might be -1")
    for i in range(N):
        split = dt.Split(feature_value=i)
        print(f"Split evaluation at index {i} is {tree._eval_split(split, 0, N, original_entropy)}")

    # Testing uniform set
    y = [1.0 for _ in range(N)]
    
    tree.set_data(x, y)
    original_entropy = tree._calc_entropy(0, N)
    print("Uniform set")
    print(y)
    print(f"Ideal split is any of them? perhaps? or the first element")
    for i in range(N):
        split = dt.Split(feature_value=i)
        print(f"Split evaluation at index {i} is {tree._eval_split(split, 0, N, original_entropy)}")

def dt_find_best_entropy_split():
    N = 20
    x = [float(i) for i in range(N)]
    tree = dt.DecisionTree(n_split_samples=20, is_regression=False)

    # Testing set with entropy thats easy to split 
    y = []
    for _ in range(N // 2):
        y.append(0.0)
    for _ in range(N // 2, N):
        y.append(1.0)

    tree.set_data(x, y)
    print("Finding best split for easy to split data")
    print(y)
    results = tree._find_best_split(0, N)
    print(f"Best split: {results[0]} Reduced entropy: {results[1]} Original entropy: {tree._calc_entropy(0, N)}")
    
    # Testing uniform set
    y = [1.0 for _ in range(N)]
    
    tree.set_data(x, y)
    print("Finding best split for uniform data")
    print(y)
    results = tree._find_best_split(0, N)
    print(f"Best split: {results[0]} Reduced entropy: {results[1]} Original entropy: {tree._calc_entropy(0, N)}")

def dt_classification_training():
    print("Simple set")
    N = 30
    epsilon = 0
    max_depth = 20
    n_split_samples = 20
    x = [float(x) for x in range(N)]
    y = []
    for _ in range(5):
        y.append(0.0)
    for _ in range(10):
        y.append(1.0)
    for _ in range(5):
        y.append(0.0)
    for _ in range(3):
        y.append(1.0)
    for _ in range(2):
        y.append(0.0)
    for _ in range(5):
        y.append(1.0)

    finalboss = dt.DecisionTree(x, y, epsilon, max_depth, n_split_samples, is_regression=False)
    finalboss.start_train()
    for i in range(N):
        prediction = finalboss.predict(x[i])
        print(f"{i}: Actual={y[i]} Predicted={prediction}")

    print("Random inputs")
    for _ in range(N):
        input = random.random() * N
        prediction = finalboss.predict(input)
        print(f"{input}: Actual={y[round(input)]} Predicted={prediction}")

def gb_calc_loss():
    pass

def gb_find_best_rho():
    pass

def gb_find_partials():
    pass

def gb_training_classification():
    pass

def clear_terminal():
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()
# dt_prediction(), dt_calc_variance(), dt_split_evaluation(), dt_best_split(), dt_training(), dt_constant_tree()

# gb_residuals(), gb_prediction(), gb_training()

dt_tests = [dt_prediction, dt_calc_variance, dt_split_evaluation, dt_best_split, dt_training, dt_constant_tree, 
            dt_calc_entropy, dt_eval_entropy_split, dt_find_best_entropy_split, 
            dt_classification_training]

gb_tests = [gb_residuals, gb_prediction, gb_training]

user_input = ''
while True:
    user_input = input("Enter option: \n1: decision tree\n2: gradient boosting\nq: quit\n")
    clear_terminal()
    if user_input == 'q':
        clear_terminal()
        break

    if user_input == '1':
        user_input = input("Enter test: 1-6\n1: dt_prediction\n2: dt_calc_variance\n3: dt_split_evaluation\n4: dt_best_split\n5: dt_training\n6: dt_constant_tree\n7: dt_calc_entropy\n8: dt_eval_entropy_split\n9: dt_find_best_entropy_split\n10: dt_classification_training\nq: quit\n")
        if user_input == 'q':
            clear_terminal()
            break
        else:
            clear_terminal()
            user_input = int(user_input) - 1
            dt_tests[user_input]()
    elif user_input == '2':
        user_input = input("Enter test: 1-3\n1: gb_residuals\n2: gb_prediction\n3: gb_training\nq: quit\n")
        if user_input == 'q':
            clear_terminal()
            break
        else:
            clear_terminal()
            user_input = int(user_input) - 1
            gb_tests[user_input]()

