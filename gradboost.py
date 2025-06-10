'''
Gradient Boosting Regresssion Algorithm
Author: Milian Ingco
Date: 6/9/2025

Implementation of gradient boosting algorithm for summer research
Takes some 1-d input and returns a 1-d prediction

'''
import dectree as dt

# Gradient Boosting
class GradientBooster:

    def __init__(self, x:list[float] = [0], y:list[float] = [0], M:int = 1, alpha:float = 1, epsilon:float = 0, max_depth:int = 1, n_tests:int = 1):
        self.x = x
        self.y = y
        self.M = M
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.n_tests = n_tests
        self.trees:list[dt.DecisionTree] = []
        self.constant_model = 0

    def set_data(self, x:list[float], y:list[float]):
        self.x = x
        self.y = y

    def predict(self, x: float) -> float:
        res = self.constant_model
        for tree in self.trees:
            res += self.alpha * tree.predict(x).prediction
        return res 

    def start_train(self):
        # Find mean to set the constant model
        mean = 0
        for yi in self.y:
            mean += yi
        mean /= len(self.y)

        # Train constant model
        root = dt.DecisionTree([0.0], [mean], 0, 1, 1)
        root.start_train()
        self.constant_model = root.predict(0).prediction # Will always predict same thing

        for _ in range(self.M):
            # Find residual from constant model
            residuals = self._find_residuals()

            # Train tree on residuals
            tree = self._train_tree(self.x, residuals)

            # Add tree
            self.trees.append(tree)

    def _find_residuals(self) -> list[float]:
        residuals = []
        for i, x in enumerate(self.x):
            predicted = self.predict(x)
            residuals.append(self.y[i] - predicted)

        return residuals 

    def _train_tree(self, x:list[float], y:list[float]) -> dt.DecisionTree:
        tree = dt.DecisionTree(x, y, self.epsilon, self.max_depth, self.n_tests)
        tree.start_train()

        return tree


'''
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
'''
