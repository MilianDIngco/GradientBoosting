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
        self.constant_model = mean

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


