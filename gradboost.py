'''
Gradient Boosting Regresssion Algorithm
Author: Milian Ingco
Date: 6/9/2025

Implementation of gradient boosting algorithm for summer research
Takes some 1-d input and returns a 1-d prediction

'''
import dectree as dt
import math
import sys

# Gradient Boosting
class GradientBooster:

    def __init__(self, x:list[float] = [0], y:list[float] = [0], M:int = 1, alpha:float = 1, epsilon:float = 0, max_depth:int = 1, n_split_samples:int = 1, n_rho_samples:int = 1, rho_min:int = 0, rho_max:int = 1, using_rho:bool=False, is_regression:bool = True):
        self.x = x
        self.y = y
        self.M = M
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.n_split_samples = n_split_samples
        self.n_rho_samples = n_rho_samples
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.trees:list[dt.DecisionTree] = []
        self.constant_model = 0
        self.is_regression = is_regression
        self.using_rho = using_rho
        if self.is_regression:
            self.rho = [1 for _ in range(M)]
            self.find_gradient = self._find_residuals_regression
        else:
            self.rho = []
            self.find_gradient = self._find_partial_classification

    def set_data(self, x:list[float], y:list[float]):
        self.x = x
        self.y = y

    def predict(self, x: float) -> float:
        res = self.constant_model
        for i, tree in enumerate(self.trees):
            res += self.rho[i] * self.alpha * tree.predict(x).prediction

        if self.is_regression:
            return res 
        else:
            return 1 / (1 + math.exp(-res))

    def start_train(self):
        # Find mean to set the constant model
        mean = 0
        for yi in self.y:
            mean += yi
        mean /= len(self.y)

        # Train constant model
        if self.is_regression:
            self.constant_model = mean
        else:
            self.constant_model = mean / (1 - mean)

        for _ in range(self.M):
            # Find residual from model
            gradient = self.find_gradient()

            # Train tree on residuals
            tree = self._train_tree(self.x, gradient)

            # Add tree
            if not self.is_regression and self.using_rho:
                self.rho.append(self._find_best_rho(tree))
            else:
                self.rho.append(1)
            self.trees.append(tree)

    # Finds the difference between the actual and predicted values
    def _find_residuals_regression(self) -> list[float]:
        residuals = []
        for i, x in enumerate(self.x):
            predicted = self.predict(x)
            residuals.append(self.y[i] - predicted)

        return residuals 

    # Finds the partial of the loss function with respect to the model for every y_i
    def _find_partial_classification(self) -> list[float]:
        g = []
        for yi in self.y:
            g.append(1 / (math.exp(self.predict(yi)) + 1))
        return g

    # Tests rho values from rho min to rho max sampled uniformly n times
    def _find_best_rho(self, tree:dt.DecisionTree) -> float:
        max_loss = sys.float_info.min
        test_values = [rho / self.n_rho_samples for rho in range(self.rho_min, self.rho_max)]
        best_rho = self.rho_min

        # Add tree and each value temporarily to test each rho value
        self.trees.append(tree)
        for value in test_values:
            self.rho.append(value)
            loss = self._calc_loss()
            if loss > max_loss:
                best_rho = value
                max_loss = loss
            self.rho.pop(-1)
        self.trees.pop(-1)

        # Add the best value found
        return best_rho
        
    # Find the log-likelihood of each datapoint and average them to find the average loss
    def _calc_loss(self) -> float:
        loss = 0
        for i, yi in enumerate(self.y):
            yhat = self.predict(self.x[i])
            loss += -( yi * math.log(yhat) + (1 - yi) * math.log(1 - yhat) )
        loss /= len(self.y)
        return loss

    # Train a tree on a given dataset
    def _train_tree(self, x:list[float], y:list[float]) -> dt.DecisionTree:
        tree = dt.DecisionTree(x, y, self.epsilon, self.max_depth, self.n_split_samples)
        tree.start_train()

        return tree


