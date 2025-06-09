''' 
- Decision Tree Algorithm -
Author: Milian Ingco
Date: 6/9/2025
Purpose: Train a regression decision tree on a 1-dimensional input and return an output
'''

import sys

class Split:
    def __init__(self, min_range:float = 0, max_range:float = 0, feature_index:int = 0, feature_value:float = 0, prediction:float = 0):
        self.index = feature_index
        self.value = feature_value
        self.min_range = min_range
        self.max_range = max_range
        self.prediction = prediction
        self.left = None
        self.right = None

    def eval(self, value:float) -> bool:
        return value > self.value

    def set_left(self, left:'Split') -> None:
        self.left = left

    def set_right(self, right:'Split') -> None:
        self.right = right

    def set_prediction(self, prediction:float) -> None:
        self.prediction = prediction


class DecisionTree:

    def __init__(self, x:list[float] = [0], y:list[float] = [0], epsilon:float = 0, max_depth:int = 0, n_tests:int = 2):
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.n_tests = n_tests
        self.root = Split()

    def predict(self, x):

        current_split = self.root
        next = self.root

        while (next is not None):
            current_split = next

            if current_split.eval(x):
                next = current_split.right
            else:
                next = current_split.left

        return current_split.prediction

    def set_data(self, x:list[float], y:list[float]) -> None:
        self.x = x
        self.y = y

    def set_root(self, root:Split) -> None:
        self.root = root

    def start_train(self):
        self._train(self.root, 0, len(self.y) - 1, 0)
        
    def _train(self, split, min_index, max_index, depth):
        # If not enough elements to split by, stop
        if max_index - min_index < 3:
            return
        # If not reached maximum depth, stop
        if depth > self.max_depth:
            return 

        # Find best split value
        split_values = self._find_best_split(min_index, max_index)
        split.value = split_values[0]
        variance_reduction = split_values[1]

        # If change in variance is below epsilon, stop
        if variance_reduction < self.epsilon:
            return

        # Find where splits at
        index = min_index 
        for i in range(min_index, max_index):
            if split.eval(self.x[i]):
                index = i
                break
        
        # Find children
        split.left = Split()
        split.right = Split()
        self._train(split.left, min_index, index - 1, depth + 1)
        self._train(split.right, index,  max_index, depth + 1)

    # Returns a tuple of the best found split value and the variance reduction
    def _find_best_split(self, min_index, max_index):
        # Find range
        labels = self.y[min_index:max_index]
        min_value = min(labels)
        max_value = max(labels)
        split_width = (max_value - min_value) / self.n_tests
        split_range = []
        current = min_value
        while current < max_value:
            split_range.append(current)
            current += split_width

        # Find best split based on variance
        max_variance_reduction = sys.float_info.min
        current_variance = self._calc_variance(min_index, max_index)
        best_split_value = 0
        for value in split_range:
            split = Split(feature_value = value)
            variance_reduction = self._eval_split(split, min_index, max_index, current_variance)
            if variance_reduction > max_variance_reduction:
                max_variance_reduction = variance_reduction
                best_split_value = split.value

        res = (best_split_value, max_variance_reduction)
        return res

    # Returns the variance reduction given indices and the current variance of the split
    def _eval_split(self, split:Split, min_index:int, max_index:int, current_variance:float) -> float:
        # Find index of split
        index = min_index
        for i in range(min_index, max_index):
            if split.eval(self.x[i]):
                index = i
                break

        # Find variance of left side
        left_variance = self._calc_variance(min_index, index - 1)
        right_variance = self._calc_variance(index, max_index)

        # Find weighted variance
        n_left = (index - min_index)
        n_right = (max_index - index)
        n_total = (max_index - min_index)
        weighted_variance =  (n_left / n_total) * left_variance + (n_right / n_total) * right_variance

        return current_variance - weighted_variance

    # Calculates the variance of a range of values
    def _calc_variance(self, min_index:int, max_index:int) -> float:
        n = max_index - min_index
        mean = sum(self.y[min_index:max_index]) / n
        variance = 0
        for i in range(min_index, max_index):
            variance += (mean - self.y[i]) ** 2

        return variance / n

