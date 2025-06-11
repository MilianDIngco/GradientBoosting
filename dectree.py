''' 
- Decision Tree Algorithm -
Author: Milian Ingco
Date: 6/9/2025
Purpose: Train a regression decision tree on a 1-dimensional input and return an output

Future features:
 - Expand to n-dimensional input
 - Expand to n-dimensional output
 - Option for classification decision tree
'''

import sys
import math

class Split:
    def __init__(self, feature_index:int = 0, feature_value:float = 0, prediction:float = 0):
        self.index = feature_index
        self.value = feature_value
        self.prediction = prediction
        self.left = None
        self.right = None
        self.datapoints = []

    def eval(self, value:float) -> bool:
        return value > self.value

    def set_left(self, left:'Split') -> None:
        self.left = left

    def set_right(self, right:'Split') -> None:
        self.right = right

    def set_prediction(self, prediction:float) -> None:
        self.prediction = prediction

    def calc_prediction(self, is_regression:bool = True) -> None:
        if len(self.datapoints) > 0:
            if is_regression:
                for i in self.datapoints:
                    self.prediction += i
                self.prediction /= len(self.datapoints)
            else:
                totals = {}
                max_seen = 0
                for i in self.datapoints:
                    if i in totals:
                        totals[i] += 1
                    else:
                        totals[i] = 1
                    if totals[i] > max_seen:
                        max_seen = totals[i]
                        self.prediction = i

    def __str__(self) -> str:
        return f"{self.prediction}"


class DecisionTree:

    def __init__(self, x:list[float] = [0], y:list[float] = [0], epsilon:float = 0, max_depth:int = 1, n_tests:int = 1, is_regression:bool = True):
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.n_tests = n_tests
        self.root = Split()
        self.leafs:list[Split] = []

        self.is_regression = is_regression
        if self.is_regression:
            self.eval_metric = self._calc_variance
        else:
            self.eval_metric = self._calc_entropy

    def predict(self, x):
        current_split = self.root
        next = self.root

        while (next is not None):
            current_split = next

            if current_split.eval(x):
                next = current_split.right
            else:
                next = current_split.left

        return current_split

    def set_data(self, x:list[float], y:list[float]) -> None:
        self.x = x
        self.y = y

    def set_root(self, root:Split) -> None:
        self.root = root

    def start_train(self):
        self._train(self.root, 0, len(self.y), 0)
        self._assign_predictions()

    def _assign_predictions(self):
        splits = set()
        for i, value in enumerate(self.x):
            prediction = self.predict(value)
            prediction.datapoints.append(self.y[i])
            splits.add(prediction)
        for split in splits:
            split.calc_prediction(self.is_regression)
        
    def _train(self, split, min_index, max_index, depth):
        # If not enough elements to split by, stop
        if max_index - min_index < 1:
            return
        # If not reached maximum depth, stop
        if depth > self.max_depth:
            return 

        # Find best split value
        split_values = self._find_best_split(min_index, max_index)
        split.value = split_values[0]
        split_performance = split_values[1]

        # If change in variance is below epsilon, stop
        if split_performance < self.epsilon:
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
        self._train(split.left, min_index, index, depth + 1)
        self._train(split.right, index,  max_index, depth + 1)

    # Returns a tuple of the best found split value and the split performance 
    def _find_best_split(self, min_index, max_index):
        # Find range
        values = self.x[min_index:max_index]
        min_value = min(values)
        max_value = max(values)
        split_width = (max_value - min_value) / self.n_tests
        split_range = []
        current = min_value
        while current < max_value:
            split_range.append(current)
            current += split_width

        # Find best split based on variance
        max_eval_reduction = sys.float_info.min
        current_eval = self.eval_metric(min_index, max_index)
        best_split_value = 0
        for value in split_range:
            split = Split(feature_value = value)
            eval_reduction = self._eval_split(split, min_index, max_index, current_eval)
            if eval_reduction > max_eval_reduction:
                max_eval_reduction = eval_reduction
                best_split_value = split.value

        res = (best_split_value, max_eval_reduction)
        return res

    # Returns the evaluation reduction given indices and the current evaluation of the split
    def _eval_split(self, split:Split, min_index:int, max_index:int, current_eval:float) -> float:
        # Find index of split
        index = min_index
        for i in range(min_index, max_index):
            if split.eval(self.x[i]):
                index = i
                break

        # Find evaluation of left side
        left_eval = 0
        if index - min_index != 0:
            left_eval = self.eval_metric(min_index, index)
        right_eval = 0
        if max_index - index != 0:
            right_eval = self.eval_metric(index, max_index)

        # Find weighted evaluation
        n_left = (index - min_index)
        n_right = (max_index - index)
        n_total = (max_index - min_index)
        weighted_eval =  (n_left / n_total) * left_eval + (n_right / n_total) * right_eval

        return current_eval - weighted_eval

    # Calculates the variance of a range of values
    # min_index is where it starts, inclusive
    # max_index is where it ends, exclusive
    def _calc_variance(self, min_index:int, max_index:int) -> float:
        n = max_index - min_index
        mean = sum(self.y[min_index:max_index]) / n
        variance = 0
        for i in range(min_index, max_index):
            variance += (mean - self.y[i]) ** 2

        return variance / n

    # Calculates the entropy of a range of values, start is inclusive, end is exclusive
    def _calc_entropy(self, min_index:int, max_index:int) -> float:
        n = max_index - min_index
        f_ID3 = (1 / n) * sum(self.y[min_index:max_index])

        H = -f_ID3 * math.log(f_ID3) - (1 - f_ID3) * math.log(1 - f_ID3)
        return H
