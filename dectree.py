''' 
- Decision Tree Algorithm -
Author: Milian Ingco
Date: 6/9/2025
Purpose: Train a regression decision tree on a 1-dimensional input and return an output
'''

class Split:
    def __init__(self, feature_index:int = 0, feature_value:float = 0, prediction:float = 0):
        self.index = feature_index
        self.value = feature_value
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

    def __init__(self, x:list[float] = [0], y:list[float] = [0]):
        self.x = x
        self.y = y
        self.root = Split()

    def predict(self, x):

        current_split = self.root
        next = current_split.left

        while (next is not None):
            if current_split.eval(x):
                next = current_split.right
            else:
                next = current_split.left

        return current_split.prediction

    def set_data(self, x:list[float], y:list[float]) -> None:
        self.x = x
        self.y = y

    def _eval_split(self):
        pass

    def _objective_function(self) -> float:
        # Optimization criteria
        # 1/N * sum [ y_i * ln(f_ID3(x_i) + (1 - y_i) * ln(1 - f_ID3(x_i)) ]
        return 0
