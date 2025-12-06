from abc import ABC, abstractmethod
import numpy as np

class BaseClassifierModel(ABC):

    @abstractmethod
    def evaluate(self, X_train : np.array, X_test : np.array, y_train : np.array, y_test : np.array):
        """Evaluate the model's performance on the provided data and labels."""
        pass