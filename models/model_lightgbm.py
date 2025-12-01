from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class LightGBMModel:
    def __init__(self, random_state=0, n_cv=1, n_refit=0,
                 n_estimators=100, learning_rate=0.1, max_depth=-1,
                 num_leaves=31, verbosity=-1):
        self.params = {
            'random_state': random_state,
            'n_cv': n_cv,
            'n_refit': n_refit,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'verbosity': verbosity
        }        

    def lightgbm_model(self, X_train_fold : np.array, X_test_fold : np.array, y_train_fold : np.array, y_test_fold : np.array):
        # Train LightGBM model
        model = LGBMClassifier(**self.params)
        model.fit(X_train_fold, y_train_fold)

        # Evaluate model
        y_pred = model.predict(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        esp = recall_score(y_test_fold, y_pred, pos_label=0)
        mean = np.mean([rec, esp])

        return (acc, prec, rec, esp, mean)