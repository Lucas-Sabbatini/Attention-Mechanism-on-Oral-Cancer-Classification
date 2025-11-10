from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from catboost import CatBoostClassifier, Pool

class CatBoostModel:
    def __init__(self, depth=5, learning_rate=0.1, l2_leaf_reg=5, min_data_in_leaf=3,
                 iterations=500, bagging_temperature=0.5):
        self.params = {
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'min_data_in_leaf': min_data_in_leaf,
            'iterations': iterations,
            'bagging_temperature': bagging_temperature,
            'auto_class_weights': 'Balanced',
            'verbose': 0,
            'early_stopping_rounds': 50,
            'border_count': 64
        }

    def catboost_model(self, X_train_fold : np.array, X_test_fold : np.array, y_train_fold : np.array, y_test_fold : np.array):
        
        # Create validation pool for early stopping
        eval_pool = Pool(X_test_fold, y_test_fold)
        
        #Train CatBoost model
        model = CatBoostClassifier(**self.params)
        model.fit(X_train_fold, y_train_fold, eval_set=eval_pool, verbose=False)

        # Evaluate model
        y_pred = model.predict(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        esp = recall_score(y_test_fold, y_pred, pos_label=0)
        f1 = f1_score(y_test_fold, y_pred)

        return (acc, prec, rec, esp, f1)