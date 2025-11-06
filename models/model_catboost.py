from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from catboost import CatBoostClassifier

class CatBoostModel:
    def catboost_model(self, X_train_fold : np.array, X_test_fold : np.array, y_train_fold : np.array, y_test_fold : np.array):
        
        #Train CatBoost model
        model = CatBoostClassifier(verbose=0)
        model.fit(X_train_fold, y_train_fold)

        #Evaluate model
        y_pred = model.predict(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        esp = recall_score(y_test_fold, y_pred, pos_label=0)
        f1 = f1_score(y_test_fold, y_pred)

        return (acc, prec, rec, esp, f1)