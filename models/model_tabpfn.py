from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from tabpfn import TabPFNClassifier

class TabPFNModel:    
    def tabpfn_model(self, X_train_fold : np.array, X_test_fold : np.array, y_train_fold : np.array, y_test_fold : np.array):
        
        #Train SVM model
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X_train_fold, y_train_fold)

        #Evaluate model
        y_pred = clf.predict(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        esp = recall_score(y_test_fold, y_pred, pos_label=0)
        mean = np.mean([rec, esp])

        return (acc, prec, rec, esp, mean)