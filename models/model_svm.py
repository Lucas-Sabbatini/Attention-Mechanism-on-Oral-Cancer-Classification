from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
import numpy as np

class SVMRBFModel:
    def __init__(self, C=1, gamma='scale'):
        self.kernel = 'rbf'
        self.C = C
        self.gamma = gamma
    
    def svm_rbf_model(self, X_train_fold : np.array, X_test_fold : np.array, y_train_fold : np.array, y_test_fold : np.array):
        
        #Train SVM model
        clf = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        clf.fit(X_train_fold, y_train_fold)

        #Evaluate model
        y_pred = clf.predict(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        esp = recall_score(y_test_fold, y_pred, pos_label=0)
        f1 = f1_score(y_test_fold, y_pred)

        return (acc, prec, rec, esp, f1)