from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class RealMLPModel:
    def __init__(self,device='cpu', random_state=0, n_cv=1, n_refit=0,
                              n_epochs=128, batch_size=128, hidden_sizes=[128] * 3,
                              val_metric_name='cross_entropy',
                              use_ls=False,
                              lr=0.08, verbosity=0):
        self.params = {
            'device': device,
            'random_state': random_state,
            'n_cv': n_cv,
            'n_refit': n_refit,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'hidden_sizes': hidden_sizes,
            'val_metric_name': val_metric_name,
            'use_ls': use_ls,
            'lr': lr,
            'verbosity': verbosity
        }

    def realmlp_model(self, X_train_fold : np.array, X_test_fold : np.array, y_train_fold : np.array, y_test_fold : np.array):
        # Train RealMLP model
        model = RealMLP_TD_Classifier(**self.params)
        model.fit(X_train_fold, y_train_fold)

        # Evaluate model
        y_pred = model.predict(X_test_fold)

        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        esp = recall_score(y_test_fold, y_pred, pos_label=0)
        mean = np.mean([rec, esp])

        return (acc, prec, rec, esp, mean)