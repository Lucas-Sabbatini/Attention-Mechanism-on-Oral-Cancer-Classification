from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class LightGBMModel:
    def __init__(self, random_state=0,
                 boosting_type='dart',
                 max_depth=7,
                 num_leaves=71,
                 n_estimators=484, 
                 learning_rate=0.2514969057626459, 
                 min_child_samples=16,
                 min_child_weight=7.971331973511825e-05,
                 min_split_gain=0.09020796891626659,
                 reg_alpha=9.818279167092566,
                 reg_lambda=1.1548485836723954,
                 colsample_bytree=0.728829463226579,
                 is_unbalance=True,
                 subsample=0.8347716683146402,
                 subsample_freq=3,
                 drop_rate=0.32020182427300875,
                 skip_drop=0.19062794145029188,
                 verbosity=-1):
        self.params = {
            'random_state': random_state,
            'boosting_type': boosting_type,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'min_child_samples': min_child_samples,
            'min_child_weight': min_child_weight,
            'min_split_gain': min_split_gain,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'colsample_bytree': colsample_bytree,
            'is_unbalance': is_unbalance,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'drop_rate': drop_rate,
            'skip_drop': skip_drop,
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