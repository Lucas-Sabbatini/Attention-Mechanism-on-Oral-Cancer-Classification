from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

class LightGBMModel:
    def __init__(self, random_state=0,
                 boosting_type='dart',
                 max_depth=3,
                 num_leaves=6,
                 n_estimators=225, 
                 learning_rate=0.16043735816565025, 
                 min_child_samples=9,
                 min_child_weight=0.00013750492437919427,
                 min_split_gain=0.969636350863479,
                 reg_alpha=1.0853779984465421,
                 reg_lambda=0.6477536506338741,
                 colsample_bytree=0.8963125360673033,
                 is_unbalance=True,
                 subsample=0.6369808335483202,
                 subsample_freq=0,
                 drop_rate=0.24756745181851222,
                 skip_drop=0.34157410975192776,
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