import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

from models.model import BaseClassifierModel

class XGBModel(BaseClassifierModel):
    def __init__(self, max_depth=4, eta=1, objective='binary:logistic', nthread=4, eval_metric=['auc'], num_round=10):
        self.params = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': objective,
            'nthread': nthread,
            'eval_metric': eval_metric
        }
        self.num_round = num_round

    def evaluate(self, X_train : np.array, X_test : np.array, y_train : np.array, y_test : np.array):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        # Instantiate the model
        model = xgb.train(self.params, dtrain, self.num_round, evallist)

        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)

        #Eval metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        esp = recall_score(y_test, y_pred, pos_label=0)
        mean = np.mean([rec, esp])

        return (acc, prec, rec, esp, mean)