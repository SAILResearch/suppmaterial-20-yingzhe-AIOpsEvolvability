from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import mean_squared_error
import numpy as np

# MSE_R for true : false = 1:10
P_TRUE = 1 / 11
P_FALSE = 1 - P_TRUE
MSE_R = P_TRUE*(1-P_TRUE)*(1-P_TRUE) + P_FALSE*(1-P_FALSE)*(1-P_FALSE)
EPS = 1e-5

class OnlineModel(object):
    def __init__(self, ensemble_size):
        self.name = "Base online model"
        self.model = None
        self.scaler = None
        self.ensemble_size = ensemble_size

    def fit(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return np.clip(self.model.predict_proba(X), 0.0, 1.0)
        
    def get_name(self):
        return self.name


class ARFModel(OnlineModel):
    def __init__(self, ensemble_size):
        super().__init__(ensemble_size)
        self.name = "ARF"
        self.model = AdaptiveRandomForestClassifier(n_estimators=ensemble_size, performance_metric='kappa')


class HTModel(OnlineModel):
    def __init__(self, ensemble_size):
        super().__init__(ensemble_size)
        self.name = "HT"
        self.model = HoeffdingTreeClassifier()


class AUEModel(OnlineModel):
    def __init__(self, ensemble_size):
        super().__init__(ensemble_size)
        self.name = 'AUE'
        self.models = []
        self.weights = []
        
    def fit(self, X, y):
        model = HoeffdingTreeClassifier()
        weight = 1 / (MSE_R + EPS)
        model.fit(X, y)  # fit or partial fit?
        
        for idx in range(len(self.models)):
            probas = self.models[idx].predict_proba(X)[:, 1]
            mse = mean_squared_error(y, probas)
            self.weights[idx] = 1 / (MSE_R + mse + EPS)

        if len(self.models) < self.ensemble_size:
            self.models.append(model)
            self.weights.append(weight)
            self.add = True
        else:
            if min(self.weights) < weight:
                idx = self.weights.index(min(self.weights))
                self.models[idx] = model
                self.weights[idx] = weight
                self.add = True
            else:
                self.add = False

        for idx in range(len(self.models)):
            if self.models[idx] is not model:
                self.models[idx].partial_fit(X, y)
                
    def predict_proba(self, X):
        probas = np.array([self.models[i].predict_proba(X) for i in range(len(self.models))])
        return np.clip(np.average(probas, axis=0, weights=self.weights), 0.0, 1.0)
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5

    def is_added(self):
        return self.add
