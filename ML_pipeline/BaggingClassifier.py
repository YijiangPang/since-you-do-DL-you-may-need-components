from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from multiprocessing import Pool
from itertools import repeat

    
class BaggingClassifier():
    """
    voting_type = "hard"                                      #hard voting using all n_estimators
    voting_type = "hard_val_f", val_threshold = 0.6           #hard voting using n_estimators whose performance on val is > val_threshold and < 1-val_threshold.
    voting_type = "hard_prob_f", prob_threshold = 0.6         #hard voting using n_estimators whose outputed probability on training data is > prob_threshold and < 1-prob_threshold.
    voting_type = "soft"                                      #soft voting using probability outputs of all n_estimators
    voting_type = "soft_val_f", val_threshold = 0.6           #soft voting using probability outputs of n_estimators whose performance on val is > val_threshold and < 1-val_threshold.
    voting_type = "soft_prob_f", prob_threshold = 0.6         #hard voting using probability outputs of n_estimators whose outputs probability on training data is > prob_threshold and < 1-prob_threshold.
    """
    num_pool = 8
    def __init__(self, estimator, n_estimators, max_samples, voting_type, prob_threshold = None, val_threshold = None):
        super(BaggingClassifier, self).__init__()
        self.clf = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.voting_type = voting_type
        assert self.voting_type in ["hard", "hard_val_f", "hard_prob_f", "soft", "soft_val_f", "soft_prob_f"]
        if self.voting_type in ["hard_prob_f", "soft_prob_f"]: self.prob_threshold = prob_threshold
        if self.voting_type in ["hard_val_f", "soft_val_f"]: self.val_threshold = val_threshold
    def fit_predict(self, x_train, y_train, x_test):
        yhat_list = []
        with Pool(self.num_pool) as p:
            for yhat_b in p.imap_unordered(self.func_single, repeat([x_train, y_train, x_test], times = self.n_estimators)):
                if yhat_b is not None: yhat_list.append(yhat_b)
        yhat_list = np.array(yhat_list)
        if self.voting_type in ["hard", "hard_val_f"]:
            yhat = np.sum(yhat_list, axis=0)
            yhat[np.where(yhat < yhat_list.shape[0]/2)] = 0
            yhat[np.where(yhat >= yhat_list.shape[0]/2)] = 1
        elif self.voting_type in ["soft", "soft_val_f"]:
            yhat = np.mean(yhat_list, axis=0)
            yhat[np.where(yhat < 0.5)] = 0
            yhat[np.where(yhat >= 0.5)] = 1
        elif self.voting_type == "hard_prob_f":
            yhat = [1 if len(np.ravel(np.where(yhat_list[:, i] >= self.prob_threshold))) >= len(np.ravel(np.where(yhat_list[:, i] < (1 - self.prob_threshold)))) else 0 \
                        for i in range(yhat_list.shape[1])]
        elif self.voting_type == "soft_prob_f":
            yhat = [np.mean(np.concatenate([yhat_list[:, i][np.where(yhat_list[:, i] >= self.prob_threshold)], yhat_list[:, i][np.where(yhat_list[:, i] < (1 - self.prob_threshold))]]))
                    for i in range(yhat_list.shape[1])]
            yhat = [0 if i < 0.5 else 1 for i in yhat]
        else:
            raise Exception("Error! self.voting_type = %s"%(self.voting_type))
        return yhat
    
    def func_single(self, inputs):
        x_train, y_train, x_test = inputs
        x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size = 1 - self.max_samples, stratify = y_train)
        self.clf.fit(x_train_, y_train_)
        if self.voting_type in ["hard"]:
            yhat = self.clf.predict(x_test)
        elif self.voting_type in ["hard_val_f"]:
            yhat = self.clf.predict(x_val)
            acc = accuracy_score(y_val, yhat)
            yhat =  self.clf.predict(x_test) if acc > self.val_threshold else None
        elif self.voting_type in ["soft_val_f"]:
            yhat = self.clf.predict(x_val)
            acc = accuracy_score(y_val, yhat)
            if acc > self.val_threshold:
                yhat = self.clf.predict_proba(x_test)
                yhat = yhat[:, 1]
            else:
                yhat = None
        elif self.voting_type in ["soft"]:
            yhat = self.clf.predict_proba(x_test)
            yhat = yhat[:, 1]
        elif self.voting_type in ["hard_prob_f"]:
            yhat = self.clf.predict_proba(x_test)
            yhat = yhat[:, 1]
        elif self.voting_type in ["soft_prob_f"]:
            yhat = self.clf.predict_proba(x_test)
            yhat = yhat[:, 1]
        return yhat