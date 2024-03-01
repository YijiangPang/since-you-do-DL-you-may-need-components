import numpy as np


class Spectral_Signature():
    func_names = ["Spectral_sign", "Quantum_scoring"]
    def __init__(self, alpha = 4) -> None:
        self.alpha = alpha  #Quantum Entropy Scoring

    def detect(self, data_all, epsilon, flag_case = None):
        r_list = []
        for case in flag_case:
            function_detect = getattr(self, case)
            r_list.append(function_detect(data_all, epsilon) + [case])
        return r_list

    def Spectral_sign(self, data_all, epsilon):     #flag_case == 0
        data_center  = data_all - np.mean(data_all, axis=0)
        c_m = np.cov(data_center.T)    #covariance of a matrix is irrelevant with mean: np.cov(data_all.T)
        l, e, r = np.linalg.svd(c_m)
        dir_large = r[0]
        data_score = ((data_center)@dir_large[:, np.newaxis])**2
        id_sorted = [x for _, x in sorted(zip(data_score, range(data_score.shape[0])))]
        id_kept = id_sorted[:-int(len(id_sorted)*epsilon)]
        id_rm = id_sorted[-int(len(id_sorted)*epsilon):]    
        return [id_kept, [], id_rm]

    #smooth on both eigenvalues and eigenvectors
    def Quantum_scoring(self, data_all, epsilon):
        data_center  = data_all - np.mean(data_all, axis=0)
        c_m = np.cov(data_center.T)
        e_vectors_l, e_values, e_vectors_r = np.linalg.svd(c_m)
        Q = np.exp(self.alpha*c_m/e_values[0])
        U = Q/np.trace(Q)
        data_score = np.diag(data_center@U@(data_center.T))
        id_sorted = [x for _, x in sorted(zip(data_score, range(data_score.shape[0])))]
        id_kept = id_sorted[:-int(len(id_sorted)*epsilon)]
        id_rm = id_sorted[-int(len(id_sorted)*epsilon):] 
        return [id_kept, [], id_rm]


