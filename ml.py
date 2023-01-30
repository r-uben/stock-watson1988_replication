import numpy as np
import pandas as pd

from kf import KF
from __helpers.aux import Aux
from scipy.optimize import minimize 
from numpy.linalg import det, inv, LinAlgError

class ML(object):

    def __init__(self,  Y, kf_params, name) -> None:

        self._beta  = kf_params["beta"]
        self._H     = kf_params["H"]
        self._phi   = kf_params["phi"]
        self._gamma = kf_params["gamma"]
        self._D1    = kf_params["D"][1]
        self._D2   = kf_params["D"][2]
        self._Sigma = kf_params["Sigma"]
        self._Y     = Y
        self._params = [self.beta] + [self.H] + self.phi + self.gamma + self.D1 + self.D2 + self.Sigma
        self._params = np.array(self._params)
        self._num_it = 0
        self.aux = Aux()
        self._name = name
        if "new" in self._name: self._columns = ["USIPMANG", "USGPYD", "USPPM", "USWRKW"]
        else: self._columns = ["IP", "GMYX", "LPMH", "LPMHU"]

    def optimisation(self) -> None:
        bounds = self.bounds
        mle = minimize(self.L, self.params, method='BFGS', hess=True)
        se = list(np.sqrt(np.diag(mle.hess_inv)))
        se = self.dict_se(se)
        self.aux.save_se(se, self._name)

    def kalman_filter(self, params) -> tuple:
        # params
        beta, H, phi1, phi2, gamma1, gamma2, gamma3, gamma4, d11, d12, d13, d14, d21, d22, d23, d24, sigma1, sigma2, sigma3, sigma4 = params     
        phi     = np.array([phi1, phi2])
        D1      = np.diag([d11,d12,d13,d14])
        D2      = np.diag([d21,d22,d23,d24])
        D       = {1: D1, 2: D2}
        Sigma   = np.diag([1, sigma1**2, sigma2**2, sigma3**2, sigma4**2])

        gamma   = np.array([gamma1,gamma2,gamma3,gamma4])
        # Initial values:
        C_vals = []
        P_C_vals = []

        # Forecast errors:
        nu = []
        F  = []
        for t in range(len(self._Y.index)):
            y = self._Y.iloc[t].to_numpy().reshape(len(self._Y.columns),1) # I need this because neither .T nor .transpose() is working
            if t==0:
                kf = KF(Y=y,phi=phi, D=D, Sigma=Sigma, beta=beta, gamma=gamma, H=0)
            else:
                kf = KF(Y=y,phi=phi, D=D, Sigma=Sigma, beta=beta, gamma=gamma, H=0,alpha=alpha, P=P)
            kf.predict()
            kf.update()
            alpha = kf.alpha
            P     = kf.P
            # Save the index, Ct, nu and F:
            C_vals.append(kf.C)
            P_C_vals.append(kf.P_C)
            nu.append(kf.nu)
            F.append(kf.F)
        return C_vals, P_C_vals, nu, F

    def L(self, params):
        self._params = params
        self._num_it +=1

        C_vals, P_C_vals, nu, F = self.kalman_filter(params)
        self.aux.save_C(C=C_vals, P_C=P_C_vals, idx=self._Y.index, name=self._name)
        T = len(nu)
        L = 0
        for t in range(T):
            nu_t = nu[t]
            F_t = F[t]
            if det(F_t) > 0:
                F_t_inv = inv(F_t)
                L += float(nu_t.T.dot(F_t_inv).dot(nu_t)) +np.log(det(F_t))
            else:
                L = np.nan
                break
        L *= 1/2
        print(str(self._num_it) + ":", L)
        d_params = self.dict_params(params=params)
        self.aux.save_params(params=d_params, L=L, name=self._name, name_cols=self._columns)
        return L

    def save_se(self, mle):
        se = list(np.sqrt(np.diag(mle.hess_inv)))
        params = self.dict_params(params=self.params)
        params = params.keys()
        se = dict(zip(params,se))
        se = pd.DataFrame(se, index=[0])
        se.to_csv(self.aux.results_path + self._name + "se.csv")

    def dict_params(self, params) -> dict:
        beta, H, phi1, phi2, gamma1, gamma2, gamma3, gamma4, d11, d12, d13, d14, d21, d22, d23, d24, sigma1, sigma2, sigma3, sigma4 = params
        my_dict ={
            "beta" : beta,
            "H": H, 
            "phi1": phi1,
            "phi2": phi2,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "gamma3": gamma3,
            "gamma4": gamma4,
            "d11": d11,
            "d12": d12,
            "d13": d13,
            "d14": d14,
            "d21": d21,
            "d22": d22,
            "d23": d23,
            "d24": d24,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma3": sigma3,
            "sigma4": sigma4
        }
        return my_dict

    def dict_se(self, se) -> dict:
        return self.dict_params(se)

    @property
    def bounds(self):
        bound_beta = (-1,1)
        bound_H    = (0,1)
        bound_phi1 = bound_beta
        bound_phi2 = bound_beta   
        bound_gamma1 = bound_beta
        bound_gamma2 = bound_beta
        bound_gamma3 = bound_beta
        bound_gamma4 = bound_beta
        bound_d11    = bound_beta
        bound_d12    = bound_beta
        bound_d13    = bound_beta
        bound_d14    = bound_beta
        bound_d21    = bound_beta
        bound_d22    = bound_beta
        bound_d23    = bound_beta
        bound_d24    = bound_beta
        bound_sigma1 = bound_H
        bound_sigma2 = bound_H
        bound_sigma3 = bound_H
        bound_sigma4 = bound_H
        bounds = bound_beta + bound_H + bound_phi1 + bound_phi2
        bounds = bounds + bound_gamma1 + bound_gamma2 + bound_gamma3 + bound_gamma4
        bounds = bounds + bound_d11 + bound_d12 + bound_d13 + bound_d14
        bounds = bounds + bound_d21 + bound_d22 + bound_d23 + bound_d24
        bounds = bounds + bound_sigma1 + bound_sigma2 + bound_sigma3 + bound_sigma4
        return bounds
    
    @property
    def params(self):
        return self._params

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def H(self) -> float:
        return self._H
    
    @property
    def phi(self) -> list:
        return self._phi
    
    @property
    def gamma(self) -> list:
        return list(self._gamma)

    @property
    def D1(self) -> list:
        D1_list = []
        for i in range(self._D1.shape[0]):
            D1_list.append(self._D1[i,i])
        return D1_list

    @property
    def D2(self) -> list:
        D2_list = []
        for i in range(self._D2.shape[0]):
            D2_list.append(self._D2[i,i])
        return D2_list

    @property
    def Sigma(self) -> list:
        Sigma_list = []
        for i in range(1,self._Sigma.shape[0]):
            Sigma_list.append(self._Sigma[i,i])
        return Sigma_list