import numpy as np
import pandas as pd

from numpy.linalg import inv, matrix_power


from __helpers.exceptions import WrongInput
class KF(object):


    def __init__(self,  Y: np.array, # We only want one step to the other so we are iterating outside this class
                        phi: list or np.array,
                        D: zip,
                        Sigma: np.array,
                        beta: float,
                        gamma: np.array,
                        H: float,
                        alpha=None,
                        P=None,
                        ) -> None:

        # Vector of data (increments of logs)
        if (Y.shape[0] == 1): Y = Y.T
        self._Y = Y
        # phi should be a row vector (np.array)
        if type(phi)==list: self._phi = np.array([phi])
        else: 
            if (phi.shape[0] > 1): 
                self._phi = phi.reshape(1, phi.shape[0])
        self._p = int(self._phi.shape[1])
        # D should be a dictionary of matrices
        try:
            self.isdict(type(D))
            self._D = D
        except WrongInput as ex: 
            print(ex)

        self._k = int(len(self._D))
        # Number of variables used to estimate Ct
        self._n = int(Y.shape[0])

        # Variance of the shocks:
        self._Sigma = Sigma
        # Constants:
        self._beta  = beta #beta # beta
        self._H     = H
        self._gamma = gamma.reshape(self._n, 1)

        # Initial Values:
        if alpha is None:
            self._alpha = np.zeros((self.p+self.n*self.k+1,1))
            self._P     = 0.2*np.eye(self.p+self.n*self.k+1)
        else:
            self._alpha = alpha
            self._P = P
        self._pred_alpha = self._alpha
        self._pred_P     = self._P

    def predict(self) -> None:
        # alpha = T alpha
        # P     = T P T' + R Sigma R'
        new_alpha = self.T.dot(self._alpha)
        new_P     = self.T.dot(self._P).dot(self.T.T) + self.Q
        
        self._pred_alpha = new_alpha
        self._pred_P     = new_P

    def update(self):
        # alpha = pred_alpha + pred_P Z' F_inv nu
        # P     = pred_P    + pred_P Z' F_inv Z pred_P
        new_alpha   = self._pred_alpha + self.K.dot(self.nu) # + H=0
        new_P       = self._pred_P - self.K.dot(self.Z).dot(self._pred_P) 

        self._alpha = new_alpha
        self._P     = new_P

    @property
    def nu(self) -> np.array:
        return self._Y - self._beta*np.ones((self.n,1)) - self.Z.dot(self._pred_alpha)

    @property
    def F(self) -> np.array:
        return self.Z.dot(self._pred_P).dot(self.Z.T) + self._H
    
    @property
    def K(self) -> np.array:
        F_inv = inv(self.F)
        return self._pred_P.dot(self.Z.T).dot(F_inv)

    @property
    def Q(self) -> np.array:
        return self.R.dot(self._Sigma).dot(self.R.T)

    @property
    def T(self):
        # top
        top_left        = self.Phi_star
        top_center      = np.zeros((self.p, self.n*self.k))
        top_right       = np.zeros((self.p,1))
        top             = np.concatenate((top_left, top_center, top_right), axis=1)
        # middle
        middle_left     = np.zeros((self.n*self.k,self.p))
        middle_center   = self.D_star
        middle_right    = np.zeros((self.n*self.k,1)) 
        middle          = np.concatenate((middle_left, middle_center, middle_right), axis=1)
        # bottom
        bottom_left     = self.Zc
        bottom_center   = np.zeros((1,self.n*self.k))
        bottom_right    = np.ones((1,1))
        bottom          = np.concatenate((bottom_left, bottom_center, bottom_right), axis=1)
        return np.concatenate((top,middle,bottom), axis=0)
    
    @property
    def R(self):
        # aux
        ones = np.ones((self.p,1))
        # top
        top_left    = self.Zc.T
        top_right   = np.zeros((self.p,self.n))
        top         = np.concatenate((top_left, top_right), axis=1)
        # middle 
        middle_left     = np.zeros((self.k*self.n, 1))
        middle_right    = self.Zu.T
        middle          = np.concatenate((middle_left, middle_right), axis=1)
        # bottom    
        bottom_left     = np.zeros((1,1))
        bottom_right    = np.zeros((1,self.n))
        bottom          = np.concatenate((bottom_left, bottom_right), axis=1)
        return np.concatenate((top,middle,bottom),axis=0) 

    @property
    def Phi_star(self):
        ''' Phi_star is a pxp matrix'''
        I = np.eye(self.p-1)
        zeros = np.zeros((self.p-1,1))
        top = self._phi
        bottom = np.concatenate((I,zeros),axis=1)
        return np.concatenate((top,bottom),axis=0)

    @property
    def D_star(self):
        ''' D_star is a (nk)x(nk) matrix'''
        top     = self._D[1]
        for i in range(2,self.k+1):
            top = np.concatenate((top,self._D[i]),axis=1)
        I       = np.eye(self.n*(self.k-1))
        zeros   = np.zeros((self.n*(self.k-1), self.n))
        bottom  = np.concatenate((I,zeros),axis=1)
        return np.concatenate((top,bottom),axis=0)

    @property
    def Zc(self):
        ''' Zc is a 1xp vector'''
        left  = np.ones((1,1))
        right = np.zeros((1,self.p-1)) 
        return np.concatenate((left,right),axis=1)

    @property
    def Zu(self):
        ''' Zu is a nxnk matrix'''
        left = np.eye(self.n)
        right = np.zeros((self.n,self.n*(self.k-1)))
        return np.concatenate((left,right), axis=1)

    @property
    def Z(self):
        '''Z=[ gamma*Zc Zu 0 ]'''
        left    = self._gamma.dot(self.Zc)
        middle  = self.Zu
        right   = np.zeros((self.n,1))
        return np.concatenate( (left, middle, right), axis=1)

    ## OUTPUT:
    @property
    def alpha(self) -> np.array:
        return self._alpha
    
    @property
    def P(self) -> np.array:
        return self._P
    
    @property
    def C(self) -> float:
        '''C_{t|t} = [Zc 0 1] alpha'''
        left = np.concatenate([self.Zc,np.zeros((1,self.k*self.n)),np.ones((1,1))], axis=1)
        right = self.alpha 
        return float(left.dot(right))

    @property
    def P_C(self) -> float:
        '''C_{t|t} = [Zc 0 1] alpha'''
        left = np.concatenate([self.Zc,np.zeros((1,self.k*self.n)),np.ones((1,1))], axis=1)
        center = self.P 
        right = left.T
        return float(left.dot(center).dot(right))

    ## PARAMETERS:
    @property
    def p(self):
        '''p = dim(Ct*), i.e., number of lags of Ct used in Ct*.'''
        return self._p

    @property
    def k(self) -> int:
        '''
            k = dim(ut*)/n, i.e., number of lags of ut used in ut*.
        '''
        return self._k

    @property
    def n(self) -> int:
        '''
            n = dim(Yt), i.e., number of variables used to 
            estimate the 'unobserved' index Ct.
        '''
        return self._n

    # Functions to get exceptions:
    def isdict(self, my_type) -> None:
        if my_type is not dict: raise WrongInput(my_type, dict)
    
    def isarray(self, my_type) -> None:
        if my_type is not np.array: raise WrongInput(my_type, np.array)
