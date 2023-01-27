import pandas   as pd
import numpy    as np
import eikon    as ek
import math     as m
import matplotlib.pyplot as plt


from kf import KF
from ml import ML
from matplotlib.animation import FuncAnimation
from random import randrange
from warnings import filterwarnings
from numpy.linalg import det, inv, LinAlgError

pd.set_option('display.width', 40)
filterwarnings('ignore')

# DISPLAYERS:

def display_control(it, params, L):
    print("\n-------------\n")
    print("Num.iter.=", it, "\n\n", pd.DataFrame(params), "\n")
    display_L(L)
    print("\n-------------\n\n")

def display_L(L, it=None) -> None:
    if it is None:
        print("Likelihood: ", L)
    else: 
        print(str(it)+": ","Likelihood:", L)

# DATA:

def log_df(df):
    for col in df.columns:
        df[col] = np.log(df[col])
    df = df.replace(-np.inf,np.nan).dropna()
    return df

def Δ(df):
    df =df.diff()
    df = df.dropna()
    return df

def standardise(dataframe):
    for col in dataframe.columns:
        dataframe[col] = (dataframe[col]-dataframe[col].mean()) / dataframe[col].std()
    return dataframe

def ts_vars(columns=None) -> pd.DataFrame:
    if columns is not None:
        api_key = '4019aab16a6b45cf86e11574d74e2e5b3681ee13'
        ek.set_app_key(api_key)
        Z = ek.get_timeseries(columns,
        start_date='2014-12-15',
        end_date='2022-8-31',
        interval='monthly')
        Z.columns = ["Industrial Production", "Disposable Personal Income", "Trade Sales", "Average Workweek"]
        Z.index = pd.to_datetime(Z.index, format="%Y-%m%-d")
    else:
        Z = pd.read_csv("data/s&w1989.csv", index_col=0)
        Z.index.name = None
    Y = standardise(Δ(log_df(Z)))
    return Y

# PARAMETERS:

def param_space(n, p, N=int(1e6)):
    default_sample = np.linspace(-1,1,num=N)
    default_positive_sample = np.linspace(1e-20,0.5,num=N)

    betas = default_sample
    Hs = default_positive_sample
    # k=2
    d1s = {}
    d2s = {}
    sigmas = {}
    gammas = {}
    for i in range(1,n+1):
        d1s[i] = default_sample
        d2s[i] = default_sample    
        sigmas[i] = default_positive_sample
        gammas[i] = default_sample

    phis = {}
    for j in range(1,p+1):
        phis[j] = default_sample
    return betas, d1s, d2s, sigmas, gammas, phis, Hs

def take_random_params(phis,d1s,d2s,sigmas,betas,gammas, Hs):
    n = len(gammas.keys())
    p = len(phis.keys())
    phi = {}
    gamma = {}
    sigma = {}
    d1 = {}
    d2 = {}
    for j in range(1,p+1): phi[j] = phis[j][randrange(len(phis[2]))]
    for i in range(1,n+1): gamma[i] = gammas[i][randrange(len(gammas[i]))]
    for i in  range(1,n+1): sigma[i] = sigmas[i][randrange(len(sigmas[i]))]
    for i in range(1,n+1):
        d1[i] = d1s[i][randrange(len(d1s[i]))]
        d2[i] = d1s[1][randrange(len(d2s[i]))]

    beta = betas[randrange(len(betas))]
    H = Hs[randrange(len(Hs))]

    # params:
    phi = [phi[i] for i in range(1,p+1)]
    D = {
        1: np.diag([d1[i] for i in range(1,n+1)]),
        2: np.diag([d2[i] for i in range(1,n+1)])
    }
    Sigma = np.diag([1] +[sigma[i]**2 for i in range(1,n+1)])
    gamma = np.array([gamma[i] for i in range(1,n+1)]).T
    return  phi, D, Sigma, beta, gamma, H

def update_params(  beta: float,
                        phi: list,
                        D: dict,
                        Sigma: np.array,
                        gamma: np.array,
                        H: float) -> dict:
    d1 = D[1]
    d2 = D[2]
    params = {}
    params["beta"]  = [beta]
    params["H"]  = [H]
    for i in range(len(d1)):
        i=int(i)
        params["d1%i"%(i+1)] = [d1[i,i]]
        params["d2%i"%(i+1)] = [d2[i,i]]
        params["sigma%i"%(i+1)] = [np.sqrt(Sigma[i+1,i+1])]
        params["gamma%i"%(i+1)] = [gamma[i]]
    for j in range(len(phi)):
        j=int(j)
        params["phi%i"%(j+1)] = [phi[j]]    
    
    for x in params.keys():
        params[x] = [round(params[x][0], 4)]

    return params

# KALMAN FILTER:

def kf_parameters(params):
    kf_params = {}

    beta = params['beta'][0]
    H =  params['H'][0]
    phi = [params['phi1'][0],params['phi2'][0]]
    D = {
        1: np.diag([params['d11'][0],params['d12'][0],params['d13'][0],params['d14'][0]]),
        2: np.diag([params['d21'][0],params['d22'][0],params['d23'][0],params['d24'][0]])
    }
    Sigma = np.diag([1,params['sigma1'][0]**2,params['sigma2'][0]**2,params['sigma3'][0]**2,params['sigma4'][0]**2])
    gamma = np.array([params['gamma1'][0],params['gamma2'][0],params['gamma3'][0],params['gamma4'][0]]).T
    
    kf_params["beta"] = beta
    kf_params["H"] = H
    kf_params['phi'] = phi
    kf_params['D'] = D
    kf_params['Sigma'] = Sigma
    kf_params['gamma'] = gamma
    return kf_params 

def kalman_filter(Y, kf_params) -> tuple:
    # params
    phi     = kf_params['phi']
    D       = kf_params['D']
    Sigma   = kf_params['Sigma']
    beta    = kf_params['beta']
    H       = kf_params['H']
    gamma   = kf_params['gamma']
    # Initial values:
    C_vals = []
    P_C_vals = []

    # Forecast errors:
    nu = []
    F  = []
    for t in range(len(Y.index)):
        y = Y.iloc[t].to_numpy().reshape(len(Y.columns),1) # I need this because neither .T nor .transpose() is working
        if t==0:
            kf = KF(Y=y,phi=phi, D=D, Sigma=Sigma, beta=beta, gamma=gamma, H=H)
        else:
            kf = KF(Y=y,phi=phi, D=D, Sigma=Sigma, beta=beta, gamma=gamma, H=H,alpha=alpha, P=P)
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

def L(nu, F) -> float:
    T = len(nu)
    L = 0
    for t in range(T):
        nu_t = nu[t]
        F_t = F[t]
        if det(F_t) > 0:
            F_t_inv = inv(F_t)
            L += -float(nu_t.T.dot(F_t_inv).dot(nu_t)) - np.log(det(F_t))
        else:
            L = np.nan
            break
    L *= 1/2
    return L

# SAVERS:

def save_params(params, L, names=["IP", "DPI", "TS", "AW"]) -> None:
    ts_vars_df = pd.DataFrame(index=["gammai", "d1i", "d2i", "sigmai"], columns=names)
    C_params_df = pd.DataFrame(index = ["value"], columns=["phi1", "phi2"])
    L_df = pd.DataFrame(index=["value"], columns=["L"])

    for i in range(len(ts_vars_df.columns)):
        # the [0] is for taking just the number and not the [number]
        ts_vars_df.loc["$\gamma_i$", ts_vars_df.columns[i]] = params["gamma%i"%(i+1)][0]
        ts_vars_df.loc["$d_{1 i}$", ts_vars_df.columns[i]] = params["d1%i"%(i+1)][0]
        ts_vars_df.loc["$d_{2 i}$", ts_vars_df.columns[i]] = params["d2%i"%(i+1)][0]
        ts_vars_df.loc["$\sigma_i$", ts_vars_df.columns[i]] = params["sigma%i"%(i+1)][0]
 
    for j in range(2):
        C_params_df.loc["value", "phi%i"%(j+1)] = params["phi%i"%(j+1)][0]
    
    L_df.loc["value", "L"] = round(L,4)

    ts_vars_df.to_csv("results/ts_vars.csv", index=True)
    C_params_df.to_csv("results/C_params.csv", index=False)
    L_df.to_csv("results/L.csv", index=False)
    
def save_L(Ls,i) -> None:
    Ls_df = pd.DataFrame()
    Ls_df["n_iter"] = i
    Ls_df["L"] = Ls
    Ls_df.to_csv("results/Ls.csv")

def save_C(C, P_C, idx):
    C = {"C": C, "P": P_C}
    C = pd.DataFrame(C, columns=["C", "P"], index=idx)
    C.to_csv("results/C.csv")

# CONDITIONS: 

def min_is_not_the_min_anymore(min_value, new_value)->bool:
    return min_value > new_value

def main():
    # Let's take the time series variables. Thiese variables, as in Stock and Watson (1989), consists of the growth rates of 
    # the logarithms of (describe it).
    Y = ts_vars() 
    n = len(Y.columns)
    # Space of params
    betas, d1s, d2s, sigmas, gammas, phis, Hs = param_space(n,2,int(3e5))
    phi, D, Sigma, beta, gamma, H = take_random_params(phis,d1s,d2s,sigmas,betas,gammas, Hs)
    params = update_params(beta,phi,D,Sigma,gamma, H)
    kf_params  = kf_parameters(params)
    ml = ML(Y=Y, kf_params=kf_params)
    print(ml.optimisation())
    # Initiation:
    first_iteration = True
    max_it = int(2e6)
    it=1
    i=[]
    Ls = []
    # for it in range(max_it):
    #     # Randomly choose a parameter
    #     phi, D, Sigma, beta, gamma, H = take_random_params(phis,d1s,d2s,sigmas,betas,gammas, Hs)
    #     params = update_params(beta,phi,D,Sigma,gamma, H)
    #     try: 
    #         kf_params               = kf_parameters(params)
    #         C_vals, P_C_vals, nu, F = kalman_filter(Y, kf_params) # Apply the Kalman filter
    #         new_L                   = -L(nu,F)  # Calculate the Maximum likelihood
    #         if ~((np.isnan(new_L)) or (np.isinf(new_L))):
    #             if it%100==0: display_L(new_L, it) # Display each 100 iterations
    #             if first_iteration:
    #                 min_L = new_L
    #                 first_iteration = False
    #             elif min_is_not_the_min_anymore(min_L,new_L) and not first_iteration: # minimising the -L
    #                 min_L = new_L
    #                 display_control(it, params, new_L)
    #                 # Create and save the dataframe for the index:
    #                 save_C(C_vals, P_C_vals, Y.index)
    #                 # Save L:
    #                 Ls.append(min_L)
    #                 i.append(it)
    #             else:
    #                 continue
    #             save_L(Ls,i)
    #     except(LinAlgError):
    #         continue


if __name__ == "__main__":
    main()

