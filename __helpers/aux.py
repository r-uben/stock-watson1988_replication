import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Aux:

    def __init__(self) -> None:
        self._main_path    = "/Users/ruben/Library/Mobile Documents/com~apple~CloudDocs/phd/uni/courses/2nd year/advanced-econometris-2/stock-watson1988_replication/"
        self._results_path = self.folder("results/")
        self._fig_path     = self.folder("tex/fig/")
    

    def find(self, name, subfolder):
        folder = self.folder(subfolder)
        return folder + name

    def folder(self, subpath):
        return self._main_path + subpath + '/'

    def aurum_fig(self) -> plt.figure:
        phi = (1+np.sqrt(5))/2
        a = 5
        b = a/phi
        fig=plt.figure(figsize=(a+b,a))
        return fig

    def log_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for col in dataframe.columns:
            dataframe[col] = np.log(dataframe[col])
        dataframe = dataframe.replace(-np.inf,np.nan).dropna()
        return dataframe

    def Δ(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe =dataframe.diff()
        dataframe = dataframe.dropna()
        return dataframe

    def standardise(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for col in dataframe.columns:
            dataframe[col] = (dataframe[col]-dataframe[col].mean()) / dataframe[col].std()
        return dataframe

    def my_idx(self,x: pd.DataFrame) -> pd.DataFrame:
        x.index = pd.date_range("1959-1",  periods=len(x), freq='M').strftime('%Y-%m')
        return x

    def dt_idx(self, x) -> pd.DataFrame:
        x.index = pd.to_datetime(x.index, format='%Y-%m-%d')
        return x

    def save_myfig(self, name: str,fig: plt.figure) -> None:
        if ".eps" not in name: name += ".eps"
        fig.savefig(self._fig_path+name, format="eps")

    def save_params(self, params, L, name, name_cols=["IP", "GMYX", "LPMH", "LPMHU"]) -> None:

        gammai = "gammai"
        d1i    = "d1i"
        d2i    = "d2i"
        sigmai = "sigmai"

        ts_vars_df = pd.DataFrame(index=[gammai], columns=name_cols)
        C_params_df = pd.DataFrame(index = ["value"], columns=["phi1", "phi2"])
        L_df = pd.DataFrame(index=["value"], columns=["L"])

        for i in range(len(ts_vars_df.columns)):
            ts_vars_df.loc[gammai, ts_vars_df.columns[i]] = round(params["gamma%i"%(i+1)],4)
            ts_vars_df.loc[d1i, ts_vars_df.columns[i]]  = round(params["d1%i"%(i+1)],4)
            ts_vars_df.loc[d2i, ts_vars_df.columns[i]]  = round(params["d2%i"%(i+1)],4)
            ts_vars_df.loc[sigmai, ts_vars_df.columns[i]] = round(params["sigma%i"%(i+1)],4)
    
        for j in range(2):
            C_params_df.loc["value", "phi%i"%(j+1)] = round(params["phi%i"%(j+1)],4)
        
        L_df.loc["value", "L"] = round(L,4)

        ts_vars_df.to_csv(self._results_path + name + "ts_vars.csv", index=True)
        C_params_df.to_csv(self._results_path + name + "C_params.csv", index=False)
        L_df.to_csv(self._results_path + name + "L.csv", index=False)

    def save_se(self, se, name, name_cols=["IP", "GMYX", "LPMH", "LPMHU"]):
        gammai = "gammai"
        d1i    = "d1i"
        d2i    = "d2i"
        sigmai = "sigmai"

        se_df = pd.DataFrame(index=[gammai], columns=name_cols)

        for i in range(len(se_df.columns)):
            se_df.loc[gammai, se_df.columns[i]] = round(se["gamma%i"%(i+1)],4)
            se_df.loc[d1i, se_df.columns[i]]  = round(se["d1%i"%(i+1)],4)
            se_df.loc[d2i, se_df.columns[i]]  = round(se["d2%i"%(i+1)],4)
            se_df.loc[sigmai, se_df.columns[i]]  = round(se["sigma%i"%(i+1)],4)

        se_df.to_csv(self.results_path + name + "se.csv")

    def save_L(self,Ls,i,name) -> None:
        Ls_df = pd.DataFrame()
        Ls_df["n_iter"] = i
        Ls_df["L"] = Ls
        Ls_df.to_csv(self._results_path + name + "Ls.csv")

    def save_C(self,C, P_C, idx, name):
        C = {"C": C, "P": P_C}
        C = pd.DataFrame(C, columns=["C", "P"], index=idx)
        C.to_csv(self._results_path + name + "C.csv")

    @property
    def results_path(self):
        return self._results_path
    
    @property
    def fig_path(self):
        return self._fig_path
    
    @property
    def main_path(self):
        return self._main_path