import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Aux:

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
        x.index = pd.to_datetime(x.index)
        return x

    def save_myfig(self,name: str,fig: plt.figure) -> None:
        save_path= '/Users/ruben/Library/Mobile Documents/com~apple~CloudDocs/phd/uni/courses/2nd year/advanced-econometris-2/stock-watson1988_replication/tex/fig/'
        fig.savefig(save_path+name, format="eps")

    def find(self, name: str, subpath='data') -> str:
        load_path='/Users/ruben/Library/Mobile Documents/com~apple~CloudDocs/phd/uni/courses/2nd year/advanced-econometris-2/stock-watson1988_replication/'+subpath+'/'
        return load_path + name