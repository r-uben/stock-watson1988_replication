import pandas as pd
import numpy as np

class Table(object):

    def __init__(self, name) -> None:
        self._tab_path  = "/Users/ruben/Library/Mobile Documents/com~apple~CloudDocs/phd/uni/courses/2nd year/advanced-econometris-2/stock-watson1988_replication/tex/tab/"
        self._results_path = "/Users/ruben/Library/Mobile Documents/com~apple~CloudDocs/phd/uni/courses/2nd year/advanced-econometris-2/stock-watson1988_replication/results/"

        self._name      = name
        
    
    def write_tab(self):
        ts_vars = pd.read_csv(self.file_ts_vars_params, index_col=0)
        se_df   = pd.read_csv(self.file_se, index_col=0)
        L       = pd.read_csv(self.file_L)
        L       = round(float(L["L"]),3)
        vars = ts_vars.columns
        n    = len(vars)

        with open(self.tex_file, 'w') as f:
            self.begin_table(f)
            self.new_line(f)
            self.centering(f)
            self.caption_setup(f)
            self.new_line(f)
            self.caption(f, L)
            self.label(f, "tab:ml-params1")
            self.new_line(f)
            self.begin_tabular(f,n)
            self.new_line(f)
            self.header(f, vars)
            for i in range(len(ts_vars)):   
                param = ts_vars.index[i]
                idx_param = self.formatter_param(param)
                for s in [0,1]:
                    if s==0: f.write(idx_param)
                    self.amp(f)
                    count=1
                    for var in ts_vars.columns:
                        val = str(ts_vars.loc[param, var])
                        se  = "(" + str(se_df.loc[param, var]) + ")"
                        if s==0: f.write(val)
                        else: f.write(se)
                        if count<n: self.amp(f)
                        else: self.new_row(f)
                        count+=1
                    if i < len(ts_vars)-1: self.new_line(f)
                
            self.hline(f,2)
            self.new_line(f)
            self.end_tabular(f)
            self.new_line(f)
            self.end_table(f)

    def formatter_param(self, param):
        # First: replace the number, if any, for _{number:
        for i in [1,2]:
            param = param.replace(str(i), "_{" + str(i))
        # Second: if no number, just remember it
        # Third: if no number, replace the subscript 'i' by '_i'
        if "_{" not in param: param = param[:-1] + "_i"
        # if number, then just replace the (last) "i" by "i}"
        if "_{" in param: param = param[:-1] + "i}"
        # Fourth: greek letters?
        for letter in self.greek_letters:
            if letter in param: 
                first_letter = param[0]
                param = param.replace(first_letter, "\\" + first_letter)
        # Fifth: add dolars
        param = self.between_dolars(param)
        return param

    def header(self, f, vars):
        count=1
        n = len(vars)
        self.amp(f)
        for var in vars:
            f.write(var)
            if count<n: self.amp(f)
            else: self.new_row(f)
            count+=1
        self.hline(f,2)
        self.new_line(f)

    def begin_table(self, f) -> None:
        f.write("\\begin{table}[h!]")

    def end_table(self, f) -> None:
        f.write("\\end{table}")
    
    def begin_tabular(self, f, ncols) -> None:
        f.write("\\begin{tabular}{l|" + (ncols)*"c" + "}")

    def end_tabular(self, f) -> None:
        f.write("\\end{tabular}")

    def end_table(self, f) -> None:
        f.write("\\end{table}")

    def centering(self, f) -> None:
        f.write("\\centering")
    
    def caption(self, f, L) -> None:
        f.write("\\caption{")
        f.write("The estimation period is 1959:2-1983:12. The parameters were estimated by Gaussian maximum likelihod as described in the text. The parameters are $\gamma = (\gamma_1,\ldots, \gamma_4)$, $D(L)=\\text{diag}\left(d_1(L),\ldots, d_4(L)\\right)$, where $d_i(L) = 1-d_{i1}L - d_{12}L^2$ and $\Sigma = \\text{diag} \left(1,\sigma_1^2,\ldots,\sigma_4^2\\right)$. Maximum likelihood is $\mathcal{L}=" + str(L) + "$.")
        f.write("}")
    
    def caption_setup(self, f) -> None:
        f.write("\captionsetup{width=0.6\\textwidth, font=small}")

    def label(self, f, lab):
        f.write("\\label{" + lab + "}")

    def new_line(self, f) -> None:
        f.write("\n")

    def amp(self,f) -> None:
        f.write("&")

    def new_row(self, f) -> None:
        f.write("\\\\")
    
    def hline(self, f, nlines) -> None:
        f.write(nlines*"\\hline")

    def between_dolars(self, x: str) -> str:
        return "$" + x + "$"

    @property
    def my_file(self) -> str:
        return self.tab_path + self._name

    @property
    def tex_file(self) -> str:
        return self.my_file + ".tex"

    @property 
    def tab_path(self) -> str:
        return self._tab_path

    @property
    def file_ts_vars_params(self) -> str:
        return self.results_path + "ts_vars.csv"

    @property
    def file_se(self) -> str:
        return self.results_path + "se.csv"

    @property
    def file_L(self) -> str:
        return self.results_path + "L.csv"

    @property 
    def results_path(self) -> str:
        return self._results_path

    @property
    def greek_letters(self) -> list:
        gr = [
            "alpha", 
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "varepsilon",
            "dseta",
            "eta",
            "zeta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "ro",
            "sigma",
            "tau",
            "ipsilon",
            "phi",
            "varphi"
            "xi",
            "psi",
            "omega"
        ]
        return gr


