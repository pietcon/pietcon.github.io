#### Packages
import os
import pandas as pd
import statsmodels.api as sm

# ──────────────────────────────
# 1. Global parameters
# ──────────────────────────────
main_var        = 'CDX_US_HY_spread'
first_datapoint = '2001-11-01'
last_datapoint  = '2023-09-01'
path_in         = os.path.join('regressoes', '0inputs/')
path_out        = os.path.join('regressoes', 'cdx_us_hy', 'outputs/')

#### OUTROS
stats_UR_threshold = 0.15
n_digits = 4; d_intercept = True
lag_lead_spam = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
##############################
##############################

# ###########
# # Data
# # IN: DF horizontal, Vars Y e X empilhadas
df = pd.read_excel(path_in + 'out_DB_M.xlsx', index_col=0)           

init_date = pd.DataFrame()
for el in df:
    init_date.loc[el, 'series start (m/y)'] = str(df[el].dropna().index[0].month) + '/' + str(df[el].dropna().index[0].year)

df = df[(df.index >= first_datapoint) & (df.index <= last_datapoint)]
#df.drop(columns=['intercept'], inplace=True)
###########

keep_cols = [x for x in df.columns if x != 0]
df = df[keep_cols]

# Defining explanatory variables
vars_Y = [x for x in df.columns if x[:7] == main_var[:7]]
vars_X = [x for x in df.columns if x not in vars_Y]
sign = str(100*stats_UR_threshold) + '%'
ans_UR = 'UR at' + sign
ans_Coint = 'Cointegration to Y at' + sign

#### Calculating simple statistics and checking for cointegration
output = pd.DataFrame()
output_ll_coef = pd.DataFrame()
output_ll_pval = pd.DataFrame()
output_ll_r2 = pd.DataFrame()
for el in df:
    X = df[el]; y = df[main_var]

    if len(X.dropna()) < len(X):
        X.dropna(inplace=True)
        y = y[y.index >= X.index[0]]
        y = y[y.index <= X.index[-1]]
    
    #### UNIVARIATE ANALYSIS
    output.loc[el, 'sample start (m/y)'] = str(X.index[0].month) + '/' + str(X.index[0].year)
    output.loc[el, 'last (m/y)'] = str(X.index[-1].month) + '/' + str(X.index[-1].year)
    output.loc[el, 'mean'] = round(X.mean(), n_digits)
    output.loc[el, 'std'] = round(X.std(), n_digits)

    #### UNIT ROOT ANALYSIS
    # Augmented Dickey Fuller test
    p_val_unit_ADF = sm.tsa.stattools.adfuller(X, regression='c')[1]
    output.loc[el, 'pval_ADR'] = round(p_val_unit_ADF, 3)

    if p_val_unit_ADF < stats_UR_threshold: output.loc[el, 'ADF - ' + ans_UR] = 'no UR'
    else:                                   output.loc[el, 'ADF - ' + ans_UR] = 'yes UR'

    # Augmented Dickey Fuller test
    #p_val_unit_KPSS = sm.tsa.stattools.kpss(X, regression='c')[1]
    #output.loc[el, 'pval_KPSS_ADF'] = round(p_val_unit_KPSS, 3)

    #if p_val_unit_KPSS < stats_UR_threshold: output.loc[el, 'KPSS - ' + ans_UR] = 'yes UR'
    #else:                                    output.loc[el, 'KPSS - ' + ans_UR] = 'no UR'

    #### COINTEGRATION ANALYSIS
    if el not in vars_Y:
        p_val_coint = sm.tsa.stattools.coint(y0=y, y1=X, trend='ct')[1]
        
        if p_val_coint < stats_UR_threshold: p_val_coint_ans = 'yes stat. resid'
        else:                                p_val_coint_ans = 'no stat. resid'

        output.loc[el, 'pval_Coint'] = round(p_val_coint, n_digits)
        output.loc[el, ans_Coint] = p_val_coint_ans

    X = X.to_frame()
    if d_intercept: X['intercept'] = 1
    #### MULTIVARIATE ANALYSIS
    if el not in vars_Y:
        reg = sm.OLS(endog=y, exog=X).fit()
        output.loc[el, 'R2'] = round(reg.rsquared, n_digits)
        output.loc[el, 'coef'] = round(reg.params[0], n_digits)
        output.loc[el, 'pval'] = round(reg.pvalues[0], n_digits)

    #### LEAD and LAG ANALYSIS
    if el not in vars_Y:
        for ll in lag_lead_spam:
            X_ll = X.shift(-ll).dropna()
            y_ll = y.copy()
            y_ll = y_ll[y_ll.index >= X_ll.index[0]]
            y_ll = y_ll[y_ll.index <= X_ll.index[-1]]           
            reg = sm.OLS(endog=y_ll, exog=X_ll).fit()

            if ll < 0: suffix = 'lag' + str(abs(ll))
            elif ll > 0: suffix = 'lead' + str(abs(ll))
            else: suffix = 'contemp'

            output_ll_coef.loc[el, 'coef_' + suffix] = round(reg.params[0], n_digits)
            output_ll_pval.loc[el, 'pval_' + suffix] = round(reg.pvalues[0], n_digits)
            output_ll_r2.loc[el, 'R2_' + suffix] = round(reg.rsquared, n_digits)


pd.concat([init_date, output], axis=1).to_excel(path_out + main_var + '_simple_analysis.xlsx')
output_ll = pd.concat([output_ll_pval, output_ll_coef, output_ll_r2], axis=1)
output_ll.to_excel(path_out + main_var + '_lead_lag_analysis.xlsx')


######## UNIT ROOT TESTS
# H0: there is a unit root
# trend: {“c”,”ct”,”ctt”,”n”} 
# “c” : constant, “ct” : constant and linear trend
# “ctt” : constant, and linear and quadratic trend.
# “n” : no constant, no trend.

# The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, 
# with the alternative that there is no unit root. 
# If the pvalue is above a critical size, then we cannot reject that there is a unit root.

######## COINTEGRATION TESTS
# H0: no cointegration
# trend: {“c”, “ct”} “c” : constant, “ct” : constant and linear trend

# Test for no-cointegration of a univariate equation.
# The null hypothesis is no cointegration. 
# Variables in y0 and y1 are assumed to be integrated of order 1, I(1).
# This uses the augmented Engle-Granger two-step cointegration test. 
# Constant or trend is included in 1st stage regression, i.e. in cointegrating equation.