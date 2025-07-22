import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

########################################################################
### REGRESSION #########################################################
########################################################################
def reg_box_jenkins(df, name, path=None, intercept=False, plot_Xs=False, rec_bol=False):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    ##################################
    ######### FUNTION SET UP #########
    # Turn interactive plotting off 
    plt.ioff()  
    
    # Creating destination folder
    if path is not None:
        path_out = os.path.join(path, name)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        if not os.path.exists(os.path.join(path_out, 'logs')):
            os.mkdir(os.path.join(path_out, 'logs'))
        if rec_bol:
            if not os.path.exists(os.path.join(path_out, 'reccursive')):
                os.mkdir(os.path.join(path_out, 'reccursive'))
        if plot_Xs:
            if not os.path.exists(os.path.join(path_out, 'univariates')):
                os.mkdir(os.path.join(path_out, 'univariates'))

    # Infer Frequency to set window size
    freq = pd.infer_freq(df.index[6:-6])
    n = len(df)
    if freq is None:   window = n
    elif 'M' in freq:  window = 24
    elif 'Q' in freq:  window = 12
    elif 'A' in freq:  window = 4

    # Adjusting window size  
    if n < window:  window = n

    # Creating intercept
    if intercept:    df['intercept'] = 1

    ##################################
    ######## DATA & REGRESSION #######
    # Data management
    y = df.iloc[:,0].dropna()
    X = df.iloc[:,1:].dropna() 
    dif = len(X) - len(y)
    X_pred = None
    if dif != 0:
        X_pred = X.iloc[-dif:,:]
        X = X.iloc[:-dif,:]
    XX = pd.concat([X, X_pred])

    # OLS Regression using statsmodels to get descriptive statistics
    reg = sm.OLS(endog=y, exog=X).fit()
    # OLS Regression using sklearn to allow for only positive coefficients restriction
    reg_sk = LinearRegression(fit_intercept=False, positive=False).fit(X, y)

    ##################################
    ######## ORGANIZE OUTPUTS ########
    # Collecting data for variable contribution graphs
    graph_contribs = reg_sk.coef_*XX
    
    # Extracting dummy variables from the data
    dummies = [x for x in graph_contribs.columns if 'Dum_' in x]
    contrib_dums = graph_contribs.loc[:, dummies].sum(axis=1)
    not_dummies = [x for x in graph_contribs.columns if 'Dum_' not in x]
    graph_contribs = graph_contribs.loc[:, not_dummies]

    # Adjusting data to show up net from dummy variables
    y_adj = y - contrib_dums.iloc[:len(y)]
    pred = reg_sk.predict(XX) - contrib_dums
    fit = pred.copy()
    resid = y - reg_sk.predict(X)

    # In case an extra amount of data is available for a prediction
    if X_pred is not None:
        fit[-len(X_pred):] = np.nan
        df_pred = pd.DataFrame(data={'Fitted': fit, 
                                     'Estimated': pred})
    else:
        df_pred = pd.DataFrame(data={'Fitted': fit})

    # Collecting data for graphs
    graph_data = pd.concat([pd.DataFrame(data={'Actual': y_adj, 
                                               'Residual': resid}), df_pred], axis=1)

    out_stats = pd.Series(dtype='float64')
    # Collecting statistics for output

    if intercept:
        out_stats['intercept (coef)'] = round(reg_sk.intercept_,4)
        out_stats['intercept (pval)'] = '('+str(round(reg.pvalues[0],3))+')'

    for i, el in enumerate(reg.params.index):
        out_stats[el + ' (coef)'] = round(reg_sk.coef_[i],4)
        out_stats[el + ' (pval)'] = '('+str(round(reg.pvalues[el],3))+')'

    out_stats['R2'] = round(reg_sk.score(X, y),2)
    out_stats['MSE'] = round((resid**2).mean(),2)
    #out_stats['JBpv'] = sm.stats.stattools.jarque_bera(resid, axis=0)[1]
    out_stats['LMpv'] = round(sm.stats.diagnostic.het_breuschpagan(resid, X)[1],2)
    out_stats['AIC'] = round(reg.aic,2)
    out_stats['BIC'] = round(reg.bic,2)
    out_stats['DW'] = round(sm.stats.stattools.durbin_watson(resid, axis=0),2)

    ##################################
    ########## OUTPUT TABLE ##########
    # Saving regression results
    if path is not None:
        summ = reg.summary(); summ_dict = {}
        # Get the whole summary table as DataFrames
        for i in range(3):
            summ_dict[i] = pd.DataFrame(summ.tables[i].data[1:], 
                                        columns=summ.tables[i].data[0])

        # Picking only the DataFrame we want the modified    
        summ_df = summ_dict[1]
        std_err_vec = summ_df['std err']
        for i, el in enumerate(X):
            coef = reg_sk.coef_[i]
            std_err = float(std_err_vec[i])
            if not std_err == 0 and not coef == 0:
                t_stat = coef / std_err; pval = 2 * (1 - t.cdf(abs(t_stat), len(resid)))
                summ_df.loc['coef',el] = str(round(coef,4))
                summ_df.loc['std err',el] = str(round(std_err,3))
                summ_df.loc['t',el] = '(' + str(round(t_stat,3)) + ')'
                summ_df.loc['P>|t|',el] = str(round(pval,3))
            else:
                summ_df.loc['coef',el] = summ_df.loc['t',el] = ''
                summ_df.loc['P>|t|',el] = summ_df.loc['std err',el] = '' 
                summ_df.loc['[0.025',el] = summ_df.loc['0.975]',el] = ''

        summ_dict[1] = summ_df

        with open(os.path.join(path, f'summary_{name}.txt'), 'w') as fh:
            header = 'OLS Regression Results'
            ref_line = summ_dict[0].to_string(index=False).split('\n')[0]
            fh.write('\n' + '=' * len(ref_line.split('\n')[0]) + '\n')
            fh.write(header.center(len(ref_line), ' '))
            fh.write('\n' + '-' * len(ref_line.split('\n')[0]))
            for el_df in summ_dict:
                line = summ_dict[el_df].to_string(index=False)
                fh.write('\n' + '=' * len(line.split('\n')[0]) + '\n')
                fh.write(line)
                fh.write('\n' + '-' * len(line.split('\n')[0]))

    ##################################
    ############# GRAPHS #############
    if rec_bol:
        try:
            # Rolling & Recursive Regression
            rols = RollingOLS(y, X, window=window)
            rres = rols.fit()
            mod = sm.RecursiveLS(y, X)
            rec = mod.fit()
        except:
            rec_bol = False

    ##### SECTION 1
    ### Graph: Actual x Estimated 
    fig, gps = plt.subplots(2, figsize=(16,6), dpi= 100)
    # First graph: Actual x Estimated
    gps[0].set_title('Actual x Estimated - ' + y.name + ' (ex dummies)')
    if X_pred is not None:
        gps[0].plot(graph_data[['Actual', 'Estimated']], 
                    label=['Actual', 'Predicted'])
    else:
        gps[0].plot(graph_data[['Actual', 'Fitted']], 
                    label=['Actual', 'Fitted'])
    gps[0].legend(loc='upper left')
    gps[0].grid(axis='y', alpha=0.7)
    gps[0].axhline(0, color='black', linewidth=0.5)

    # Second graph: Residual
    gps[1].plot(graph_data[['Residual']], label=r'Residual')
    gps[1].legend(loc='upper left')
    gps[1].grid(axis='y', alpha=0.7)
    gps[1].axhline(0, color='black', linewidth=0.5)
    if path is not None: plt.savefig(os.path.join(path_out, 'graph_Act_Est_Resid.jpg'))
    plt.close(fig)

    ##### SECTION 2
    ### Graph: Contributions
    fig, gps = plt.subplots(1, figsize=(16,6), dpi= 100)
    gps.set_title('X variables contributions - ' + y.name)
    gps.plot(graph_contribs, label=graph_contribs.columns)
    gps.legend(loc='upper left')
    gps.grid(axis='y', alpha=0.7)
    gps.axhline(0, color='black', linewidth=0.5)
    if path is not None: 
        plt.savefig(os.path.join(path_out, 'graph_Contribs.jpg'))
        graph_contribs.to_excel(os.path.join(path_out, 'logs', 'tab_Contribs.xlsx'))
    plt.close(fig)

    ### Graph: Marginal Contributions 
    fig, gps = plt.subplots(1, figsize=(16,6), dpi= 100)
    gps.set_title('X variables marginal contributions - ' + y.name)
    gps.plot(graph_contribs.diff(1), label=graph_contribs.columns)
    gps.legend(loc='upper left')
    gps.grid(axis='y', alpha=0.7)
    gps.axhline(0, color='black', linewidth=0.5)
    if path is not None: 
        plt.savefig(os.path.join(path_out, 'graph_Contribs_Marg.jpg'))
        graph_contribs.diff(1).to_excel(os.path.join(path_out, 'logs', 'tab_Contribs_Marg.xlsx'))
    plt.close(fig)

    ##### SECTION 3
    if len(y)/2 < 20: lags_acf = len(y)/2 - 1
    else:             lags_acf = 20
    # Graph: ACF
    fig, gps = plt.subplots(2,figsize=(16,6), dpi= 100)
    plot_acf(y, lags=lags_acf, ax=gps[0], title=r'Autocorrelation - ' + y.name)
    plot_pacf(y, lags=lags_acf, ax=gps[1], method='ywm')
    if path is not None: fig.savefig(os.path.join(path_out, 'graph_PACF_ACF.jpg'))
    plt.close(fig)
    
    # Graph: Residuals ACF
    fig, gps = plt.subplots(2,figsize=(16,6), dpi= 100)
    plot_acf(y - fit, lags=lags_acf, ax=gps[0], title=r'Autocorrelation - Residuals')
    plot_pacf(y - fit, lags=lags_acf, ax=gps[1], method='ywm')
    if path is not None: fig.savefig(os.path.join(path_out, 'graph_PACF_ACF_resid.jpg'))
    plt.close(fig)

    ##### SECTION 4
    # Graph: Explanatory Variables
    if plot_Xs:    
        for el in X:
            if el != 'intercept':
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df[el])
                ax.set_title(el)
                ax.grid(True)
                plt.savefig(os.path.join(path_out, 'univariates', f'graph_var_{el}.jpg'))
                plt.close(fig)

    ##### SECTION 5
    # Graph: Rolling Coefficients
    if rec_bol:
        for ell in X:
            try:
                fig = rres.plot_recursive_coefficient(variables=[ell], figsize=(10, 6))
                fig.savefig(os.path.join(path_out, 'reccursive', f'graph_rollcoef_{ell}_w{window}.jpg'))

                plt.close(fig)
            except: pass
            try:
                fig = rec.plot_recursive_coefficient(variables=[el], figsize=(10, 6))
                fig.savefig(os.path.join(path_out, 'reccursive', f'graph_reccoef_{ell}.jpg'))
                plt.close(fig)
            except: pass

        # Graph: Cumulative Residuals
        try:
            fig = rec.plot_cusum()
            fig.savefig(os.path.join(path_out, 'graph_Cusum.jpg'))
            plt.close(fig)
        except: pass
        try:
            fig = rec.plot_cusum_squares()
            fig.savefig(os.path.join(path_out, 'graph_Cusum_Squares.jpg'))
            plt.close(fig)
        except: pass
    
    ##################################
    ######## FUNCTION OUTPUT #########
    if X_pred is not None:
        return ((fit, pred), out_stats)
    else:
        return ((fit), out_stats)


def table_stats(out_stats):
    list_stats = ['AIC','BIC','DW','LMpv','MSE','R2']
    list_inter = ['Dum_', 'intercept',]
    # Organizing Statistics
    ord_last = [x for x in out_stats.index if any(y in x for y in list_stats)]
    ord_mid  = [x for x in out_stats.index if any(y in x for y in list_inter)]
    ord = [x for x in out_stats.index if x not in ord_mid + ord_last]
    out_stats = out_stats.loc[ord + ord_mid + ord_last]

    out_stats.index = [x.replace('_ZS', '').replace('_12M', '') for x in out_stats.index]
    idx = [x[0] + '(pval)' if '(pval)' in x else x for x in out_stats.index]
    out_stats.insert(0, 'idx', idx)
    out_stats.reset_index(inplace=True, drop=True)
    return out_stats