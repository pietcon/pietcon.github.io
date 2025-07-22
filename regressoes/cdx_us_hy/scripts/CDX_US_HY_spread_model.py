import os, sys, warnings, re
import pandas as pd
from datetime import datetime as dt
# roots = ['T:\\ECONOMIA\\Estrategia\\']
# for p_root in roots:
#     if os.path.exists(p_root):
#         break
# p_codes = p_root + 'codes\\'
# sys.path.append(p_codes + 'generic\\')
from p_functions import reg_box_jenkins, table_stats
from sklearn.linear_model import LinearRegression

####MUDANCA

path_scripts = os.path.join('regressoes', 'cdx_us_hy', 'scripts')
sys.path.append(path_scripts)

path_excel_in = os.path.join('regressoes', 'cdx_us_hy', 'excel', 'in')
path_excel_out = os.path.join('regressoes', 'cdx_us_hy', 'excel', 'out')
path_scripts = os.path.join('regressoes', 'cdx_us_hy', 'scripts')

sys.path.append(path_scripts)

# Arquivos de entrada
file_data = os.path.join(path_excel_in, 'out_DB_M.xlsx')
file_specs = os.path.join(path_excel_in, 'specs_fixed.xlsx')
####MUDANCA


# Filter or ignore the warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Set options
pd.set_option('mode.chained_assignment', None)

start_load = dt.now()

##############################
##############################
#### ESPECIFICACOES GERAIS
folder = os.path.join('Credit', 'HY')
region = 'US'
path_out = os.path.join(path_excel_out, region, folder)

index = 'CDX_US_HY_spread'
last_datapoint = '2023-11-01'
plan_version = index + '_cockpit'

#### PARÂMETROS
main_setup = True
version_trunch = 1; force_regs = [0,1,2,3,4,5,6,7,8,9]
full_run_OoS = True
daily_model = False

#### OUTROS
vint_robust_test = 3
reccursive_reg = True; plot_Xs = True
d_intercept = True; adjust_first_datapoint = False
#################################
#################################
#################################

#path_out = p_root + 'Modelos\\' + region + folder

if daily_model:
    first_datapoint = '2007-01-01'
    reccursive_length = 3000; step_oos = 30
    sheet = 'IN_D'
else:
    first_datapoint = '2001-11-01'
    reccursive_length = 160; step_oos = 1
    sheet = 'IN_M'

###########
# Data
# IN: DF horizontal, Vars Y e X empilhadas

file_plan = os.path.join(path_out, f'{plan_version}.xlsx')

df = pd.read_excel(file_plan, sheet_name=sheet, index_col=0)

df_daily = pd.read_excel(file_plan, sheet_name='IN_D', index_col=0)
dff_cods = pd.read_excel(file_plan, sheet_name='CAD', usecols=['Codes', 'Names']).dropna()

keep_cols = [x for x in df.columns if x != 0]
df = df[keep_cols]
df_daily = df_daily[keep_cols]
###########

###########
# Uploading names
# IN: DF horizontal, cols: [cod, name_paste]
dff_cods.columns = ['cod','name_paste']
###########

###########
# Adjusting paths
if not main_setup: 
    path_out += 'v_' + str(version_trunch) + '\\'
    reg_versions = [int(re.sub('t_', '', x)) for x in os.listdir(path_out) if x[:2] == 't_']
    if force_regs != []: reg_versions = force_regs
else:
    reg_versions = [0]
    version_trunch = 0
###########

###########
# Defining explanatory variables
vars_X = [x for x in df.columns if x != index]
vars_y = [x for x in df.columns if x not in vars_X]

load_time = dt.now() - start_load
mins = int(load_time.total_seconds() // 60)
secs = int(load_time.total_seconds() % 60)
delta_time = f'{mins} min {secs} sec'
print('Load Time: ', delta_time)

#########################
#########################
#########################
### RUN VARIABLE SELECTION
for rv in reg_versions:
    path_vint = os.path.join(path_out, f't_{rv}')
    path_recc = os.path.join(path_vint, 'reccursive')
    path_coefs = os.path.join(path_recc, 'coefs')
    os.makedirs(path_coefs, exist_ok=True)

    out_recc_proj_is = pd.DataFrame()
    out_recc_proj_oos = pd.DataFrame()
    recc_proj_coef_oos_stats = pd.DataFrame()

    ###########
    # Uploading fix specifications
    # ! to check what changed from the last time we ran the model, 
    # ! just copy the last fixed_specs into the new folder 
    # ! and run the model again with this specification
    
    file_specs = os.path.join(path_vint, 'specs_fixed.xlsx')
    print("🔍 Procurando arquivo em:", file_specs)

    if os.path.exists(file_specs):
        df_specs = pd.read_excel(file_specs, sheet_name='Sheet1', index_col=0)
    else:
        print('No fixed specification defined yet. Run Dynamic Selection first.')
        raise SystemExit
    ###########

    out = {}; out_stats = {}; out_y = {}; fixed_spec = {}
    out_stats_vint = {}; out_robust_k_targ = {}

    # Defining how many times wil will repeat the dependent variable to run multiple specifications
    sectors = [index]
    for i in range(1, df_specs.shape[1]): sectors += [index + str(i)] 

    ##################
    ##################
    if adjust_first_datapoint:   first_datapoint = years[rv] + '-01-01'
    ##################
    ##################

    for sec in sectors:
        # Create folder for each sector
        path_sec = os.path.join(path_vint, sec)
        os.makedirs(path_sec, exist_ok=True)

        # Explained variable
        df_x_test = df[vars_X].copy()
        var_y = sec; var_y_fix = index
        y = df[[var_y_fix]]

        # Dropping rows with NaNs
        dff = pd.concat([y, df_x_test], axis=1)
        dff = dff[(dff.index >= first_datapoint)]
        if not daily_model: dff = dff[(dff.index <= last_datapoint)]

        # Aligning both dfs
        y = dff.iloc[:,0].dropna() # dropping NaN in the end to align X and X_proj
        X = dff.iloc[:,1:]
        X_proj = X.loc[X.index > y.index[-1]]
        X = X.loc[X.index <= y.index[-1]]

        # Dropping variables with NaN in the middle
        X_proj = X_proj[X.columns] # align both dataframes
        spec_used = df_specs[sec].dropna().tolist()
        missing_vars = [col for col in spec_used if col not in X.columns]
        if missing_vars:
            print("🚨 Variáveis da especificação não encontradas nos dados:")
            print(missing_vars)
            #print([col for col in df.columns if 'sloos' in col.lower()])

            raise SystemExit

        # ! check if variables names have changed, if so, rerun dynamic selection will be needed
        # Running official final regression with selected variables
        X_adj = X[spec_used].copy()
        X_adj = X_adj.dropna()
        y_adj = y[y.index >= X_adj.index[0]]
        X_est = X_adj[spec_used].copy(); X_proj_est = X_proj[spec_used]
        dff_reg = pd.concat([y_adj, pd.concat([X_est, X_proj_est])], axis=1)

        ##### START TIMER
        start = dt.now()
        ##### RUNNING REPORT
        out_start_time = start.strftime('%d%b%Y - %H:%M:%S')
        out_first_datapoint = dt.strptime(first_datapoint, '%Y-%m-%d').strftime('%d%b%Y')
        out_last_datapoint = dt.strptime(last_datapoint, '%Y-%m-%d').strftime('%d%b%Y')       
        print('')
        print('START   Y Var: ', sec, '- vers: ', str(version_trunch), '- regs: ', str(rv))
        print('# Start:  ', out_start_time, '    ||   Sample: ', out_first_datapoint, '-', out_last_datapoint)
        print('# N Obs:', y.shape[0], '- Fit:', X.shape[0], '- Pred:', X_proj.shape[0], '  ||   X Var Init:', X.shape[1], '- Spec: ', len(spec_used), '      ...running...') 

        for v in range(vint_robust_test):
            dff_reg_run = dff_reg.copy()
            if v == 0:           
                reg = reg_box_jenkins(dff_reg_run, name=sec, path=path_vint, intercept=d_intercept, 
                                        plot_Xs=plot_Xs, rec_bol=reccursive_reg)
                # Saving the contemporaneous regression results ONLY
                out_stats[var_y] = reg[1]
                out[var_y + '_proj'] = reg[0][1]
                out_y[var_y] = dff_reg[var_y_fix]
                out_stats_vint['t_' + str(v)] = reg[1]

                # Out of Sample reccursion
                if full_run_OoS or main_setup:
                    recc_proj = pd.DataFrame()
                    recc_proj_coef = pd.DataFrame()

                    for t in range(1, reccursive_length, step_oos):
                        df_reg = dff_reg_run.iloc[:-t]
                        df_proj = dff_reg_run.iloc[-t:]
                        reg_recc = LinearRegression(fit_intercept=d_intercept).fit(df_reg.iloc[:,1:], 
                                                                                df_reg.iloc[:,:1])
                        pred = reg_recc.predict(df_proj.iloc[:,1:])
                        data = pd.DataFrame([pred[0][0]], index=[df_proj.index[0]], columns=['mod' + str(rv)])
                        recc_proj = pd.concat([recc_proj, data])
                        coefs = reg_recc.coef_[0]
                        coefs[-1] = reg_recc.intercept_[0]
                        coefs = pd.DataFrame([reg_recc.coef_[0]], index=[df_proj.index[0]], 
                                                                columns=reg_recc.feature_names_in_)
                        recc_proj_coef = pd.concat([recc_proj_coef, coefs], axis=0)

                    recc_proj_coef.sort_index(ascending=True, inplace=True)
                    recc_proj_coef.to_excel(os.path.join(path_coefs, f'out_recc_coefs_{re.sub(index, "", sec)}.xlsx'))


                    # Saving OoS projections
                    first_date = recc_proj_coef.index[0]
                    cols_proj = [col for col in recc_proj_coef.columns if col in df_daily.columns]
                    df_daily_proj = df_daily[cols_proj]

                    #df_daily_proj = df_daily[recc_proj_coef.columns]
                    df_daily_proj = df_daily_proj[(df_daily_proj.index >= first_date)]
                    expand_index = pd.DataFrame(index = df_daily_proj.index)
                    recc_proj_coef_proj = recc_proj_coef.merge(expand_index, how='outer', left_index=True, right_index=True)
                    recc_proj_coef_proj.fillna(method='ffill', inplace=True)
                    out = (recc_proj_coef_proj*df_daily_proj).sum(axis=1)
                    out_recc_proj_oos[sec] = out.loc[~(out==0)]

                    # Counting how many times coefficients have flipped signals in OoS projections
                    adjust = ((-1)*~(recc_proj_coef_proj.iloc[-1]>0))
                    adjust[adjust == 0] = 1
                    obs = recc_proj_coef_proj.shape[0]
                    out = 1 - ((adjust*recc_proj_coef_proj) > 0).sum()/obs
                    out.name = sec
                    recc_proj_coef_oos_stats = pd.concat([recc_proj_coef_oos_stats, out], axis=1)

            else:
                dff_reg_run.dropna(inplace=True)
                dff_reg_run = dff_reg_run.iloc[:-v]
                reg = reg_box_jenkins(dff_reg_run, name=sec, intercept=d_intercept)
                out_stats_vint['t_' + str(v)] = reg[1]

        pd.DataFrame.from_dict(out_stats_vint).to_excel(os.path.join(path_sec, 'vintages_regs_summary.xlsx'))

        ##### END TIMER
        t_elap = dt.now() - start
        mins = int(t_elap.total_seconds() // 60)
        secs = int(t_elap.total_seconds() % 60)
        delta_time = f'{mins} min {secs} sec'
        ##### RUNNING REPORT
        print('# Run time: ', delta_time)
        print('END     Y Var: ', sec, '- vers: ', str(version_trunch), '- regs: ', str(rv))

        if full_run_OoS or main_setup: 
            out_recc_proj_is = pd.concat([out_recc_proj_is, recc_proj], axis=1)

    # Saving Testing Output
    fixed_spec = pd.DataFrame.from_dict(fixed_spec, orient='index').transpose()
    pd.DataFrame.from_dict(out).to_excel(os.path.join(path_vint, 'sectors_proj.xlsx'))

    pd.DataFrame.from_dict(out_y).to_excel(os.path.join(path_vint, 'sectors_actual.xlsx'))
    X.to_excel(os.path.join(path_vint, 'varsX_actual.xlsx'))
    out_sum = table_stats(pd.DataFrame.from_dict(out_stats), df_cods=dff_cods) 
    out_sum.to_excel(os.path.join(path_vint, 'sectors_regs_summary.xlsx'))

    # Saving Final Output
    if main_setup or full_run_OoS:
        recc_proj_coef_oos_stats.to_excel(os.path.join(path_recc, 'out_main_proj_reccursive_oos_signal_flip.xlsx'))
        out_recc_proj_oos.to_excel(os.path.join(path_recc, 'out_main_proj_reccursive_oos.xlsx'))
        out_recc_proj_is.columns = out_recc_proj_oos.columns
        out_recc_proj_is.to_excel(os.path.join(path_recc, 'out_main_proj_reccursive_is.xlsx'))
        pd.DataFrame.from_dict(out).to_excel(os.path.join(path_recc, 'sectors_proj.xlsx'))
        pd.DataFrame.from_dict(out_y).to_excel(os.path.join(path_recc, 'sectors_actual.xlsx'))
        X.to_excel(os.path.join(path_recc, 'varsX_actual.xlsx'))
        out_sum.to_excel(os.path.join(path_recc, 'sectors_regs_summary.xlsx'))

