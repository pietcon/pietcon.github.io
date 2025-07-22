import os
import pandas as pd

# Caminhos base
path_in = os.path.join('regressoes', 'cdx_us_hy', 'excel', 'in')
path_out = os.path.join('regressoes', 'cdx_us_hy', 'excel', 'out', 'US', 'Credit', 'HY')
os.makedirs(path_out, exist_ok=True)

# Arquivos de entrada
file_in_m = os.path.join(path_in, 'out_DB_M.xlsx')
file_in_d = os.path.join(path_in, 'out_DB_d.xlsx')
file_cad = os.path.join(path_in, 'legendas.xlsx')

# Leitura dos dados
df_mensal = pd.read_excel(file_in_m, index_col=0)
df_diario = pd.read_excel(file_in_d, index_col=0)
df_codigos = pd.read_excel(file_cad)

# Gerar arquivo cockpit com as 3 abas
file_cockpit = os.path.join(path_out, 'CDX_US_HY_spread_cockpit.xlsx')
with pd.ExcelWriter(file_cockpit) as writer:
    df_mensal.to_excel(writer, sheet_name='IN_M')
    df_diario.to_excel(writer, sheet_name='IN_D')
    df_codigos.to_excel(writer, sheet_name='CAD', index=False)

print(f'Arquivo gerado com sucesso: {file_cockpit}')
