import pandas as pd
from xlsxwriter.utility import xl_col_to_name

df = pd.read_excel("regressoes/cdx_us_hy/excel/out/CDX_US_HY_spread_simple_analysis_legend.xlsx")

# Colunas que receberão formatação condicional
cols_to_fmt = ["mean", "std", "pval_ADR", "pval_Coint", "R2", "coef", "pval"]

with pd.ExcelWriter("regressoes/cdx_us_hy/excel/out/CDX_US_HY_spread_simple_analysis_legend_formatado.xlsx", engine="xlsxwriter") as writer:
    
    # Escreve todo o DataFrame na planilha “Dados” (sem o índice pandas)
    df.to_excel(writer, index=False, sheet_name="Dados")
    
    # Referências ao workbook e worksheet para aplicar formatações
    wb  = writer.book
    ws  = writer.sheets["Dados"]

    # Defina aqui qualquer formato customizado para reutilizar depois
    #   - Exemplo: fonte vermelha para destacar p-values abaixo de 0.05
    
    fmt_red = wb.add_format({"font_color": "red"})

    #   - Você pode criar formatos adicionais, e.g.:
    # fmt_bg_yellow = wb.add_format({"bg_color": "#FFFF99"})
    # fmt_num_2dec = wb.add_format({"num_format": "0.00"})
    
    last_row = len(df) + 1   # +1 porque tem cabeçalho na linha 1

    # Loop pelas colunas que vão receber formatação condicional
    for col in cols_to_fmt:
        # Determina índice numérico da coluna no DataFrame
        idx = df.columns.get_loc(col)
        # Converte índice em letra de coluna tipo “A”, “B”, …
        letter = xl_col_to_name(idx)
        # Intervalo de células: da linha 2 até a última com dados
        cell_range = f"{letter}2:{letter}{last_row}"

        # Diversos estilos de formatação conforme o tipo de dado
        if col in ["mean", "std", "R2"]:
            # Escala de 3 cores padrão (menor → médio → maior)
            ws.conditional_format(
                cell_range,
                {"type": "3_color_scale"}
            )
            # → Para mudar as cores, substitua por:
            #   {"type":"3_color_scale",
            #    "min_color":"#DDD", "mid_color":"#AAA", "max_color":"#333"}
            
        elif col == "coef":
            # Escala de 3 cores personalizada
            ws.conditional_format(cell_range, {
                "type":       "3_color_scale",
                "min_type":   "percentile", "min_value": 0,   "min_color": "#F8696B",
                "mid_type":   "percentile", "mid_value": 50,  "mid_color": "#FFEB84",
                "max_type":   "percentile", "max_value": 100, "max_color": "#63BE7B",
            })
            # → Para ajustar:
            #   - mude min_color/mid_color/max_color para outra paleta
            #   - use "num" em vez de "percentile" se quiser valores absolutos
            
        else:
            # Formatação de p-values: cor de fonte vermelha se < 0.05
            ws.conditional_format(cell_range, {
                "type":     "cell",
                "criteria": "<",
                "value":    0.05,
                "format":   fmt_red
            })
            # → Para trocar o critério:
            #   - criteria: ">", "<=", "between", etc.
            #   - value pode ser um número ou referência a outra célula (e.g. "$A$1")
            # → Para usar cor de fundo em vez de fonte:
            #   - crie fmt_bg = wb.add_format({"bg_color":"#FFCCCC"})
            #   - use "format": fmt_bg

print("Gerado: regressoes/cdx_us_hy/excel/out/CDX_US_HY_spread_simple_analysis_legend_formatado.xlsx")

    # (Opcional) Exemplo de ajuste de largura de coluna
    # for i, width in enumerate([12, 10, 8, 15, 8, 8, 8, 8]):  # personalize as larguras
    #     ws.set_column(i, i, width)

    # (Opcional) Adicionar um cabeçalho fixo ou rodapé
    # ws.set_header('&C&"Arial,Bold"Análise de Regressores')
    # ws.set_footer('&L&D &T &R Página &P de &N')