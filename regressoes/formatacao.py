import pandas as pd
from xlsxwriter.utility import xl_col_to_name

bla = pd.read_excel("regressoes/bla.xlsx")

cols_to_fmt = ["mean", "std", "pval_ADR", "pval_Coint", "R2", "coef", "pval"]

with pd.ExcelWriter("regressoes/bla_formatado.xlsx", engine="xlsxwriter") as writer:
    bla.to_excel(writer, index=False, sheet_name="Dados")
    wb  = writer.book
    ws  = writer.sheets["Dados"]

    # formato vermelho para p-values baixos
    fmt_red = wb.add_format({"font_color": "red"})

    last_row = len(bla) + 1   # +1 porque tem cabeçalho na linha 1

    for col in cols_to_fmt:
        idx    = bla.columns.get_loc(col)
        letter = xl_col_to_name(idx)
        cell_range = f"{letter}2:{letter}{last_row}"

        if col in ["mean", "std", "R2"]:
            ws.conditional_format(cell_range, {"type":"3_color_scale"})
        elif col == "coef":
            ws.conditional_format(cell_range, {
                "type":     "3_color_scale",
                "min_type": "percentile", "min_value": 0,   "min_color": "#F8696B",
                "mid_type": "percentile", "mid_value": 50,  "mid_color": "#FFEB84",
                "max_type": "percentile", "max_value": 100, "max_color": "#63BE7B",
            })
        else:
            # todos os p-values < 0.05 em vermelho
            ws.conditional_format(cell_range, {
                "type":     "cell",
                "criteria": "<",
                "value":    0.05,
                "format":   fmt_red
            })

print("Gerado: regressoes/CDX_US_HY_spread_simple_analysis_legend_formatado.xlsx")