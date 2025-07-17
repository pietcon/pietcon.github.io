import pandas as pd
import math

# 1) lê o Excel com os dados já formatados
df = pd.read_excel("regressoes/bla_formatado.xlsx")

# 2) define suas cores-base em hex
# para 3_color_scale genérico (min → mid → max)
COLORS = {
    "mean": ("#F8696B", "#FFEB84", "#63BE7B"),
    "std":  ("#F8696B", "#FFEB84", "#63BE7B"),
    "R2":   ("#F8696B", "#FFEB84", "#63BE7B"),
    "coef": ("#F8696B", "#FFEB84", "#63BE7B"),
}

# converte "#RRGGBB" → (r,g,b)
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

# interpola entre duas cores
def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

# mapeia valor v ao respectivo hex
def map_3color(v, mn, mx, min_col, mid_col, max_col):
    if mx==mn: t = 0.5
    else: t = (v-mn)/(mx-mn)
    if t<=0.5:
        c = lerp_color(min_col, mid_col, t*2)
    else:
        c = lerp_color(mid_col, max_col, (t-0.5)*2)
    return '#{:02x}{:02x}{:02x}'.format(*c)

# 3) prepara estatísticas de cada escala
scales = {}
for col in ["mean","std","R2","coef"]:
    vals = df[col].astype(float)
    mn, mx = vals.min(), vals.max()
    min_col, mid_col, max_col = map(hex2rgb, COLORS[col])
    scales[col] = (mn, mx, min_col, mid_col, max_col)

html = ['<table class="table-cond">']
html.append('<tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr>')

for _, row in df.iterrows():
    html.append('<tr>')
    for col in df.columns:
        val = row[col]
        style = ""
        # prepara o texto que vai para o <td>
        cell_text = "" if pd.isna(val) else val

        # só aplica 3-color-scale se for número válido
        if col in scales and pd.notna(val):
            mn, mx, mc1, mc2, mc3 = scales[col]
            style = f"background:{map_3color(float(val), mn, mx, mc1, mc2, mc3)}"
        # só colore p-valores se for número válido
        elif col.startswith("pval") and pd.notna(val):
            style = "color:red" if float(val) < 0.05 else ""

        html.append(f'<td style="{style}">{cell_text}</td>')
    html.append('</tr>')
html.append('</table>')

with open("assets/table/CDX_US_HY.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html))
    
print("✅ Incluído em assets/table/CDX_US_HY.html")
