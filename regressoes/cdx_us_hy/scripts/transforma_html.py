import pandas as pd

df = pd.read_excel("regressoes/cdx_us_hy/excel/out/CDX_US_HY_spread_simple_analysis_legend_formatado.xlsx")

# ================================
# Definição das cores-base para escala de 3 tons
# - Para alterar a paleta, substitua os valores hex abaixo
#   por outras cores de sua preferência (min → mid → max)
# ================================
COLORS = {
    "mean": ("#F8696B", "#FFEB84", "#63BE7B"),
    "std":  ("#F8696B", "#FFEB84", "#63BE7B"),
    "R2":   ("#F8696B", "#FFEB84", "#63BE7B"),
    "coef": ("#F8696B", "#FFEB84", "#63BE7B"),
}

# ================================
# Funções auxiliares de conversão e interpolação de cores
# - Geralmente não precisam de ajustes
# ================================

def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

def map_3color(v, mn, mx, min_col, mid_col, max_col):
    if mx==mn:
        t = 0.5
    else:
        t = (v-mn)/(mx-mn)
    if t<=0.5:
        c = lerp_color(min_col, mid_col, t*2)
    else:
        c = lerp_color(mid_col, max_col, (t-0.5)*2)
    return '#{:02x}{:02x}{:02x}'.format(*c)

# ================================
# Cálculo de valores mínimos e máximos para cada coluna
# - Necessário para mapear corretamente as cores
# ================================
scales = {}
for col in ["mean","std","R2","coef"]:
    vals = df[col].astype(float)
    mn, mx = vals.min(), vals.max()
    min_col, mid_col, max_col = map(hex2rgb, COLORS[col])
    scales[col] = (mn, mx, min_col, mid_col, max_col)

# ================================
# Montagem do HTML com CSS e JavaScript
# - Aqui você pode alterar estilos globais da tabela:
#   * font-family, font-size, cores de fundo e bordas
#   * espaçamento (padding/margin) e largura máxima
# - Para trocar a fonte, ajuste a <link> do Google Fonts e a propriedade font-family abaixo
# - Para ajustar responsividade, mexa em width, max-width e media queries
# ================================
html = []
html.append('<!DOCTYPE html>')
html.append('<html lang="pt-BR">')
html.append('<head>')
html.append('  <meta charset="UTF-8">')
html.append('  <title>CDX_US_HY Table</title>')
# Importa fontes do Google Fonts — substitua pelo URL da família desejada
html.append('  <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,400,600&display=swap" rel="stylesheet">')
html.append('  <style>')
# Classe .table-cond: ajuste font-family, font-size, cores de fundo e espaçamento
html.append('    .table-cond {')
html.append('      width:90%;')           # ocupa 90% da largura da viewport
html.append('      max-width:1400px;')    # limite máximo de largura
html.append('      margin:30px auto;')     # centraliza horizontalmente
html.append('      border-collapse:collapse;')
html.append('      font-family:Open Sans,sans-serif;')  # altere para "Arial", "Roboto", etc.
html.append('      font-size:0.95rem;')    # ajuste o tamanho do texto
html.append('      text-align:center;')
html.append('      background:#fff;')      # cor de fundo da tabela
html.append('    }')
# Estilo de células: bordas e espaçamento interno
html.append('    .table-cond th, .table-cond td {')
html.append('      border:1px solid rgba(0,0,0,0.1);')  # cor e espessura da borda
html.append('      padding:8px 12px;')                  # ajuste padding vertical e horizontal
html.append('      white-space:nowrap;')                 # impede quebra de linha
html.append('    }')
# Cabeçalho: fundo levemente diferenciado e peso da fonte
html.append('    .table-cond thead th {')
html.append('      background:rgba(0,0,0,0.04);')
html.append('      font-weight:600;')
html.append('    }')
# Linhas pares: fundo alternado
html.append('    .table-cond tbody tr:nth-child(even) {')
html.append('      background:rgba(0,0,0,0.02);')
html.append('    }')
# Hover: destaque ao passar o mouse
html.append('    .table-cond tbody tr:hover {')
html.append('      background:rgba(0,0,0,0.05);')
html.append('    }')
# Controles de colunas: posicione e estilize inputs de checkbox aqui
html.append('    .col-toggle {')
html.append('      margin:20px auto;')
html.append('      text-align:center;')
html.append('    }')
html.append('    .col-toggle label {')
html.append('      margin-right:10px;')
html.append('      font-size:0.9rem;')  # ajuste o tamanho do rótulo
html.append('    }')
html.append('  </style>')
html.append('  <script>')
html.append('    // Função para esconder/exibir colunas pelo índice (0-based)')
html.append('    function toggleColumn(idx) {')
html.append('      const table = document.querySelector(".table-cond");')
html.append('      table.querySelectorAll("tr").forEach(tr => {')
html.append('        const cell = tr.children[idx];')
html.append('        if (!cell) return;')
html.append('        cell.style.display = cell.style.display === "none" ? "table-cell" : "none";')
html.append('      });')
html.append('    }')
html.append('  </script>')
html.append('</head>')
html.append('<body>')

# Bloco de checkboxes para controle de visibilidade das colunas
html.append('  <div class="col-toggle">')
for i, col in enumerate(df.columns):
    html.append(f'    <label><input type="checkbox" checked onchange="toggleColumn({i})"> {col}</label>')
html.append('  </div>')


# Construção da tabela HTML
html.append('  <table class="table-cond">')
html.append('    <thead><tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr></thead>')
html.append('    <tbody>')
for _, row in df.iterrows():
    html.append('      <tr>')
    for col in df.columns:
        val = row[col]
        cell = "" if pd.isna(val) else val
        style = ""
        if col in scales and pd.notna(val):
            mn, mx, c1, c2, c3 = scales[col]
            style = f'background:{map_3color(float(val), mn, mx, c1, c2, c3)};'
        elif col.startswith("pval") and pd.notna(val) and float(val) < 0.05:
            style = 'color:red;'
        html.append(f'        <td style="{style}">{cell}</td>')
    html.append('      </tr>')
html.append('    </tbody>')
html.append('  </table>')
html.append('</body>')
html.append('</html>')

# Salva em arquivo
with open("regressoes/cdx_us_hy/html/CDX_US_HY.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print("✅ HTML com colunas ocultáveis gerado em regressoes/cdx_us_hy/html/CDX_US_HY.html")
