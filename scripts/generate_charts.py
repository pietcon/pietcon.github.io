import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

# === Configuração de diretórios ===
csv_path = "files/data/brl_usd.csv"
chart_path = "assets/charts/brl_usd.html"
os.makedirs("files/data", exist_ok=True)
os.makedirs("assets/charts", exist_ok=True)

# === Parâmetros ===
ticker = "BRL=X"
start_date = "2020-01-01"

# === Carrega histórico, se existir ===
if os.path.exists(csv_path):
    df_antigo = pd.read_csv(csv_path, parse_dates=["data"])
    ultima_data = df_antigo["data"].max() + pd.Timedelta(days=1)
    start_date = ultima_data.strftime("%Y-%m-%d")
    print(f"Atualizando a partir de {start_date}")
else:
    df_antigo = pd.DataFrame()
    print("Criando histórico do zero.")

# === Baixa dados novos com yfinance ===
df_novo = yf.download(ticker, start=start_date, interval="1d", progress=False)

# Remove MultiIndex se houver
if isinstance(df_novo.columns, pd.MultiIndex):
    df_novo.columns = df_novo.columns.get_level_values(0)

# Organiza DataFrame
df_novo = df_novo.reset_index()
df_novo = df_novo[["Date", "Close"]].rename(columns={"Date": "data", "Close": "valor"})
df_novo["data"] = pd.to_datetime(df_novo["data"])

# === Junta com dados antigos ===
df_total = pd.concat([df_antigo, df_novo], ignore_index=True)
df_total = df_total.drop_duplicates(subset="data").sort_values("data")

# === Salva CSV atualizado ===
df_total.to_csv(csv_path, index=False)
print(f"CSV salvo em {csv_path} com {len(df_total)} linhas.")

# === Criar gráfico interativo ===
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_total['data'].values,
    y=df_total['valor'].values,
    mode='lines+markers',
    name='BRL/USD',
    line=dict(color='green'),
    marker=dict(size=6),
    hovertemplate='<b>Data:</b> %{x|%d/%m/%Y}<br><b>Valor:</b> US$ %{y:.4f}<extra></extra>',
))

fig.update_layout(
    title='Taxa de Câmbio - BRL/USD',
    xaxis_title='Data',
    yaxis_title='US$ por R$',
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1 sem", step="day", stepmode="backward"),
                dict(count=1, label="1 mês", step="month", stepmode="backward"),
                dict(count=3, label="3 meses", step="month", stepmode="backward"),
                dict(step="all", label="Tudo")
            ]
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(fixedrange=False),
    margin=dict(l=50, r=20, t=80, b=40),
    autosize=True,
)

# === Exporta HTML interativo ===
pio.write_html(
    fig,
    file=chart_path,
    full_html=True,
    auto_open=False,
    include_plotlyjs='cdn'
)

print(f"Gráfico exportado para {chart_path}")
