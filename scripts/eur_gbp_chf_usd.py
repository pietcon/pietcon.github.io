import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

# === Configuração de diretórios ===
os.makedirs("files/data", exist_ok=True)
os.makedirs("assets/charts", exist_ok=True)

# === Tickers e aliases ===
pares = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "CHF/USD": "CHFUSD=X"
}

start_date_base = "2020-01-01"
dfs = {}  # dicionário com DataFrames por par

# === Loop por cada par ===
for nome, ticker in pares.items():
    csv_path = f"files/data/{nome.lower().replace('/', '_')}.csv"

    # Verifica se já existe histórico
    if os.path.exists(csv_path):
        df_antigo = pd.read_csv(csv_path, parse_dates=["data"])
        ultima_data = df_antigo["data"].max() + pd.Timedelta(days=1)
        start_date = ultima_data.strftime("%Y-%m-%d")
        print(f"[{nome}] Atualizando desde {start_date}")
    else:
        df_antigo = pd.DataFrame()
        start_date = start_date_base
        print(f"[{nome}] Criando histórico do zero.")

    # Baixa novos dados
    df_novo = yf.download(ticker, start=start_date, interval="1d", progress=False)

    if isinstance(df_novo.columns, pd.MultiIndex):
        df_novo.columns = df_novo.columns.get_level_values(0)

    df_novo = df_novo.reset_index()
    df_novo = df_novo[["Date", "Close"]].rename(columns={"Date": "data", "Close": "valor"})
    df_novo["data"] = pd.to_datetime(df_novo["data"])

    # Junta dados antigos e novos
    df_total = pd.concat([df_antigo, df_novo], ignore_index=True)
    df_total = df_total.drop_duplicates(subset="data").sort_values("data")

    # Salva CSV
    df_total.to_csv(csv_path, index=False)
    print(f"[{nome}] CSV salvo com {len(df_total)} linhas.")

    # Armazena para o gráfico
    dfs[nome] = df_total

# === Criação do gráfico com 3 linhas ===
fig = go.Figure()

for nome, df in dfs.items():
    fig.add_trace(go.Scatter(
        x=df["data"],
        y=df["valor"],
        mode="lines+markers",
        name=nome,
        hovertemplate='<b>Data:</b> %{x|%d/%m/%Y}<br><b>Valor:</b> US$ %{y:.4f}<extra></extra>',
    ))

fig.update_layout(
    title="Cotações - EUR/USD, GBP/USD e CHF/USD",
    xaxis_title="Data",
    yaxis_title="US$ por 1 unidade da moeda base",
    template="plotly_white",
    hovermode="x unified",
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
    margin=dict(l=50, r=20, t=80, b=40),
    autosize=True,
)

# === Exporta o HTML do gráfico ===
chart_path = "assets/charts/forex_trio.html"
pio.write_html(
    fig,
    file=chart_path,
    full_html=True,
    auto_open=False,
    include_plotlyjs="cdn"
)

print(f"Gráfico exportado para {chart_path}")
