import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

# Diretórios
os.makedirs("files/data", exist_ok=True)
os.makedirs("assets/charts", exist_ok=True)

# Commodities e tickers do Yahoo Finance
commodities = {
    "Ouro": "GC=F",
    "Petróleo WTI": "CL=F",
    "Petróleo Brent": "BZ=F",
    "Gás Natural": "NG=F",
    "Milho": "ZC=F",
    "Soja": "ZS=F",
    "Trigo": "ZW=F",
    "Cobre": "HG=F",
    "Açúcar": "SB=F",
    "Café": "KC=F"
}

# Data de início
start_date = "2020-01-01"

# Gráfico
fig = go.Figure()

for nome, ticker in commodities.items():
    print(f"Processando: {nome}")

    # Nome do arquivo CSV
    nome_arquivo = nome.lower().replace(" ", "_").replace("ç", "c").replace("á", "a").replace("é", "e")
    caminho_csv = f"files/data/{nome_arquivo}.csv"

    # Verifica se já existe
    if os.path.exists(caminho_csv):
        df_antigo = pd.read_csv(caminho_csv, parse_dates=["data"])
        ultima_data = df_antigo["data"].max() + pd.Timedelta(days=1)
        start_date_local = ultima_data.strftime("%Y-%m-%d")
        print(f"  ↳ Atualizando desde {start_date_local}")
    else:
        df_antigo = pd.DataFrame()
        start_date_local = start_date
        print("  ↳ Criando histórico do zero.")

    # Baixa novos dados
    df_novo = yf.download(ticker, start=start_date_local, interval="1d", progress=False)
    if df_novo.empty:
        print("  ↳ Nenhum dado novo.")
        df_total = df_antigo
    else:
        df_novo = df_novo.reset_index()[["Date", "Close"]].rename(columns={"Date": "data", "Close": "valor"})
        df_novo["data"] = pd.to_datetime(df_novo["data"])
        # Junta antigo + novo
        df_total = pd.concat([df_antigo, df_novo], ignore_index=True)
        df_total = df_total.drop_duplicates(subset="data").sort_values("data")

    # Salva CSV atualizado
    df_total.to_csv(caminho_csv, index=False)
    print(f"  ↳ CSV atualizado: {caminho_csv}")

    # Adiciona ao gráfico
    fig.add_trace(go.Scatter(
        x=df_total["data"],
        y=df_total["valor"],
        mode="lines",
        name=nome,
        hovertemplate="<b>%{text}</b><br>Data: %{x|%d/%m/%Y}<br>Preço: %{y:.2f}<extra></extra>",
        text=[nome] * len(df_total)
    ))

# Layout do gráfico
fig.update_layout(
    title="Commodities – Histórico de Preços",
    xaxis_title="Data",
    yaxis_title="Preço (USD ou centavos)",
    template="plotly_white",
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    ),
    margin=dict(l=50, r=20, t=80, b=40),
)

# Exporta gráfico HTML
chart_path = "assets/charts/commodities.html"
pio.write_html(fig, file=chart_path, full_html=True, auto_open=False, include_plotlyjs="cdn")
print(f"Gráfico exportado para {chart_path}")
