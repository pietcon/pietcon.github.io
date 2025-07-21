import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Carrega os dados (ajuste os caminhos conforme necessário)
dados = {
    "Ouro": pd.read_csv("ouro_limpo.csv", parse_dates=["Date"]),
    "Petróleo WTI": pd.read_csv("petroleo_wti_limpo.csv", parse_dates=["Date"]),
    "Petróleo Brent": pd.read_csv("petroleo_brent_limpo.csv", parse_dates=["Date"]),
    "Gás Natural": pd.read_csv("gas_natural_limpo.csv", parse_dates=["Date"]),
    "Milho": pd.read_csv("milho_limpo.csv", parse_dates=["Date"]),
    "Soja": pd.read_csv("soja_limpo.csv", parse_dates=["Date"]),
    "Trigo": pd.read_csv("trigo_limpo.csv", parse_dates=["Date"]),
    "Cobre": pd.read_csv("cobre_limpo.csv", parse_dates=["Date"]),
    "Açúcar": pd.read_csv("acucar_limpo.csv", parse_dates=["Date"]),
    "Café": pd.read_csv("cafe_limpo.csv", parse_dates=["Date"]),
}

# Inicializa o app Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='commodities-graph'),
])

@app.callback(
    Output('commodities-graph', 'figure'),
    Input('commodities-graph', 'relayoutData')
)
def update_graph(relayout_data):
    # 1. Captura robusta das datas escolhidas
    start_date = end_date = None
    if relayout_data:
        # range-slider ou zoom manual devolvem chaves ligeiramente diferentes
        if 'xaxis.range[0]' in relayout_data:
            start_date = pd.to_datetime(relayout_data['xaxis.range[0]'])
            end_date   = pd.to_datetime(relayout_data['xaxis.range[1]'])
        elif 'xaxis.range' in relayout_data:
            start_date = pd.to_datetime(relayout_data['xaxis.range'][0])
            end_date   = pd.to_datetime(relayout_data['xaxis.range'][1])

    fig = go.Figure()

    for nome, df in dados.items():
        df_temp = df.copy()

        # 2. Usa só os pontos que realmente estão no intervalo visível
        if start_date is not None and end_date is not None:
            mask = (df_temp['Date'] >= start_date) & (df_temp['Date'] <= end_date)
            df_temp = df_temp.loc[mask]

        # Se, depois do filtro, não restou nada, pula para a próxima série
        if df_temp.empty:
            continue

        # 3. Recalcula o índice com base no 1º valor *visível*
        primeiro_valor = df_temp['Close'].iloc[0]
        df_temp['Indexado'] = df_temp['Close'] / primeiro_valor * 100

        fig.add_trace(go.Scatter(
            x=df_temp['Date'],
            y=df_temp['Indexado'],
            mode='lines',
            name=nome
        ))

    fig.update_layout(
        title="Tendência Relativa de Commodities (Indexa-se ao primeiro ponto visível)",
        xaxis_title="Data",
        yaxis_title="Índice (100 = primeiro valor visível)",
        template="plotly_white",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.3),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1a", step="year", stepmode="backward"),
                    dict(step="all", label="Tudo")
                ]
            ),
            type="date"
        )
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)