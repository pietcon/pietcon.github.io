import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd


ouro_limpo = pd.read_csv("ouro_limpo.csv", parse_dates=["Date"])    
petroleo_wti_limpo = pd.read_csv("petroleo_wti_limpo.csv", parse_dates=["Date"])
petroleo_brent_limpo = pd.read_csv("petroleo_brent_limpo.csv", parse_dates=["Date"])
gas_natural_limpo = pd.read_csv("gas_natural_limpo.csv", parse_dates=["Date"])
milho_limpo = pd.read_csv("milho_limpo.csv", parse_dates=["Date"])
soja_limpo = pd.read_csv("soja_limpo.csv", parse_dates=["Date"])
trigo_limpo = pd.read_csv("trigo_limpo.csv", parse_dates=["Date"])
cobre_limpo = pd.read_csv("cobre_limpo.csv", parse_dates=["Date"])
acucar_limpo = pd.read_csv("acucar_limpo.csv", parse_dates=["Date"])
cafe_limpo = pd.read_csv("cafe_limpo.csv", parse_dates=["Date"])

print(ouro_limpo.head())
print(ouro_limpo.dtypes)

commodities = {
    "Ouro": ouro_limpo,
    "Petróleo WTI": petroleo_wti_limpo,
    "Petróleo Brent": petroleo_brent_limpo,
    "Gás Natural": gas_natural_limpo,
    "Milho": milho_limpo,
    "Soja": soja_limpo,
    "Trigo": trigo_limpo,
    "Cobre": cobre_limpo,
    "Açúcar": acucar_limpo,
    "Café": cafe_limpo
}

# Cria app Dash
app = dash.Dash(__name__)

# Intervalo total
datas = pd.concat([df["Date"] for df in commodities.values()])
data_min, data_max = datas.min(), datas.max()

app.layout = html.Div([
    html.H3("Comparação Normalizada de Commodities"),
    dcc.DatePickerRange(
        id='intervalo-datas',
        min_date_allowed=data_min,
        max_date_allowed=data_max,
        start_date=data_min,
        end_date=data_max
    ),
    dcc.Graph(id='grafico-commodities')
])


@app.callback(
    Output('grafico-commodities', 'figure'),
    Input('intervalo-datas', 'start_date'),
    Input('intervalo-datas', 'end_date')
)
def atualizar_grafico(data_inicio, data_fim):
    fig = go.Figure()

    for nome, df in commodities.items():
        df_filtrado = df[(df["Date"] >= data_inicio) & (df["Date"] <= data_fim)].copy()

        if not df_filtrado.empty:
            preco_inicial = df_filtrado["Close"].iloc[0]
            df_filtrado["Normalizado"] = df_filtrado["Close"] / preco_inicial * 100

            fig.add_trace(go.Scatter(
                x=df_filtrado["Date"],
                y=df_filtrado["Normalizado"],
                mode='lines',
                name=nome
            ))

    fig.update_layout(
        title="Preços Normalizados (Início = 100)",
        xaxis_title="Data",
        yaxis_title="Índice (base 100)",
        template="plotly_white"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
