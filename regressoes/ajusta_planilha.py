import pandas as pd

df = pd.read_excel("regressoes/CDX_US_HY_spread_simple_analysis.xlsx")
leg = pd.read_excel("regressoes/legendas.xlsx")

df.columns.values[0] = "Codes"
df.insert(1, "Names", "")

mapping = dict(zip(leg["Codes"], leg["Names"]))

print(mapping)

for idx, code in df["Codes"].items():
    if code in mapping:
        df.at[idx, "Names"] = mapping[code]

df.to_excel("regressoes/CDX_US_HY_spread_simple_analysis_legend.xlsx", index=False)
