---
title: "PCA da Matriz de Câmbio: Correlação entre Variações Cambiais"
#permalink: /research/pca_fx_corr/
author_profile: true
layout: single
excerpt: 'Estudo empírico da estrutura de correlação entre moedas via PCA aplicada à matriz de retornos cambiais.'
date: 2025-07-23
---

Este trabalho examina a aplicação de **Principal Component Analysis (PCA)** à matriz de variação de taxas de câmbio, com o objetivo de decompor e interpretar os padrões de correlação entre diferentes moedas. O uso de PCA em séries financeiras é consolidado na literatura: por exemplo, em aplicações sobre commodities e mercados FX .

### 1. Fundamentação teórica  
PCA é uma técnica de redução de dimensionalidade baseada na análise dos autovalores e autovetores da matriz de covariância ou correlação . No contexto cambial, a utilização da matriz de correlação é preferível, pois elimina a influência das escalas das moedas e permite uma análise comparável das variações .

### 2. Metodologia  
1. Coleta diária de taxas de câmbio de um conjunto de *p* moedas.  
2. Cálculo dos retornos logarítmicos e centralização da série.  
3. Construção da matriz de correlação R (p × p).  
4. Decomposição em autovalores/eigenvetoes para extrair os componentes principais.  
5. Seleção dos primeiros PCs que explicam entre 70% e 90% da variância total, conforme metodologia recomendada .

### 3. Interpretação dos resultados  
- **Primeiro componente principal (PC1)**: geralmente representa um “modo de mercado” global, com cargas (loadings) positivas em todas as moedas — indicando movimentos covariantes conforme descrito em estudos do FMI .  
- **Componentes subsequentes**: destacam clusters regionais ou características específicas — por exemplo, podem revelar grupos de moedas emergentes com comportamento correlacionado, semelhante ao que foi identificado em estudos de mercado de petróleo .

### 4. Aplicações práticas  
- Reconstrução de índices sintéticos de exposição cambial via projeções em PCs.  
- Identificação de clusters de moedas para estratégias de hedging ou arbitragem .  
- Simplificação de portfólios cambiais sem perda significativa de informação — estratégia que segue os mesmos princípios utilizados em portfólios de risco financeiro .

### 5. Conclusão  
A aplicação de PCA à matriz de variações cambiais oferece uma estrutura robusta para analisar correlações, reduzir dimensionalidade e construir indicadores de risco/hedge. A abordagem é extensível e pode ser complementada com variantes como PCA hierárquico, PCA robusto ou técnicas de Machine Learning (ex: autoencoders) para explorar estruturas mais complexas .
