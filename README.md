# Análise de Cluster em Dados de Países

Este repositório contém uma análise de cluster em dados de países utilizando diferentes métodos de clustering hierárquico aglomerativo. O objetivo é identificar grupos de países com características semelhantes com base em várias métricas econômicas e sociais.

## Requisitos de Instalação

Para rodar este script, você precisará instalar os seguintes pacotes Python:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly
pip install scipy
pip install scikit-learn
pip install pingouin
```

## Estrutura do Código

### Importando os Pacotes

O código começa importando os pacotes necessários para a análise:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
```

### Importando o Banco de Dados

Os dados são importados de um arquivo CSV:

```python
dados_paises = pd.read_csv('dados_paises.csv')
## Fonte: https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data
```

### Visualizando Informações sobre os Dados e Variáveis

O código gera estatísticas descritivas e uma matriz de correlações para entender melhor os dados:

```python
tab_desc = dados_paises.describe()
paises = dados_paises.drop(columns=['country'])
matriz_corr = pg.rcorr(paises, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
```

### Mapa de Calor da Correlação entre Atributos

Um mapa de calor é gerado para visualizar as correlações entre as variáveis:

```python
corr = paises.corr()
fig = go.Figure()
fig.add_trace(go.Heatmap(x = corr.columns, y = corr.index, z = np.array(corr), text=corr.values, texttemplate='%{text:.2f}', colorscale='viridis'))
fig.update_layout(height = 600, width = 600)
fig.show()
```

### Padronização das Variáveis

As variáveis são padronizadas usando Z-Score:

```python
paises_pad = paises.apply(zscore, ddof=1)
```

### Cluster Hierárquico Aglomerativo

Três métodos de encadeamento (single, average e complete) são aplicados usando a distância euclidiana, e os dendrogramas são gerados:

```python
dist_euclidiana = pdist(paises_pad, metric='euclidean')

# Single Linkage
plt.figure(figsize=(16,8))
dend_sing = sch.linkage(paises_pad, method = 'single', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing)
plt.title('Dendrograma Single Linkage', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

# Average Linkage
plt.figure(figsize=(16,8))
dend_avg = sch.linkage(paises_pad, method = 'average', metric = 'euclidean')
dendrogram_a = sch.dendrogram(dend_avg)
plt.title('Dendrograma Average Linkage', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

# Complete Linkage
plt.figure(figsize=(16,8))
dend_compl = sch.linkage(paises_pad, method = 'complete', metric = 'euclidean')
dendrogram_c = sch.dendrogram(dend_compl, color_threshold = 8)
plt.title('Dendrograma Complete Linkage', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.axhline(y = 8, color = 'red', linestyle = '--')
plt.show()
```

### Atribuição de Clusters

Os clusters são atribuídos aos dados utilizando o método complete linkage com 5 clusters:

```python
cluster_comp = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(paises_pad)
dados_paises['cluster_complete'] = indica_cluster_comp
paises_pad['cluster_complete'] = indica_cluster_comp
dados_paises['cluster_complete'] = dados_paises['cluster_complete'].astype('category')
paises_pad['cluster_complete'] = paises_pad['cluster_complete'].astype('category')
```

### Análise de Variância de um Fator (ANOVA)

Uma análise ANOVA é realizada para verificar se há diferenças significativas entre os clusters para várias variáveis:

```python
pg.anova(dv='child_mort', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='exports', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='imports', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='health', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='income', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='inflation', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='life_expec', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='total_fer', between='cluster_complete', data=paises_pad, detailed=True).T
pg.anova(dv='gdpp', between='cluster_complete', data=paises_pad, detailed=True).T
```

### Gráfico 3D dos Clusters

Os clusters são visualizados em gráficos 3D para diferentes combinações de variáveis:

```python
fig = px.scatter_3d(dados_paises, x='total_fer', y='income', z='life_expec', color='cluster_complete')
fig.show()

fig = px.scatter_3d(dados_paises, x='gdpp', y='income', z='life_expec', color='cluster_complete')
fig.show()
```

### Identificação das Características dos Clusters

Os dados são agrupados por cluster, e estatísticas descritivas são geradas para cada grupo:

```python
analise_paises = dados_paises.drop(columns=['country']).groupby(by=['cluster_complete'])
tab_medias_grupo = analise_paises.mean().T
tab_desc_grupo = analise_paises.describe().T
```

## Conclusão

Este projeto demonstra como aplicar métodos de clustering hierárquico aglomerativo e K-Means em dados de países para identificar grupos de países com características semelhantes. A análise inclui a padronização dos dados, geração de dendrogramas, análise de variância e visualização em 3D dos clusters.

## Referências

- [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)

---

### Como Utilizar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   ```

2. Navegue até o diretório do projeto:
   ```bash
   cd nome-do-repositorio
   ```

3. Instale os pacotes necessários:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute o script de análise:
   ```bash
   python script_analise.py
   ```

5. Os resultados serão exibidos diretamente no seu navegador, utilizando o Plotly para visualização dos gráficos 3D.

---

Para quaisquer dúvidas ou sugestões, sinta-se à vontade para abrir uma issue ou enviar um pull request.
