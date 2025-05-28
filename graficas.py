import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

# Cargar datos
ruta_csv = "/Users/christianlomeli/Downloads/nba_historical_stats_final.csv"
if not os.path.exists(ruta_csv):
    print("⚠️ Archivo no encontrado. Verifica la ruta.")
    exit()

# Leer datos y filtrar por temporada 2024-25
df = pd.read_csv(ruta_csv, low_memory=False)
df = df[df['SEASON_ID'] == '2024-25'] 

# Seleccionar columnas relevantes
cols = ["Player", "PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "MIN", "GP", "PLAYER_AGE"]
df = df[cols]

# Procesamiento de datos
numeric_cols = ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "MIN", "GP", "PLAYER_AGE"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna()

if df.empty:
    print("⚠️ No hay datos para la temporada 2024-25.")
    exit()

# Métrica de importancia y selección de top 100 jugadores
df["score"] = df["PTS"] * 0.5 + df["AST"] * 0.3 + df["REB"] * 0.2 + df["MIN"] * 0.1
df = df.nlargest(200, "score").drop(columns=["score"])

# Escalado de datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

# Determinar número óptimo de clusters (Método del codo)
sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(df_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(range(1, 10), sse, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('SSE')
plt.title('Método del Codo - K-Means (2024-25)')
plt.show()

# Aplicar K-Means con k=4
k_optimo = 4
kmeans = KMeans(n_clusters=k_optimo, random_state=0, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(df_scaled)

# Visualización de clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='PTS', y='FG_PCT', hue='kmeans_cluster', data=df, 
                palette='viridis', s=100, alpha=0.8)
plt.title('Clustering de Jugadores NBA 2024-25 (PTS vs AST)')
plt.xlabel('Puntos por Partido')
plt.ylabel('Porcentajes de tiro')

# Etiquetar jugadores top
for i, row in df.nlargest(20, 'PTS').iterrows():
    plt.text(row['PTS']+0.5, row['AST'], row['Player'], 
             fontsize=8, alpha=0.75,
             bbox=dict(facecolor='white', alpha=0.5))

plt.show()

# Visualización avanzada con Plotly
fig = px.scatter_3d(df, x='PTS', y='AST', z='PLAYER_AGE',
                    color='kmeans_cluster', hover_name='Player',
                    title='Jugadores NBA 2024-25 - Cluster Analysis',
                    labels={'PTS': 'Puntos', 'AST': 'Asistencias', 'PLAYER_AGE': 'Edad'})
fig.show()