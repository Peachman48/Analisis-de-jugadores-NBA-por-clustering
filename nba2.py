import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px



ruta_csv = "/Users/christianlomeli/Downloads/nba_historical_stats_final.csv"


if not os.path.exists(ruta_csv):
    print("⚠️ Archivo no encontrado. Verifica la ruta.")
    exit()

df = pd.read_csv(ruta_csv, low_memory=False)


cols = ["Player", "PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "MIN", "GP"]
df = df[cols]

numeric_cols = ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "MIN", "GP"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna()

# Verificar si hay suficientes datos
if df.empty:
    print("⚠️ No hay suficientes datos numéricos para realizar el clustering.")
    exit()

# 4. Definir una métrica de importancia
df["score"] = df["PTS"] * 0.5 + df["AST"] * 0.3 + df["REB"] * 0.2 + df["MIN"] * 0.1

# 5. Seleccionar solo los 1000 jugadores más importantes
df = df.nlargest(100, "score")
df = df.drop(columns=["score"])  # Eliminar la columna temporal

# 6. Escalar los datos (solo las columnas numéricas)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

# 7. K-MEANS: Determinar el número óptimo de clusters con el método del codo
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(df_scaled)
    sse.append(km.inertia_)

# Graficar el método del codo
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('SSE')
plt.title('Método del Codo - K-Means')
plt.show()

# 8. Aplicar K-Means con el número óptimo de clusters (ajusta manualmente)
k_optimo = 4
kmeans = KMeans(n_clusters=k_optimo, random_state=0, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(df_scaled)

# 9. Clustering Jerárquico
linkage_matrix = linkage(df_scaled, method='ward')

# Graficar dendrograma
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode="level", p=5)
plt.title("Dendrograma - Clustering Jerárquico")
plt.xlabel("Índice de Jugador")
plt.ylabel("Distancia")
plt.show()

# 10. Aplicar Clustering Jerárquico con el mismo número de clusters
hc = AgglomerativeClustering(n_clusters=k_optimo, linkage="ward")
df["hierarchical_cluster"] = hc.fit_predict(df_scaled)

# 11. Visualizar los grupos (PTS vs AST)
plt.figure(figsize=(12, 6))

# K-Means
plt.subplot(1, 2, 1)
sns.scatterplot(x=df["PTS"], y=df["AST"], hue=df["kmeans_cluster"], palette="viridis")
plt.title("Clustering K-Means (PTS vs AST)")
plt.xlabel("Puntos por Partido")
plt.ylabel("Asistencias por Partido")

# Etiquetar algunos jugadores destacados aleatorios
for i, row in df.sample(50).iterrows():  
    plt.text(row["PTS"], row["AST"], row["Player"], fontsize=8, alpha=0.7)

# Clustering Jerárquico
plt.subplot(1, 2, 2)
sns.scatterplot(x=df["PTS"], y=df["AST"], hue=df["hierarchical_cluster"], palette="coolwarm")
plt.title("Clustering Jerárquico (PTS vs AST)")
plt.xlabel("Puntos por Partido")
plt.ylabel("Asistencias por Partido")

# Etiquetar algunos jugadores destacados aleatorios
for i, row in df.sample(50).iterrows():  
    plt.text(row["PTS"], row["AST"], row["Player"], fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()

# 12. Mostrar algunos jugadores de cada grupo
for i in range(k_optimo):
    print(f"\nJugadores en el cluster {i} (K-Means):")
    print(df[df["kmeans_cluster"] == i][["Player", "PTS", "AST", "REB", "FG_PCT"]].head(10))

    print(f"\nJugadores en el cluster {i} (Jerárquico):")
    print(df[df["hierarchical_cluster"] == i][["Player", "PTS", "AST", "REB", "FG_PCT"]].head(10))

    # (El código anterior de carga y preprocesamiento permanece igual hasta el escalado)

# Visualización mejorada
from sklearn.decomposition import PCA

# 1. Reducción de dimensionalidad con PCA para visualizar múltiples variables
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df['PCA1'] = df_pca[:, 0]
df['PCA2'] = df_pca[:, 1]

# 2. Matriz de dispersión para relaciones entre variables
plt.figure(figsize=(15, 10))
sns.pairplot(df, 
             vars=["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "MIN"],
             hue="kmeans_cluster",
             palette="viridis",
             plot_kws={'alpha': 0.6})
plt.suptitle("Relaciones entre Variables por Cluster", y=1.02)
plt.show()

# 3. Visualización PCA con clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='kmeans_cluster', 
                data=df, palette='viridis', s=100, alpha=0.8)

# Etiquetar jugadores importantes
for i, row in df.nlargest(20, 'PTS').iterrows():
    plt.text(row['PCA1']+0.1, row['PCA2'], row['Player'], 
             fontsize=8, alpha=0.75, 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

plt.title("PCA de Jugadores NBA (2 Componentes Principales)")
plt.xlabel(f"Componente 1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
plt.ylabel(f"Componente 2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
plt.grid(alpha=0.3)
plt.show()

# 4. Heatmap de correlación por cluster
plt.figure(figsize=(12, 6))
cluster_means = df.groupby('kmeans_cluster')[numeric_cols].mean()
sns.heatmap(cluster_means.T, 
            annot=True, fmt=".1f", 
            cmap="YlGnBu",
            linewidths=.5)
plt.title("Promedio de Estadísticas por Cluster")
plt.show()

fig = px.scatter_matrix(df, dimensions=["PTS", "AST", "REB"], 
                        color="kmeans_cluster", hover_name="Player")
fig.show()