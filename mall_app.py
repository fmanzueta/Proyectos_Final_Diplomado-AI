#----------------------------------------------------------------
# Proyecto Final 2: Segmentación de Clientes (Clustering)
#----------------------------------------------------------------

# 1. Carga y Análisis Exploratorio del Dataset (EDA)

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df_mall = pd.read_csv('Mall_Customers.csv')

# Vistazo inicial a los datos
print("Primeras 5 filas del dataset:")
print(df_mall.head())
print("\nInformación general del dataset:")
df_mall.info()

# Revisar valores faltantes
print(f"\nValores faltantes por columna:\n{df_mall.isnull().sum()}") # No hay valores faltantes

# Análisis exploratorio
print("\nDescripción estadística del dataset:")
print(df_mall.describe())

# Visualizamos las distribuciones de las variables clave
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df_mall['Age'], kde=True, bins=20)
plt.title('Distribución de Edad')

plt.subplot(1, 3, 2)
sns.histplot(df_mall['Annual Income (k$)'], kde=True, bins=20)
plt.title('Distribución de Ingreso Anual')

plt.subplot(1, 3, 3)
sns.histplot(df_mall['Spending Score (1-100)'], kde=True, bins=20)
plt.title('Distribución de Puntuación de Gasto')
plt.tight_layout()
plt.show()

# Para el clustering, nos enfocaremos en 'Annual Income' y 'Spending Score'
X_cluster = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']]
print("\nCaracterísticas seleccionadas para el clustering:")
print(X_cluster.head())


#----------------------------------------------------------------
# 2. Preprocesamiento
#----------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

# Escalar las variables numéricas es crucial para K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

X_scaled_df = pd.DataFrame(X_scaled, columns=X_cluster.columns)
print("\nDatos escalados (primeras 5 filas):")
print(X_scaled_df.head())


#----------------------------------------------------------------
# 3. Determinación del Número de Clusters (k)
#----------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Método del Codo (Elbow Method)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Método del Codo para Determinar k')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS (Suma de cuadrados dentro del clúster)')
plt.grid(True)
plt.show()
print("El 'codo' en la gráfica sugiere que k=5 es una buena opción.")

# Coeficiente de Silueta
silhouette_coefficients = []
for k in range(2, 11): # El coeficiente de silueta se define para k >= 2
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_coefficients, marker='o', linestyle='--')
plt.title('Coeficiente de Silueta para Determinar k')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silueta Promedio')
plt.grid(True)
plt.show()
print("El coeficiente de silueta más alto también se obtiene para k=5.")

# Justificación: Ambos métodos, el codo y la silueta, apuntan a que 5 es el número óptimo de clusters.
# Con k=5 se logra un buen equilibrio entre la cohesión intra-cluster y la separación inter-cluster.


#----------------------------------------------------------------
# 4. Aplicación del Algoritmo K-Means
#----------------------------------------------------------------
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
# Ajustar y predecir los clusters
y_kmeans = kmeans.fit_predict(X_scaled)

# Añadir la etiqueta del cluster al dataframe original
df_mall['Cluster'] = y_kmeans
print("\nPrimeras filas del dataframe con la etiqueta del cluster:")
print(df_mall.head())


#----------------------------------------------------------------
# 5. Visualización de los Clusters
#----------------------------------------------------------------

# Obtener los centroides de los clusters (en el espacio escalado)
centroids_scaled = kmeans.cluster_centers_

plt.figure(figsize=(12, 8))
# Graficar los puntos de datos, coloreados por cluster
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df_mall, palette='viridis', s=100, alpha=0.7, legend='full')

# Para graficar los centroides, necesitamos des-escalarlos
centroids = scaler.inverse_transform(centroids_scaled)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroides')

plt.title('Segmentación de Clientes del Centro Comercial')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.legend()
plt.grid(True)
plt.show()


#----------------------------------------------------------------
# 6. Análisis de Segmentos
#----------------------------------------------------------------

# Analizar las características de cada cluster
cluster_analysis = df_mall.drop('CustomerID', axis=1).groupby('Cluster').mean()
print("\nAnálisis descriptivo de cada cluster:")
print(cluster_analysis)

# Propuesta de Estrategias Específicas
print("\n--- Descripción y Estrategias por Segmento ---")

# Cluster 0: Ingreso medio, puntuación de gasto media
print("\nSegmento 0 (Estándar):")
print("Descripción: Clientes con ingresos y gastos promedio. Representan el cliente 'típico'.")
print("Estrategia: Campañas de marketing generales, programas de lealtad para incentivar un mayor gasto, ofertas en productos populares.")

# Cluster 1: Ingreso alto, puntuación de gasto baja
print("\nSegmento 1 (Ahorradores / Cuidadosos):")
print("Descripción: Clientes con altos ingresos pero que gastan poco. Son cautelosos y buscan valor.")
print("Estrategia: Marketing enfocado en productos de alta calidad, durabilidad y exclusividad. Promociones sutiles y programas de membresía premium.")

# Cluster 2: Ingreso bajo, puntuación de gasto baja
print("\nSegmento 2 (Precavidos):")
print("Descripción: Clientes con bajos ingresos y bajos gastos. Muy sensibles al precio.")
print("Estrategia: Promociones agresivas, descuentos, cupones y ofertas '2x1'. Marketing enfocado en la asequibilidad.")

# Cluster 3: Ingreso bajo, puntuación de gasto alta
print("\nSegmento 3 (Jóvenes / Impulsivos):")
print("Descripción: Clientes con bajos ingresos pero que gastan mucho. Probablemente jóvenes, interesados en tendencias y moda.")
print("Estrategia: Marketing en redes sociales, colaboraciones con influencers, promociones de 'compra ahora y paga después', y enfoque en productos de moda rápida.")

# Cluster 4: Ingreso alto, puntuación de gasto alta
print("\nSegmento 4 (Objetivo / VIP):")
print("Descripción: El segmento ideal. Altos ingresos y alto gasto. Leales a marcas que les gustan.")
print("Estrategia: Trato VIP, acceso anticipado a nuevas colecciones, eventos exclusivos, servicios de personal shopper. Marketing de lujo y experiencias personalizadas.")
