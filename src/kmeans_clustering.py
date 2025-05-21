import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    # Usamos solo características numéricas para clustering
    features = df[['area', 'perimeter', 'centroid_x', 'centroid_y']].values
    return df, features

def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler

def elbow_method(scaled_features, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_features)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(range(1, max_k+1), inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para k óptimo')
    plt.show()

def run_kmeans(scaled_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters, kmeans

def save_clustered_data(df, clusters, output_csv):
    df['cluster'] = clusters
    df.to_csv(output_csv, index=False)
    print(f"Datos con clusters guardados en {output_csv}")

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df['centroid_x'], df['centroid_y'], c=df['cluster'], cmap='viridis')
    plt.xlabel('Centroid X')
    plt.ylabel('Centroid Y')
    plt.title('Distribución de Clusters por Coordenadas de Centroides')
    plt.colorbar(scatter, label='Cluster')
    plt.show()
