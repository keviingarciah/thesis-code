import os
import cv2  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PLOT_DIR = './plots'

def ensure_plot_dir():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    features = df[['area', 'perimeter', 'centroid_x', 'centroid_y']].values
    return df, features

def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler

def elbow_method(scaled_features, max_k=10):
    ensure_plot_dir()

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
    plt.grid(True)
    
    # Guardar gráfica
    plot_path = os.path.join(PLOT_DIR, 'elbow_method.png')
    plt.savefig(plot_path)
    print(f"Gráfica del método del codo guardada en {plot_path}")
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
    if 'cluster' not in df.columns:
        print("No se encontró columna 'cluster' para graficar.")
        return

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df['centroid_x'], df['centroid_y'], c=df['cluster'], cmap='tab10', alpha=0.7)
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Distribución de Clústeres')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)

    # Guardar gráfica
    plot_path = os.path.join(PLOT_DIR, 'cluster_distribution.png')
    plt.savefig(plot_path)
    print(f"Gráfica del método del codo guardada en {plot_path}")
    plt.show()

def calculate_centroids(contours):
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroids.append((cX, cY))
    return centroids

def save_clusters_with_centroids(img_names, contours, clusters, output_csv_path):
    centroids = calculate_centroids(contours)
    data = []
    for img_name, (cX, cY), cluster in zip(img_names, centroids, clusters):
        data.append({
            'image': img_name,
            'centroid_x': cX,
            'centroid_y': cY,
            'cluster': cluster
        })
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Archivo CSV georreferenciado guardado en {output_csv_path}")
