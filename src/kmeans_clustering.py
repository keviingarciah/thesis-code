import os
import cv2  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from scipy.spatial import ConvexHull

PLOT_DIR = './plots'
COLORS = [
    (0, 255, 0),    # Verde
    (255, 0, 0),    # Azul (BGR)
    (0, 0, 255),    # Rojo
    (255, 255, 0),  # Cian
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Amarillo
    (128, 0, 0),    # Azul oscuro
    (0, 128, 0),    # Verde oscuro
    (0, 0, 128),    # Rojo oscuro
    (128, 128, 0),  # Cian oscuro
]

def ensure_plot_dir():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

def load_features(csv_path):
    """
    Carga los datos del CSV asegurando que image_filename sea string
    """
    df = pd.read_csv(csv_path, dtype={'image_filename': str})
    return df

def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler

def find_optimal_k(inertias, k_range):
    """
    Encuentra el valor óptimo de k usando el método del codo con la biblioteca kneed
    """
    try:
        knee = KneeLocator(
            k_range, 
            inertias,
            curve='convex', 
            direction='decreasing',
            interp_method='polynomial'
        )
        if knee.knee is None:
            # Si no se encuentra un punto de codo claro, usar un valor predeterminado
            return min(3, len(k_range))
        return int(knee.knee)
    except Exception as e:
        print(f"Error al encontrar k óptimo: {e}")
        return min(3, len(k_range))  # Valor predeterminado

def elbow_method_per_image(features_df, image_name, max_k=10):
    ensure_plot_dir()
    
    # Convertir image_name a string si es necesario
    image_name = str(image_name)
    
    print(f"Buscando características para imagen: '{image_name}'")
    print(f"Valores únicos de image_filename en el DataFrame: {features_df['image_filename'].unique()}")
    
    # Filtrar características solo para la imagen actual
    img_features = features_df[features_df['image_filename'] == image_name]
    
    # Verificar si hay características para esta imagen
    if img_features.empty:
        print(f"Advertencia: No se encontraron características para la imagen {image_name}")
        print(f"Tipos de datos - image_name: {type(image_name)}, DataFrame column: {features_df['image_filename'].dtype}")
        return None, None
        
    features = img_features[['area', 'perimeter', 'centroid_x', 'centroid_y']].values
    
    print(f"Encontradas {len(features)} características para la imagen {image_name}")
    
    # Verificar si hay suficientes muestras
    n_samples = len(features)
    if n_samples < 2:
        print(f"Advertencia: La imagen {image_name} tiene menos de 2 muestras ({n_samples}). No se puede aplicar clustering.")
        return None, None
    
    # Escalar características
    scaled_features, _ = scale_features(features)
    
    max_k = min(max_k, n_samples)  # Evitar k > número de muestras

    k_range = range(1, max_k + 1)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_features)
        inertias.append(km.inertia_)

    # Encontrar k óptimo
    optimal_k = find_optimal_k(inertias, list(k_range))
    
    # Generar gráfica
    plt.figure()
    plt.plot(k_range, inertias, 'bo-')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'k óptimo = {optimal_k}')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title(f'Método del Codo para k óptimo - {image_name}')
    plt.legend()
    plt.grid(True)

    # Crear subdirectorio para la imagen
    img_plot_dir = os.path.join(PLOT_DIR, image_name)
    if not os.path.exists(img_plot_dir):
        os.makedirs(img_plot_dir)
    
    plot_path = os.path.join(img_plot_dir, 'elbow_method.png')
    plt.savefig(plot_path)
    print(f"Gráfica del método del codo para {image_name} guardada en {plot_path}")
    plt.close()
    
    return scaled_features, optimal_k

def run_kmeans_per_image(features_df, image_name, n_clusters):
    # Convertir image_name a string si es necesario
    image_name = str(image_name)
    
    # Filtrar características solo para la imagen actual
    img_features = features_df[features_df['image_filename'] == image_name]
    
    # Verificar si hay características para esta imagen
    if img_features.empty:
        print(f"Advertencia: No se encontraron características para la imagen {image_name}")
        return None, None
        
    features = img_features[['area', 'perimeter', 'centroid_x', 'centroid_y']].values
    
    # Verificar si hay suficientes muestras
    if len(features) < n_clusters:
        print(f"Advertencia: La imagen {image_name} tiene menos muestras ({len(features)}) que clusters solicitados ({n_clusters})")
        return None, None
    
    

    # Escalar características
    scaled_features, _ = scale_features(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters, kmeans

def save_clustered_data_per_image(features_df, output_dir):
    """Guarda los resultados de clustering para cada imagen en archivos CSV separados"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    unique_images = features_df['image_filename'].unique()
    
    for image_name in unique_images:
        img_features = features_df[features_df['image_filename'] == str(image_name)].copy()
        output_csv = os.path.join(output_dir, f'{str(image_name)}_clustered.csv')
        img_features.to_csv(output_csv, index=False)
        print(f"Datos con clusters para {image_name} guardados en {output_csv}")

def plot_clusters_per_image(features_df, image_name):
    """Genera gráfico de clusters para una imagen específica"""
    # Convertir image_name a string si es necesario
    image_name = str(image_name)
    
    img_features = features_df[features_df['image_filename'] == image_name]
    
    if 'cluster' not in img_features.columns:
        print(f"No se encontró columna 'cluster' para graficar en {image_name}")
        return

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(img_features['centroid_x'], 
                         img_features['centroid_y'], 
                         c=img_features['cluster'], 
                         cmap='tab10', 
                         alpha=0.7)
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title(f'Distribución de Clústeres - {image_name}')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)

    # Crear subdirectorio para la imagen
    img_plot_dir = os.path.join(PLOT_DIR, image_name)
    if not os.path.exists(img_plot_dir):
        os.makedirs(img_plot_dir)
    
    plot_path = os.path.join(img_plot_dir, 'cluster_distribution.png')
    plt.savefig(plot_path)
    print(f"Gráfica de clusters para {image_name} guardada en {plot_path}")
    plt.close()

def draw_cluster_boundaries(image_path, df_clusters, output_path):
    """
    Dibuja los límites de los clusters en la imagen con contornos filtrados.
    
    Args:
        image_path: Ruta a la imagen con contornos filtrados
        df_clusters: DataFrame con las columnas 'centroid_x', 'centroid_y', 'cluster'
        output_path: Ruta donde guardar la imagen resultante
    """
    # Leer la imagen con contornos filtrados
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    # Crear una copia de la imagen para dibujar
    result = image.copy()
    
    # Obtener clusters únicos
    unique_clusters = df_clusters['cluster'].unique()
    
    for cluster_id in unique_clusters:
        # Obtener puntos de este cluster
        cluster_points = df_clusters[df_clusters['cluster'] == cluster_id]
        points = cluster_points[['centroid_x', 'centroid_y']].values
        
        if len(points) < 3:
            # Si hay menos de 3 puntos, dibujar círculos
            color = COLORS[int(cluster_id) % len(COLORS)]
            for point in points:
                cv2.circle(result, (int(point[0]), int(point[1])), 20, color, 2)
        else:
            # Si hay 3 o más puntos, calcular y dibujar el hull convexo
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.round(hull_points).astype(np.int32)
                
                # Dibujar el polígono
                color = COLORS[int(cluster_id) % len(COLORS)]
                cv2.polylines(result, [hull_points], True, color, 3)  # Aumentado el grosor de la línea
                
                # Opcional: rellenar el polígono con transparencia
                overlay = result.copy()
                cv2.fillPoly(overlay, [hull_points], color)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)  # Ajustado la transparencia
                
                # Agregar etiqueta del cluster
                centroid = np.mean(hull_points, axis=0, dtype=np.int32)
                cv2.putText(result, f'Cluster {cluster_id}', 
                          (centroid[0], centroid[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            except Exception as e:
                print(f"Error al dibujar cluster {cluster_id}: {e}")
                continue

    # Guardar la imagen resultante
    cv2.imwrite(output_path, result)
    print(f"Imagen con clusters guardada en: {output_path}")

def process_all_images(csv_path, output_dir):
    """Procesa todas las imágenes en el CSV aplicando clustering a cada una"""
    # Cargar datos
    df = load_features(csv_path)
    
    # Verificar si hay datos
    if df.empty:
        print("Error: El archivo CSV está vacío o no se pudo cargar correctamente")
        return pd.DataFrame()
    
    # Verificar que existan las columnas necesarias
    required_columns = ['image_filename', 'area', 'perimeter', 'centroid_x', 'centroid_y']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Faltan las siguientes columnas en el CSV: {missing_columns}")
        return pd.DataFrame()
    
    unique_images = df['image_filename'].unique()
    if len(unique_images) == 0:
        print("Error: No se encontraron imágenes en el CSV")
        return pd.DataFrame()
    
    results_df = pd.DataFrame()
    
    for image_name in unique_images:
        print(f"\nProcesando imagen: {image_name}")
        
        # Generar gráfico del método del codo y obtener k óptimo
        scaled_features, optimal_k = elbow_method_per_image(df, image_name)
        
        if scaled_features is None or optimal_k is None:
            print(f"Saltando imagen {image_name} debido a errores en el procesamiento")
            continue
            
        print(f"Valor óptimo de k para {image_name}: {optimal_k}")
        
        # Aplicar K-means con k óptimo
        clusters, kmeans_model = run_kmeans_per_image(df, image_name, optimal_k)
        
        if clusters is not None:
            # Actualizar DataFrame con los clusters
            img_df = df[df['image_filename'] == str(image_name)].copy()
            img_df['cluster'] = clusters
            results_df = pd.concat([results_df, img_df])
            
            # Generar gráfico de clusters
            plot_clusters_per_image(results_df, image_name)
            
            # Dibujar clusters en la imagen con contornos filtrados
            try:
                # Construir rutas
                img_number = str(image_name)
                # Usar la imagen con contornos filtrados
                filtered_img_path = f'./results/img/{img_number}/06_filtered_contours_final.png'
                
                # Crear directorio de salida si no existe
                output_img_dir = os.path.join(output_dir, img_number)
                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)
                    
                output_img_path = os.path.join(output_img_dir, f'clusters_visualization.png')
                
                # Verificar si existe la imagen con contornos filtrados
                if not os.path.exists(filtered_img_path):
                    print(f"Error: No se encontró la imagen con contornos filtrados en {filtered_img_path}")
                    continue
                
                # Dibujar y guardar visualización de clusters
                draw_cluster_boundaries(filtered_img_path, img_df, output_img_path)
            except Exception as e:
                print(f"Error al generar visualización de clusters para imagen {image_name}: {e}")
    
    # Guardar resultados
    if not results_df.empty:
        save_clustered_data_per_image(results_df, output_dir)
    else:
        print("Advertencia: No se generaron resultados para ninguna imagen")
    
    return results_df

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
