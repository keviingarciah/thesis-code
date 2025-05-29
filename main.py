from src.archeo_detection import process_images, get_contours_and_features
from src.kmeans_clustering import load_features, scale_features, elbow_method, run_kmeans, save_clustered_data, plot_clusters, save_clusters_with_centroids
from src.qgis_export import convert_to_geojson
from src.spatial_analysis import run_spatial_analysis
import numpy as np
import pandas as pd

def main():
    # input_folder = './data/dataset_tikal/' Structures
    # input_folder = './data/dataset_materials/'

    input_folder = './data/dataset/'
    output_folder = './data/results/'
    csv_path = './data/features.csv'
    clustered_csv = './data/features_clustered.csv'
    georef_csv = './data/features_georeferenced.csv'
    geojson_path = './data/features_clusters.geojson'

    # color_range = (np.array([45, 0, 163]), np.array([179, 255, 255])) Structures
    # color_range = (np.array([0, 0, 0]), np.array([180, 255, 90]))  
    color_range = (np.array([0, 19, 0]), np.array([35, 255, 255]))  

    image_reference = {
        '1': {'top_left_x': 500000, 'top_left_y': 1700000, 'resolution': 0.05},
        '2': {'top_left_x': 500050, 'top_left_y': 1700000, 'resolution': 0.05},
        '3': {'top_left_x': 500100, 'top_left_y': 1700000, 'resolution': 0.05},
        '4': {'top_left_x': 500150, 'top_left_y': 1700000, 'resolution': 0.05}
    }

    while True:
        print("\nMenú Principal:")
        print("1. Procesar imágenes y extraer características (OpenCV)")
        print("2. Ejecutar clustering K-Means")
        print("3. Exportar GeoJSON para QGIS desde CSV georreferenciado")
        print("4. Análisis descriptivo de estadísticas por clúster")
        print("5. Salir")

        option = input("Selecciona una opción: ")

        if option == '1':
            process_images(input_folder, output_folder, csv_path, color_range)

        elif option == '2':
            img_names, contours, features = get_contours_and_features(input_folder, color_range)
            df, features = load_features(csv_path)
            scaled_features, scaler = scale_features(features)
            elbow_method(scaled_features)

            k = int(input("Ingresa el número óptimo de clusters (k) según gráfico: "))
            clusters, kmeans_model = run_kmeans(scaled_features, k)

            save_clustered_data(df, clusters, clustered_csv)
            save_clusters_with_centroids(img_names, contours, clusters, georef_csv)
            
            df_geo = pd.read_csv(georef_csv)
            plot_clusters(df_geo)

        elif option == '3':
            convert_to_geojson(georef_csv, geojson_path, image_reference)

        elif option == '4':
            run_spatial_analysis(clustered_csv)

        elif option == '5':
            print("Saliendo...")
            break

        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
