from src.archeo_detection import process_images
from src.kmeans_clustering import load_features, scale_features, elbow_method, run_kmeans, save_clustered_data, plot_clusters
import numpy as np

def main():
    input_folder = './data/dataset_tikal/'
    output_folder = './data/results/'
    csv_path = './data/features.csv'
    clustered_csv = './data/features_clustered.csv'

    color_range = (np.array([45, 0, 163]), np.array([179, 255, 255]))  # Ajustar si necesario

    while True:
        print("\nMenú Principal:")
        print("1. Procesar imágenes y extraer características (OpenCV)")
        print("2. Ejecutar clustering K-Means")
        print("3. Salir")
        option = input("Selecciona una opción: ")

        if option == '1':
            process_images(input_folder, output_folder, csv_path, color_range)
        elif option == '2':
            df, features = load_features(csv_path)
            scaled_features, scaler = scale_features(features)
            elbow_method(scaled_features)

            k = int(input("Ingresa el número óptimo de clusters (k) según gráfico: "))
            clusters, kmeans_model = run_kmeans(scaled_features, k)
            save_clustered_data(df, clusters, clustered_csv)
            plot_clusters(df)

        elif option == '3':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
