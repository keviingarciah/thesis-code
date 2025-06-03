from src.archeo_detection import process_images, get_contours_and_features
from src.kmeans_clustering import process_all_images
from src.qgis_export import convert_to_geojson
from src.spatial_analysis import run_spatial_analysis
import numpy as np
import pandas as pd
from src.utils import generate_color_range

def main():
    input_folder = './data/dataset/' 

    img_path = './results/img/'
    csv_path = './results/features.csv'
    output_dir = './results/clusters/'
    # georef_csv = './data/features_georeferenced.csv'
    # geojson_path = './data/features_clusters.geojson'

    color_to_detect = '#A98876' # Color de la piedra, vasija o material que se quiere detectar.
    color_range = generate_color_range(color_to_detect, 10, 50, 50)

    while True:
        print("\nMenú Principal:")
        print("1. Procesar imágenes y extraer características (OpenCV)")
        print("2. Ejecutar clustering K-Means por imagen")
        print("3. Exportar GeoJSON para QGIS desde CSV georreferenciado")
        print("4. Análisis descriptivo de estadísticas por clúster")
        print("5. Salir")

        option = input("Selecciona una opción: ")

        if option == '1':
            process_images(input_folder, img_path, csv_path, color_range)

        elif option == '2':
            results_df = process_all_images(csv_path, output_dir)
            if results_df.empty:
                print("No se pudieron procesar las imágenes correctamente.")

        elif option == '3':
            # convert_to_geojson(georef_csv, geojson_path, image_reference)
            pass

        elif option == '4':
            # Modificar para analizar cada imagen por separado
            print("Selecciona el archivo de clusters a analizar:")
            import os
            cluster_files = [f for f in os.listdir(output_dir) if f.endswith('_clustered.csv')]
            for i, f in enumerate(cluster_files, 1):
                print(f"{i}. {f}")
            
            try:
                choice = int(input("Número de archivo: "))
                if 1 <= choice <= len(cluster_files):
                    clustered_csv = os.path.join(output_dir, cluster_files[choice-1])
                    run_spatial_analysis(clustered_csv)
                else:
                    print("Opción no válida")
            except ValueError:
                print("Por favor ingresa un número válido")

        elif option == '5':
            print("Saliendo...")
            break

        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
