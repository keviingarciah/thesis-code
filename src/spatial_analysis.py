import pandas as pd
import numpy as np
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns


def create_plots_folder(folder='plots'):
    os.makedirs(folder, exist_ok=True)
    return folder


def generate_descriptive_stats(df, output_csv='plots/descriptive_stats.csv'):
    stats = df.groupby('cluster')[['area', 'perimeter']].agg(['mean', 'std', 'min', 'max']).round(2)
    stats.to_csv(output_csv)
    print(f"Estadísticas descriptivas guardadas en {output_csv}")
    return stats


def plot_boxplots(df, output_folder='plots'):
    sns.set(style="whitegrid")

    for feature in ['area', 'perimeter']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='cluster', y=feature, data=df, palette='Set2')
        plt.title(f'Distribución de {feature.capitalize()} por Clúster')
        plt.xlabel('Clúster')
        plt.ylabel(feature.capitalize())

        output_path = os.path.join(output_folder, f'boxplot_{feature}.png')
        plt.savefig(output_path)
        print(f"Gráfico guardado: {output_path}")
        plt.close()


def compute_ann_index(df, cluster_col='cluster'):
    results = []
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_points = df[df[cluster_col] == cluster_id][['centroid_x', 'centroid_y']].values

        if len(cluster_points) < 2:
            results.append({
                'cluster': cluster_id,
                'R': None,
                'interpretation': 'Insuficientes puntos'
            })
            continue

        tree = KDTree(cluster_points)
        distances, _ = tree.query(cluster_points, k=2)
        nearest_neighbor_distances = distances[:, 1]

        mean_nn_distance = np.mean(nearest_neighbor_distances)

        area = (df['centroid_x'].max() - df['centroid_x'].min()) * (df['centroid_y'].max() - df['centroid_y'].min())
        point_density = len(cluster_points) / area
        expected_distance = 0.5 / np.sqrt(point_density)

        R = mean_nn_distance / expected_distance

        interpretation = "Agrupado" if R < 1 else "Disperso" if R > 1 else "Aleatorio"

        results.append({
            'cluster': cluster_id,
            'R': round(R, 3),
            'interpretation': interpretation
        })

    return pd.DataFrame(results)


def plot_ann_results(results_df, output_path='plots/ann_results.png'):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df['cluster'].astype(str), results_df['R'], color='skyblue')
    for bar, label in zip(bars, results_df['interpretation']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, label, ha='center', fontsize=9)
    plt.xlabel("Clúster")
    plt.ylabel("Índice R de Vecino Más Cercano")
    plt.title("Distribución Espacial por Clúster")
    plt.ylim(0, results_df['R'].max() + 0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Gráfico ANN guardado: {output_path}")
    plt.close()


def generate_image_summary_table(df, output_csv='plots/image_summary.csv'):
    if 'image_filename' not in df.columns:
        print("⚠️ No se encontró la columna 'image' para generar la tabla de resumen por imagen.")
        return None

    summary = df.groupby('image_filename').agg({
        'area': ['count', 'mean', 'std'],
        'perimeter': 'mean'
    }).reset_index()

    summary.columns = ['Imagen', 'Objetos detectados', 'Área promedio', 'Desviación estándar área', 'Perímetro promedio']
    summary = summary.round(2)

    summary.to_csv(output_csv, index=False)
    print(f"📊 Tabla de resumen por imagen guardada en {output_csv}\n")
    print(summary.to_markdown(index=False))

    return summary


def run_spatial_analysis(csv_path, output_folder='plots'):
    df = pd.read_csv(csv_path)

    print("Columnas disponibles:", df.columns.tolist())

    required_columns = ['area', 'perimeter', 'cluster', 'centroid_x', 'centroid_y']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias en el CSV: {missing}")

    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df['perimeter'] = pd.to_numeric(df['perimeter'], errors='coerce')
    df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce')

    output_folder = create_plots_folder(output_folder)

    stats = generate_descriptive_stats(df, os.path.join(output_folder, 'descriptive_stats.csv'))
    plot_boxplots(df, output_folder)

    ann_results = compute_ann_index(df)
    ann_results.to_csv(os.path.join(output_folder, 'ann_results.csv'), index=False)
    plot_ann_results(ann_results, os.path.join(output_folder, 'ann_results.png'))

    image_summary = generate_image_summary_table(df, os.path.join(output_folder, 'image_summary.csv'))

    return stats, ann_results, image_summary
