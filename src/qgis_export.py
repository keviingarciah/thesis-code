import pandas as pd
import os
import json

def convert_to_geojson(csv_path, output_path, image_reference):
    df = pd.read_csv(csv_path)

    features = []

    for _, row in df.iterrows():
        img_id = str(row['image'])
        x_pix = row['centroid_x']
        y_pix = row['centroid_y']
        cluster = row['cluster']

        if img_id not in image_reference:
            continue

        ref = image_reference[img_id]
        x_real = ref['top_left_x'] + x_pix * ref['resolution']
        y_real = ref['top_left_y'] - y_pix * ref['resolution']  # y decrece hacia abajo en imagen

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x_real, y_real]
            },
            "properties": {
                "image": img_id,
                "cluster": int(cluster)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"GeoJSON exportado a {output_path}")
