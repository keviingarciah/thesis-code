import os
import cv2
import numpy as np
import csv
from src.utils import resize_image

def filter_contours(contours, min_area=500, max_area=100000):
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered.append(cnt)
    return filtered

def save_image(folder, filename, img):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), img)

def extract_features_from_contours(contours):
    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
        features.append([area, perimeter, cx, cy])
    return features

def process_images(input_folder, output_folder, csv_path, color_range):
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_features = []
    for image_file in images:
        img_name = os.path.splitext(image_file)[0]
        img_folder = os.path.join(output_folder, img_name)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        img_path = os.path.join(input_folder, image_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error cargando {img_path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        save_image(img_folder, 'hsv.png', hsv)

        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)
        save_image(img_folder, 'mask.png', mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        save_image(img_folder, 'mask_morph.png', mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_contours(contours)

        features = extract_features_from_contours(filtered_contours)
        # Añadir etiqueta de imagen para seguimiento
        for f in features:
            all_features.append([img_name] + f)

        # Dibujar contornos
        cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)
        save_image(img_folder, 'contours.png', image)

        print(f"{image_file}: {len(filtered_contours)} objetos detectados")

    # Guardar CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'area', 'perimeter', 'centroid_x', 'centroid_y'])
        writer.writerows(all_features)

    print("Procesamiento completado y CSV generado.")

def get_contours_and_features(input_folder, color_range, min_area=500, max_area=100000):
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_features = []
    all_contours = []
    all_img_names = []

    for image_file in images:
        img_name = os.path.splitext(image_file)[0]
        img_path = os.path.join(input_folder, image_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error cargando {img_path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_contours(contours, min_area, max_area)

        features = extract_features_from_contours(filtered_contours)

        for cnt, f in zip(filtered_contours, features):
            all_contours.append(cnt)
            all_features.append(f)
            all_img_names.append(img_name)

        print(f"{image_file}: {len(filtered_contours)} objetos detectados")

    return all_img_names, all_contours, all_features
