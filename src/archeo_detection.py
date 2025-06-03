import os
import cv2
import numpy as np
import csv
# from src.utils import resize_image # Assuming resize_image is defined elsewhere if needed

def filter_contours(contours, min_area=50, max_area=5000):
    """
    Filtra contornos basado en área.
    
    Args:
        contours: Lista de contornos a filtrar
        min_area: Área mínima permitida (default: 50 píxeles cuadrados)
        max_area: Área máxima permitida (default: 5000 píxeles cuadrados)
    
    Returns:
        Lista de contornos filtrados
    """
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered.append(cnt)
    return filtered

def save_image(folder, filename, img):
    """Saves an image to a specified folder, creating the folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), img)
    print(f"  Saved: {os.path.join(folder, filename)}") # Added print statement for confirmation

def extract_features_from_contours(contours):
    """Extracts area, perimeter, and centroid from contours."""
    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        cx = 0
        cy = 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            # Handle cases where contour area is zero (e.g., a line)
            # You might want to find the bounding box center or skip
            # For now, just using 0,0 but this might need adjustment based on your needs
            print(f"  Warning: Contour with zero area found. Centroid set to (0,0). Area: {area}")

        features.append([area, perimeter, cx, cy])
    return features

def process_images(input_folder, output_folder, csv_path, color_range):
    """
    Processes images in a folder to detect objects based on color,
    extract features from their contours, and save results.
    """
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f"No images found in {input_folder}. Please check the path and file extensions.")
        return

    all_features = []
    for image_file in images:
        print(f"\nProcessing image: {image_file}...")
        img_name = os.path.splitext(image_file)[0]
        img_folder = os.path.join(output_folder, img_name) 

        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        img_path = os.path.join(input_folder, image_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"  Error: Could not load image {img_path}. Skipping.")
            continue

        # 1. Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        save_image(img_folder, '01_hsv.png', hsv)

        # 2. Create mask based on color range
        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)
        save_image(img_folder, '02_mask_initial.png', mask)

        # Ensure mask is uint8 (cv2.inRange should already return this, but good to be sure)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # 3. Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8) # 5x5 kernel
        # MORPH_CLOSE fills small holes in the object
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        save_image(img_folder, '03_mask_closed.png', mask_closed)
        # MORPH_OPEN removes small noise from the background
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        save_image(img_folder, '04_mask_morph_final.png', mask_opened)

        # 4. Find contours
        contours, hierarchy = cv2.findContours(mask_opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"  Raw contours found (before filtering): {len(contours)}")

        # Draw ALL raw contours found (before filtering)
        img_with_raw_contours = image.copy()
        cv2.drawContours(img_with_raw_contours, contours, -1, (0, 0, 255), 2) # Draw all raw contours in RED
        save_image(img_folder, '05_raw_contours.png', img_with_raw_contours)

        # Print areas of raw contours
        if contours:
            print("  Areas of raw contours before filtering:")
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                print(f"    Raw Contour {i}: area = {area:.2f}")
        else:
            print("  No raw contours detected by cv2.findContours.")

        # 5. Filter contours based on area
        min_area = 50  # Ajusta estos valores según tus necesidades
        max_area = 5000  # Ajusta estos valores según tus necesidades
        print(f"  Filtering contours with min_area={min_area}, max_area={max_area}")
        filtered_contours = filter_contours(contours, min_area=min_area, max_area=max_area)
        
        print(f"  Contours after filtering: {len(filtered_contours)}")
        
        if filtered_contours:
            print("  Areas of filtered contours:")
            for i, cnt in enumerate(filtered_contours):
                area = cv2.contourArea(cnt)
                print(f"    Filtered Contour {i}: area = {area:.2f}")

        # 6. Extract features and store them
        features = extract_features_from_contours(filtered_contours)
        for f_idx, f_val in enumerate(features):
            # Prepend image name and contour index to each feature set
            all_features.append([img_name, f_idx] + f_val)

        # 7. Draw filtered contours on the original image
        if filtered_contours:
            output_img_final = image.copy()
            cv2.drawContours(output_img_final, filtered_contours, -1, (0, 255, 0), 3) # Filtered in GREEN
            save_image(img_folder, '06_filtered_contours_final.png', output_img_final)
            print(f"  ✔️  {len(filtered_contours)} objects (contours) detected and drawn in {image_file}.")
        else:
            print(f"  ⚠️  No contours remained after filtering for {image_file}.")
            # Save the original image if no contours are drawn, for reference
            save_image(img_folder, '06_no_filtered_contours.png', image.copy())

    # 8. Save all extracted features to a CSV file
    if all_features:
        # Added 'contour_index' to the header
        header = ['image_filename', 'contour_index', 'area', 'perimeter', 'centroid_x', 'centroid_y']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(all_features)
        print(f"\n✅ All processing completed. Features saved to {csv_path}")
    else:
        print("\n❌ No features were extracted from any image (possibly no contours detected or all filtered out). CSV not generated.")

def get_contours_and_features(input_folder, color_range, min_area=500, max_area=100000):
    """
    A separate function to get contours and features, perhaps for a different workflow.
    This is similar to parts of process_images but returns the data directly.
    """
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_img_names = []
    all_contours_data = [] # Storing contour points can be memory intensive if not needed.
    all_features_data = []

    for image_file in images:
        img_name = os.path.splitext(image_file)[0]
        img_path = os.path.join(input_folder, image_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading {img_path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)
        
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Using the min_area and max_area passed to this function
        filtered_contours = filter_contours(contours, min_area=min_area, max_area=max_area)

        features = extract_features_from_contours(filtered_contours)

        for cnt, f in zip(filtered_contours, features):
            all_contours_data.append(cnt) # Storing actual contour points
            all_features_data.append(f)
            all_img_names.append(img_name) # Image name for each detected contour

        print(f"{image_file}: {len(filtered_contours)} objects detected using get_contours_and_features")

    return all_img_names, all_contours_data, all_features_data