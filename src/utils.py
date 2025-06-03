import cv2
import numpy as np

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    color_bgr = np.uint8([[[b, g, r]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    return color_hsv[0][0]

def generate_color_range(hex_color, hue_offset=10, saturation_offset=50, value_offset=50):
    base_hsv = hex_to_hsv(hex_color)  # Obtener el color base en HSV
    lower_bound = np.array([
        max(0, base_hsv[0] - hue_offset),   # Limite inferior para el matiz (H)
        max(0, base_hsv[1] - saturation_offset),  # Limite inferior para la saturación (S)
        max(0, base_hsv[2] - value_offset)  # Limite inferior para el valor (V)
    ])
    upper_bound = np.array([
        min(179, base_hsv[0] + hue_offset),  # Limite superior para el matiz (H)
        min(255, base_hsv[1] + saturation_offset),  # Limite superior para la saturación (S)
        min(255, base_hsv[2] + value_offset)  # Limite superior para el valor (V)
    ])
    return lower_bound, upper_bound
