import cv2
import numpy as np

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def hex_to_hsv(hex_color):
    """Convierte hex a HSV aproximado para OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    color_bgr = np.uint8([[[b, g, r]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    return color_hsv[0][0]

