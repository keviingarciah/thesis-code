import cv2
import numpy as np
import os

def nothing(x):
    pass

def main():
    # Ruta a una imagen de ejemplo
    image_path = './data/dataset/3.png'  # Cambia este nombre si tu imagen se llama diferente
    if not os.path.exists(image_path):
        print(f'No se encuentra la imagen: {image_path}')
        return

    image = cv2.imread(image_path)
    if image is None:
        print('No se pudo cargar la imagen.')
        return

    # Redimensionar para visualización
    image = cv2.resize(image, (780, 540))

    # Convertir a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Crear ventana con trackbars
    cv2.namedWindow('Control HSV')
    cv2.createTrackbar('H Min', 'Control HSV', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'Control HSV', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Control HSV', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Control HSV', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Control HSV', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Control HSV', 255, 255, nothing)

    while True:
        # Obtener valores de trackbars
        h_min = cv2.getTrackbarPos('H Min', 'Control HSV')
        h_max = cv2.getTrackbarPos('H Max', 'Control HSV')
        s_min = cv2.getTrackbarPos('S Min', 'Control HSV')
        s_max = cv2.getTrackbarPos('S Max', 'Control HSV')
        v_min = cv2.getTrackbarPos('V Min', 'Control HSV')
        v_max = cv2.getTrackbarPos('V Max', 'Control HSV')

        # Crear máscara y mostrar resultado
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        combined = np.hstack((image, result))
        cv2.imshow('Original | Filtrado', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Presiona ESC para salir
            break

    cv2.destroyAllWindows()
    print(f'Rango HSV seleccionado:')
    print(f'Lower: {lower}')
    print(f'Upper: {upper}')

if __name__ == "__main__":
    main()
