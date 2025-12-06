import cv2
import numpy as np

# ------------------------------
# 1. Validación de keypoints
# ------------------------------
def keypoints_valid(keypoints, min_visible=5):
    """
    Revisa si los keypoints tienen suficientes puntos visibles.
    YOLO Pose retorna cada kp como [x, y, visible]
    """
    visible_count = sum([1 for _, _, v in keypoints if v > 0])
    return visible_count >= min_visible


# ------------------------------
# 2. Dibujar keypoints + líneas
# ------------------------------
# Conexiones estilo MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

def draw_hand_keypoints(img, keypoints):
    """
    Dibuja los keypoints de MediaPipe-style (21 puntos)
    """
    kps = [(int(x), int(y)) for x, y, v in keypoints]

    # Dibujar puntos
    for (x, y) in kps:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

    # Dibujar líneas entre puntos
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, kps[a], kps[b], (255, 0, 0), 2)

    return img


# ------------------------------
# 3. Utilidad: normalizar keypoints YOLO
# ------------------------------
def yolo_keypoints_to_xy(keypoints):
    """
    YOLO Pose retorna keypoints como tensor.
    Esta función los pasa a lista [(x,y,v),...]
    """
    return [(float(k[0]), float(k[1]), float(k[2])) for k in keypoints]


# ------------------------------
# 4. Calcular distancia entre 2 puntos
# ------------------------------
def distance(p1, p2):
    """
    Distancia Euclidiana
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))
