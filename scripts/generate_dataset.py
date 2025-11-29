import os
import random
from PIL import Image, ImageDraw
import numpy as np

# Clases
CLASSES = ["botella", "lata", "marcador"]

# Tamaño de imagen
IMG_SIZE = 640

# Cantidades
TRAIN_IMAGES = 200
VAL_IMAGES = 50


def create_base_canvas():
    """Crea una imagen de fondo con ruido simple."""
    noise = np.random.randint(220, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(noise)


def draw_bottle(draw):
    """Dibuja una botella simplificada."""
    x = random.randint(100, 420)
    y = random.randint(50, 400)

    width = random.randint(40, 80)
    height = random.randint(150, 280)

    shape = [x, y, x + width, y + height]
    draw.rectangle(shape, fill=(0, 150, 255))

    return shape


def draw_can(draw):
    """Dibuja una lata simple."""
    x = random.randint(150, 400)
    y = random.randint(200, 420)

    width = random.randint(50, 90)
    height = random.randint(80, 150)

    shape = [x, y, x + width, y + height]
    draw.rectangle(shape, fill=(180, 180, 180))

    return shape


def draw_marker(draw):
    """Dibuja un marcador."""
    x = random.randint(80, 450)
    y = random.randint(100, 430)

    width = random.randint(20, 40)
    height = random.randint(120, 200)

    shape = [x, y, x + width, y + height]
    draw.rectangle(shape, fill=(255, 20, 20))

    return shape


def normalize_bbox(shape):
    """Convierte (x1, y1, x2, y2) a formato YOLO."""
    x1, y1, x2, y2 = shape
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return cx / IMG_SIZE, cy / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE


def generate_image(index, split):
    """Genera una imagen y su etiqueta YOLO."""
    img = create_base_canvas()
    draw = ImageDraw.Draw(img)

    objects = []

    # Número de objetos por imagen
    num_objects = random.randint(1, 3)

    for _ in range(num_objects):
        obj_type = random.choice(CLASSES)

        if obj_type == "botella":
            shape = draw_bottle(draw)
        elif obj_type == "lata":
            shape = draw_can(draw)
        else:
            shape = draw_marker(draw)

        bbox = normalize_bbox(shape)
        objects.append((CLASSES.index(obj_type), bbox))

    # Guardar imagen
    img_path = f"./data/{split}/images/{index}.jpg"
    img.save(img_path)

    # Guardar label
    label_path = f"./data/{split}/labels/{index}.txt"
    with open(label_path, "w") as f:
        for cls_id, (cx, cy, w, h) in objects:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def setup_directories():
    """Crea carpetas del dataset."""
    for split in ["train", "val"]:
        os.makedirs(f"./data/{split}/images", exist_ok=True)
        os.makedirs(f"./data/{split}/labels", exist_ok=True)


def main():
    print("Creando dataset sintético...")

    setup_directories()

    # Generar train
    for i in range(TRAIN_IMAGES):
        generate_image(i, "train")

    # Generar val
    for i in range(VAL_IMAGES):
        generate_image(i, "val")

    print("Dataset generado exitosamente.")


if __name__ == "__main__":
    main()
