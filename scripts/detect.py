from ultralytics import YOLO
import cv2

def main():
    # Cargar el modelo entrenado (runs/detect/.../weights/best.pt)
    model = YOLO("runs/detect/yolo-botellas-latas-marcadores/weights/best.pt")

    # Imagen de prueba
    img_path = "test.jpg"

    results = model(img_path)

    # Mostrar el resultado
    for r in results:
        out = r.plot()
        cv2.imshow("Detección", out)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
