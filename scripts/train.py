from ultralytics import YOLO

def main():
    # Cargamos un modelo preentrenado (transfer learning)
    model = YOLO("yolov8n.pt")

    # Entrenamiento (fine-tuning)
    results = model.train(
        data="../config/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        name="yolo-botellas-latas-marcadores",
        pretrained=True
    )

    print("Entrenamiento completado.")
    print(results)

if __name__ == "__main__":
    main()
