from ultralytics import YOLO
import cv2
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Detección con YOLO - imagen/video/cámara en vivo")
    parser.add_argument('source', nargs='?', default='0',
                        help="Fuente: ruta a imagen/video, número de cámara (ej. 0) o URL")
    parser.add_argument('--weights', '-w', default='runs/detect/yolo-botellas-latas-marcadores/weights/best.pt',
                        help='Ruta a los pesos')
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: pesos no encontrados en '{weights}'. Indica --weights con la ruta correcta.")
        sys.exit(1)

    model = YOLO(str(weights))

    src = args.source

    # Detectar cámara si es un número (ej. '0')
    is_camera = False
    try:
        cam_index = int(src)
        is_camera = True
    except Exception:
        is_camera = False

    if is_camera:
        print(f"Iniciando detección desde cámara index={cam_index}. Presiona 'q' para salir.")
        # stream=True devuelve un generador de resultados para vídeo/cámara
        try:
            for result in model(cam_index, stream=True):
                img = result.plot()
                cv2.imshow('Detección', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error durante inferencia en cámara: {e}")
            sys.exit(2)
    else:
        # si es ruta local, validar existencia
        if not (src.startswith('http://') or src.startswith('https://')):
            p = Path(src)
            if not p.exists():
                print(f"Error: la ruta indicada '{src}' no existe.")
                sys.exit(3)

        try:
            results = model(src)
        except Exception as e:
            print(f"Error durante inferencia: {e}")
            sys.exit(4)

        for r in results:
            out = r.plot()
            cv2.imshow('Detección', out)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
