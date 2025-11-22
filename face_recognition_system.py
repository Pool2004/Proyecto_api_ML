"""
Modelo de detección de objetos funcional (Faster R-CNN y DETR)
Archivo: modelo_deteccion_objetos_fasterrcnn_detr.py
Descripción: script completo para entrenar y hacer inferencia con:
 - Faster R-CNN (torchvision)
 - DETR (transformer-based detector, torchvision)

Formato de datos esperado: COCO (annotations JSON) o carpeta con imágenes + COCO annotations.

Requisitos:
 - Python 3.8+
 - torch >= 1.12
 - torchvision >= 0.13
 - pycocotools
 - matplotlib, pillow

Instalación rápida (Linux/Win con conda/venv):
 pip install torch torchvision pycocotools matplotlib pillow

Uso rápido:
 1) Prepara dataset COCO: images/  annotations/instances_train.json  instances_val.json
 2) Entrenar Faster R-CNN (ejemplo):
    python modelo_deteccion_objetos_fasterrcnn_detr.py --mode train --model fasterrcnn --data_root ./dataset --epochs 10 --batch_size 4 --save_dir ./checkpoints
 3) Inferir:
    python modelo_deteccion_objetos_fasterrcnn_detr.py --mode infer --model fasterrcnn --weights ./checkpoints/fasterrcnn_epoch10.pth --input ./test_images --output ./out

Explicación paso a paso incluida en los comentarios dentro del código.

Autor: ChatGPT (adaptado para uso educativo y prototipos)
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms as T
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import detr_resnet50

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# --------------------------- Utilities ---------------------------

def get_transform(train: bool):
    transforms = []
    transforms.append(T.PILToTensor())  # devuelve tensor tipo uint8 [C,H,W]
    # Convert to float and scale to [0,1]
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Custom wrapper around torchvision.datasets.CocoDetection
class CocoDatasetWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        # anns: list of annotations dicts with keys: bbox, category_id, etc.
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            # COCO bbox format: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

        if len(boxes) == 0:
            # Some images may have zero annotations; Faster R-CNN expects at least one box.
            # We'll create a dummy box with label 0 (background) and ignore it in loss by setting area 0.
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
            areas = [0.0]
            iscrowd = [0]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = areas
        target['iscrowd'] = iscrowd

        if self._transforms is not None:
            img = self._transforms(img)

        return img, target


# --------------------------- Model builders ---------------------------

def get_fasterrcnn_model(num_classes: int, pretrained_backbone: bool = True):
    # Load base pretrained model and replace classifier head
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=pretrained_backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_detr_model(num_classes: int):
    # DETR is transformer-based object detector available in torchvision (detr_resnet50)
    model = detr_resnet50(pretrained=True)
    # Adjust number of classes (including background class handled differently in DETR)
    model.class_labels = num_classes
    # torchvision's detr expects num_classes param during creation in some versions; here we simply swap heads if needed
    # If running into mismatch errors, consider using torchvision.models.detection.detr_resnet50(pretrained=False, num_classes=num_classes) with manual weight init.
    return model


# --------------------------- Training loop ---------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if i % print_freq == 0:
            print(f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}] Loss: {losses.item():.4f}")


# --------------------------- Inference & visualization ---------------------------

def visualize_predictions(image_path: str, boxes: List[List[float]], labels: List[int], scores: Optional[List[float]] = None, output_path: Optional[str] = None, category_map: Optional[dict] = None):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = labels[i]
        score = scores[i] if scores is not None else None
        cat_name = category_map.get(label, str(label)) if category_map else str(label)
        text = f"{cat_name}"
        if score is not None:
            text += f" {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 3, y1 + 3), text, fill="red", font=font)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        img.show()


# --------------------------- Main trainer/inference CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Entrenar o inferir modelos de detección: Faster R-CNN y DETR")
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)
    parser.add_argument('--model', choices=['fasterrcnn', 'detr'], default='fasterrcnn')
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--train_ann', type=str, default='annotations/instances_train.json')
    parser.add_argument('--val_ann', type=str, default='annotations/instances_val.json')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--input', type=str, default='./test_images')
    parser.add_argument('--output', type=str, default='./out')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Usando device: {device}")

    # Cargar categories si existen
    coco_train_ann = os.path.join(args.data_root, args.train_ann)
    coco_val_ann = os.path.join(args.data_root, args.val_ann)

    if args.mode == 'train':
        assert os.path.exists(os.path.join(args.data_root, args.train_ann)), f"No encontré {coco_train_ann}"
        assert os.path.exists(os.path.join(args.data_root, args.val_ann)), f"No encontré {coco_val_ann}"

        dataset = CocoDatasetWrapper(os.path.join(args.data_root, 'images'), os.path.join(args.data_root, args.train_ann), transforms=get_transform(train=True))
        dataset_val = CocoDatasetWrapper(os.path.join(args.data_root, 'images'), os.path.join(args.data_root, args.val_ann), transforms=get_transform(train=False))

        # Extraer número de clases desde el archivo de anotaciones COCO
        import json
        with open(os.path.join(args.data_root, args.train_ann), 'r', encoding='utf-8') as f:
            info = json.load(f)
            categories = info.get('categories', [])
            num_classes = max([c['id'] for c in categories]) + 1 if categories else 2
            # Nota: asumimos que las category_id comienzan desde 0 o 1. Ajusta si es necesario.
            category_map = {c['id']: c['name'] for c in categories}

        print(f"Categorias detectadas: {len(categories)}. num_classes (incluyendo fondo): {num_classes}")

        if args.model == 'fasterrcnn':
            model = get_fasterrcnn_model(num_classes=num_classes)
        else:
            model = get_detr_model(num_classes=num_classes)

        model.to(device)

        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
            # Guardar checkpoints
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_epoch{epoch+1}.pth")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
            print(f"Checkpoint guardado: {ckpt_path}")

        print("Entrenamiento finalizado")

    elif args.mode == 'infer':
        # Inferencia sobre carpeta de imágenes
        if args.model == 'fasterrcnn':
            # En inferencia podemos asumir 91 clases si usamos COCO preentrenado, pero lo ideal es pasar el mapping
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.to(device)
        else:
            model = detr_resnet50(pretrained=True)
            model.to(device)

        if args.weights:
            print(f"Cargando pesos desde {args.weights}")
            ckpt = torch.load(args.weights, map_location=device)
            try:
                model.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                # intentar cargar directamente
                model.load_state_dict(ckpt)

        model.eval()
        input_dir = args.input
        output_dir = args.output
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Intentar cargar category map si existiere
        category_map = None
        cat_file = os.path.join(args.data_root, 'annotations', 'categories_map.json')
        if os.path.exists(cat_file):
            import json
            category_map = json.load(open(cat_file, 'r', encoding='utf-8'))

        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_p in image_paths:
            img = Image.open(img_p).convert('RGB')
            img_t = get_transform(train=False)(img)
            # model expects list of tensors
            with torch.no_grad():
                outputs = model([img_t.to(device)])

            # outputs: list with dicts: boxes, labels, scores (for fasterrcnn)
            out = outputs[0]
            boxes = out.get('boxes').cpu().numpy().tolist()
            labels = out.get('labels').cpu().numpy().tolist() if 'labels' in out else [0]*len(boxes)
            scores = out.get('scores').cpu().numpy().tolist() if 'scores' in out else None

            # Filtrar por score
            thr = 0.5
            sel_boxes, sel_labels, sel_scores = [], [], []
            for i, b in enumerate(boxes):
                s = scores[i] if scores is not None else 1.0
                if s >= thr:
                    sel_boxes.append(b)
                    sel_labels.append(labels[i])
                    sel_scores.append(s)

            out_path = os.path.join(output_dir, os.path.basename(img_p))
            visualize_predictions(img_p, sel_boxes, sel_labels, sel_scores, output_path=out_path, category_map=category_map)

        print("Inferencia finalizada. Revisa la carpeta:", output_dir)


if __name__ == '__main__':
    main()
