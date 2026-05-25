from pathlib import Path
from multiprocessing import freeze_support

import cv2
import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, np.ndarray):
        return x.item()
    return x


def normalize_anomaly_map(anomaly_map):
    anomaly_map = tensor_to_numpy(anomaly_map)
    anomaly_map = np.squeeze(anomaly_map)

    anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map = anomaly_map - anomaly_map.min()

    if anomaly_map.max() > 0:
        anomaly_map = anomaly_map / anomaly_map.max()

    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    return anomaly_map


def save_overlay_result(image_path, anomaly_map, label, score, output_dir):
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    h, w = image.shape[:2]

    anomaly_map = normalize_anomaly_map(anomaly_map)
    anomaly_map = cv2.resize(anomaly_map, (w, h))

    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)

    label = int(to_scalar(label))
    score = float(to_scalar(score))

    label_text = "ANOMALY" if label == 1 else "NORMAL"
    text = f"{label_text} | Score: {score:.4f}"

    color = (0, 0, 255) if label == 1 else (0, 255, 0)

    cv2.putText(
        overlay,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
        cv2.LINE_AA,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / image_path.name
    cv2.imwrite(str(save_path), overlay)

    print(f"Saved overlay: {save_path} | {text}")


def main():
    input_folder = "./test"
    output_folder = "./inference_results"
    checkpoint_path = "./model.ckpt"

    model = Patchcore()
    engine = Engine()

    dataset = PredictDataset(
        path=input_folder,
        image_size=(1280, 1280),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=checkpoint_path,
    )

    for batch in predictions:
        image_paths = batch.image_path
        labels = batch.pred_label
        scores = batch.pred_score
        anomaly_maps = batch.anomaly_map

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        for i, image_path in enumerate(image_paths):
            label = labels[i] if len(labels.shape) > 0 else labels
            score = scores[i] if len(scores.shape) > 0 else scores
            anomaly_map = anomaly_maps[i] if len(anomaly_maps.shape) > 2 else anomaly_maps

            save_overlay_result(
                image_path=image_path,
                anomaly_map=anomaly_map,
                label=label,
                score=score,
                output_dir=output_folder,
            )


if __name__ == "__main__":
    freeze_support()
    main()