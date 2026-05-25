# Anomalib PatchCore Training & Inference

This project trains a PatchCore anomaly detection model using Anomalib and runs inference on an image folder. Inference saves overlay images in the result folder.

## Project Structure

```text
project/
│
├── training.py
├── inference.py
├── model.ckpt
│
├── datasets/
│   └── my_part/
│       ├── good/
│       │   ├── img_001.jpg
│       │   ├── img_002.jpg
│       │   └── img_003.jpg
│       │
│       ├── defect/
│       │   ├── defect_001.jpg
│       │   └── defect_002.jpg
│       │
│       └── mask/
│           ├── defect_001.png
│           └── defect_002.png
│
├── test/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
│
└── inference_results/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```

## Dataset Notes

- Put normal/OK images inside `datasets/my_part/good/`.
- Put defect/NG images inside `datasets/my_part/defect/`.
- Masks are optional. Use masks only if you want pixel-level evaluation.
- For PatchCore, training mainly uses good images.

## Install Requirements

```bash
pip install anomalib opencv-python torch torchvision
```

## Train Model

Run:

```bash
python training.py
```

After training, copy or rename the generated checkpoint as:

```text
model.ckpt
```

Place it in the project root.

## Run Inference

Put test images inside:

```text
test/
```

Run:

```bash
python inference.py
```

Results will be saved in:

```text
inference_results/
```

Each output image contains the original image with anomaly heatmap overlay and prediction score.

## Windows Note

If running on Windows, keep this block in both `training.py` and `inference.py`:

```python
if __name__ == "__main__":
    freeze_support()
    main()
```

Also keep `num_workers=0` for the first successful run.
