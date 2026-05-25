#pip install anomalib

from pathlib import Path
from multiprocessing import freeze_support

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():
    dataset_root = Path(r"C:\Users\rahul.bhardwaj\Desktop\anomalib\datasets\my_part")

    print("Dataset root:", dataset_root)
    print("Good folder exists:", (dataset_root / "good").exists())
    print("Defect folder exists:", (dataset_root / "defect").exists())

    datamodule = Folder(
        name="my_part",
        root=dataset_root,
        normal_dir="good",
        abnormal_dir="defect",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,   # important for Windows first run
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        num_neighbors=9,
    )

    engine = Engine(max_epochs=1)
    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    freeze_support()
    main()