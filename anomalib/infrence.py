from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

model = Patchcore()
engine = Engine()

dataset = PredictDataset(
    path="./test",
    image_size=(1280, 1280),
)

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path="./model.ckpt",
)

for prediction in predictions:
    print("Image:", prediction.image_path)
    print("Label:", prediction.pred_label)   # 0 normal, 1 anomalous
    print("Score:", prediction.pred_score)
    anomaly_map = prediction.anomaly_map