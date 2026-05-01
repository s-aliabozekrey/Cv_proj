# Cv_proj
<<<<<<< HEAD
=======

Train the bowling-pin detector with:

```bash
python models/yolo_model.py --data "data/Bowling Pin Detection.v1i.yolov12/data.yaml"
```

The trained weights are copied to `models/yolo_weights.pt`. The video pipeline in `src/detection.py` will use, in order, `YOLO_WEIGHTS_PATH`, `models/yolo_weights.pt`, the latest `runs/detect/bowling_pin_detector/weights/best.pt`, and finally the fallback `yolo12n.pt`.

Run the video demo with:

```bash
python src/main.py
```
>>>>>>> 007508f (Document bowling pin workflow)
