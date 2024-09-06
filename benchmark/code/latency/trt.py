import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-3])
sys.path.insert(0, ROOT+'/res')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(sys.path)
from res.ultralytics.yolo.engine.model import YOLO

# Load a model
# model = YOLO("/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8n_trained.pt", task='detect')  # load a pretrained model (recommended for training)
model = YOLO(ROOT + "/res/ultralytics/yolo/weights/yolov8s_trained.pt", task='detect')  # load a pretrained model (recommended for training)

# Use the model
# path = model.export(format="engine",device=0,half=True,imgsz=544,workspace=4,batch=15,simplify=True)  # export the model to ONNX format
path = model.export(format="onnx",device=0,half=True,workspace=4,batch=15,imgsz=544)  # export the model to ONNX format
# path = model.export(format="engine",device=0,batch=15,imgsz=544,workspace=2)  # export the model to ONNX format


# /usr/src/tensorrt/bin/trtexec --onnx=/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8m_trained_f16.onnx --saveEngine=/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8m_trained.engine --workspace=2048 --minShapes=input:1x3x288x288 --optShapes=input:15x3x544x544 --maxShapes=input:15x3x544x544 --fp16


