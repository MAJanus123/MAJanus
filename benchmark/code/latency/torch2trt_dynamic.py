import sys
from copy import deepcopy
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-3])
sys.path.insert(0, ROOT+'/res')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(sys.path)
import torch
from res.ultralytics.yolo.engine.model import YOLO
from torch import nn
from torchvision.models.resnet import resnet50

# create some regular pytorch model...
model = YOLO(ROOT + "/res/ultralytics/yolo/weights/yolov8n_trained.pt", task='detect')
model = deepcopy(model).to('cuda:0')


def torch2onnx(model,onnx_path):
    im = torch.ones((1, 3, 544, 544)).to('cuda:0')
    # im, model = im.half(), model.half()  # to FP16
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(
        model,
        im,
        onnx_path,
        verbose=False,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names: {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
            output_names: {0: 'batch_size', 2: 'output_height', 3: 'output_width'}}#动态推理W纬度，若需其他动态纬度可以自行修改，不需要动态推理的话可以注释这行
    )
    print('->>模型转换成功！')

onnx_path='ROOT + "/res/ultralytics/yolo/weights/yolov8n_trained_torch2onnx.onnx'
torch2onnx(model,onnx_path)