import torch
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

import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
import torch
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a = (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
device = 'cuda:0'


def GiB(val):
    return val * 1 << 30


# 见文章前段

def allocate_buffers(engine, is_explicit_batch=False, input_shape=None):
    inputs = []
    outputs = []
    bindings = []

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    for binding in engine:

        dims = engine.get_binding_shape(binding)
        print(dims)
        if dims[-1] == -1:
            assert (input_shape is not None)
            dims[-2], dims[-1] = input_shape
        size = trt.volume(dims) * engine.max_batch_size  # The maximum batch size which can be used for inference.
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):  # Determine whether a binding is an input binding.
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings


def preprocess_image(imagepath):
    origin_img = cv2.imread(imagepath)  # BGR
    origin_height = origin_img.shape[0]
    origin_width = origin_img.shape[1]
    new_height = 1248
    new_width = 1248
    pad_img = cv2.resize(origin_img, (new_height, new_width))
    pad_img = pad_img[:, :, ::-1].transpose(2, 0, 1)
    pad_img = pad_img.astype(np.float32)
    pad_img /= 255.0
    pad_img = np.ascontiguousarray(pad_img)
    pad_img = np.expand_dims(pad_img, axis=0)
    return pad_img, (new_height, new_width), (origin_height, origin_width)

def export_onnx(model, image_shape, onnx_path, batch_size=1):
    x, y = image_shape
    img = torch.zeros((batch_size, 3, x, y))
    dynamic_onnx = True
    if dynamic_onnx:
        dynamic_ax = {'input_1': {2: 'image_height', 3: 'image_wdith'},
                      'output_1': {2: 'image_height', 3: 'image_wdith'}}
        torch.onnx.export(model, (img), onnx_path,
                          input_names=["input_1"], output_names=["output_1"], verbose=False, opset_version=11,
                          dynamic_axes=dynamic_ax)
    else:
        torch.onnx.export(model, (img), onnx_path,
                          input_names=["input_1"], output_names=["output_1"], verbose=False, opset_version=11
                          )

def build_engine(onnx_path, using_half, engine_file, dynamic_input=True):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
            network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1  # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(1)
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        ##增加部分
        if dynamic_input:
            profile = builder.create_optimization_profile()
            profile.set_shape("input_1", (1, 3, 512, 512), (1, 3, 1600, 1600), (1, 3, 1024, 1024))
            config.add_optimization_profile(profile)
        # 加上一个sigmoid层
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)
        sigmoid_layer = network.add_activation(previous_output, trt.ActivationType.SIGMOID)
        network.mark_output(sigmoid_layer.get_output(0))
        return builder.build_engine(network, config)

def profile_trt(engine, imagepath, batch_size):
    assert (engine is not None)

    input_image, input_shape = preprocess_image(imagepath)

    segment_inputs, segment_outputs, segment_bindings = allocate_buffers(engine, True, input_shape)

    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        context.active_optimization_profile = 0  # 增加部分
        origin_inputshape = context.get_binding_shape(0)
        # 增加部分
        if (origin_inputshape[-1] == -1):
            origin_inputshape[-2], origin_inputshape[-1] = (input_shape)
            context.set_binding_shape(0, (origin_inputshape))
        input_img_array = np.array([input_image] * batch_size)
        img = torch.from_numpy(input_img_array).float().numpy()
        segment_inputs[0].host = img
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in segment_inputs]  # Copy from the Python buffer src to the device pointer dest (an int or a DeviceAllocation) asynchronously,
        stream.synchronize()  # Wait for all activity on this stream to cease, then return.

        context.execute_async(bindings=segment_bindings,
                              stream_handle=stream.handle)  # Asynchronously execute inference on a batch.
        stream.synchronize()
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in segment_outputs]
        # Copy from the device pointer src (an int or a DeviceAllocation) to the Python buffer dest asynchronously
        stream.synchronize()
        results = np.array(segment_outputs[0].host).reshape(batch_size, input_shape[0], input_shape[1])
        return results.transpose(1, 2, 0)

if __name__ == '__main__':
    onnx_path = 'Singlegpu.onnx'
    usinghalf = True
    batch_size = 1
    imagepath = 'testimgs/'
    engine_file = 'Singlegpu.engine'
    init_engine = True
    load_engine = False
    if init_engine:
        trt_engine = build_engine(onnx_path, usinghalf, engine_file, dynamic_input=True)
        print('engine built successfully!')
        with open(engine_file, "wb") as f:
            f.write(trt_engine.serialize())
        print('save engine successfully')
    if load_engine:
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
        if os.path.isdir(imagepath):
            imagepaths = []
            for imagname in os.listdir(imagepath):
                temppath = os.path.join(imagepath, imagname)
                imagepaths.append(temppath)
        else:
            imagepaths = [imagepath]
        for tempimagepath in imagepaths:
            trt_result = profile_trt(trt_engine, tempimagepath, batch_size, 0, 1)
            trt_result = (trt_result > 0.5)
            cv2.imwrite(tempimagepath.replace('.jpg', 'result1.png'), trt_result * 255)

# def torch2onnx(model_path,onnx_path):
#     # Load a model
#     model = YOLO(model_path,task='detect')  # load a pretrained model (recommended for training)
#
#     test_arr = torch.randn(1,3,544,544)
#     input_names = ['input']
#     output_names = ['output']
#     torch.onnx.export(
#         model,
#         test_arr,
#         onnx_path,
#         verbose=False,
#         opset_version=11,
#         input_names=input_names,
#         output_names=output_names,
#         dynamic_axes={"input":{3:"width"}}            #动态推理W纬度，若需其他动态纬度可以自行修改，不需要动态推理的话可以注释这行
#     )
#     print('->>模型转换成功！')




model_path = "/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8n_trained.pt"
onnx_path = "/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8n_trained_.onnx"