import onnx

def change_input_dim(model,):
    batch_size = "N"

    # The following code changes the first dimension of every input to be batch_size
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_size
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        if isinstance(batch_size, str):
            # set dynamic batch size
            dim1.dim_param = batch_size
        elif (isinstance(batch_size, str) and batch_size.isdigit()) or isinstance(batch_size, int):
            # set given batch size
            dim1.dim_value = int(batch_size)
        else:
            # set batch size of 1
            dim1.dim_value = 1

def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model,)
    onnx.save(model, outfile)

input_path = '/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8n_trained.onnx'
output_path = '/home/nano/xyb/Video-Analytics-Task-Offloading_MAPPO_nano/res/ultralytics/yolo/weights/yolov8n_trained_dynamic_15_nosim.onnx'
apply(change_input_dim,input_path,output_path)