import torch


def torch2onnx(model_path,onnx_path):
    model = torch.load_model(model_path)
    test_arr = torch.randn(1,3,32,448)
    input_names = ['input']
    output_names = ['output']
    tr_onnx.export(
        model,
        test_arr,
        onnx_path,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input":{3:"width"}}            #动态推理W纬度，若需其他动态纬度可以自行修改，不需要动态推理的话可以注释这行
    )
    print('->>模型转换成功！')