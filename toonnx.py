import torch
import torch.onnx
from model import unet

def out_to_rgb_np(out):
    CLASSES=('ignore','crack', 'spall', 'rebar')
    PALETTE=[[0, 0, 0],[0, 0, 255], [255, 0, 0], [0, 255, 0]]#bgr
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    return color_seg

#-------------------------导出onnx模型-------------------------
def to_onnx():
    model = unet(n_classes = 4).cuda()
    model.load_state_dict(torch.load(r"best_epoch_weights.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 256, 256,device=device)
    torch.onnx.export(model, dummy_input, "unet.onnx", opset_version=11, verbose=False,
        input_names = ["input"], output_names=["output"])


#-------------------------验证onnx模型-------------------------#
import onnxruntime
import numpy as np

def onnx_test():
    onnx_model_path = "unet.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # 创建示例输入数据
    input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
    # 运行推理
    output = ort_session.run(None, {"input": input_data})
    print(output[0].shape)


def trt_infer():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    with open('sample.engine', "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    for idx in range(engine.num_bindings):
        name = engine.get_tensor_name(idx)
        is_input = engine.get_tensor_mode(name)
        op_type = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print('input id: ',idx, '\\tis input: ', is_input, '\\tbinding name: ', name, '\\tshape: ', shape, '\\ttype: ', op_type)


def onnxtotrt():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(r"unet2.onnx")
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        pass # Error handling code here
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
    serialized_engine = builder.build_serialized_network(network, config)
    with open("sample.engine", "wb") as f:
        f.write(serialized_engine)

def trt_infer2():
    import os
    import cv2
    import time
    import tensorrt as trt
    import pycuda.driver as cuda  #GPU CPU之间的数据传输
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open("sample.engine", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    # 分配CPU锁页内存和GPU显存
    with engine.create_execution_context() as context:
        h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
    # 创建cuda流
    stream = cuda.Stream()

    image = cv2.imread(r"crack000680.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, [2, 0, 1])  # 将通道维度放到最前面
    np.copyto(h_input, image.ravel())
    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        start_time = time.time()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        end_time = time.time()
        # Return the host output. 该数据等同于原始模型的输出数据
        h_output = h_output.reshape((4,256, 256))
        h_output = np.argmax(h_output, axis=0)
        a = np.unique(h_output)
        print(a)
        image_data = out_to_rgb_np(h_output)
        cv2.imwrite(os.path.join("crack.png"), image_data)



trt_infer2()
    
    