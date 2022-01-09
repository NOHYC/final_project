import sys
import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from tool.utils import *
from torchvision import transforms
import torch
try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def GiB(val):
    return val * 1 << 30

def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.", default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


TRT_LOGGER = trt.Logger()

def main(engine_path_yolo, engine_path_lanenet, image_path, image_size, namesfile, num_classes, videoout):

    with get_engine(engine_path_yolo) as engine_yolo,get_engine(engine_path_lanenet) as engine_lanenet, engine_yolo.create_execution_context() as context_yolo, engine_lanenet.create_execution_context() as context_lanenet:
        buffers_yolo = allocate_buffers(engine_yolo, 1)
        buffers_lanenet = allocate_buffers(engine_lanenet, 1)
        IN_IMAGE_H, IN_IMAGE_W = image_size
        context_yolo.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))
        context_lanenet.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))

        cap = cv2.VideoCapture(image_path)
        cap_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(videoout, fourcc, 30, (cap_width, cap_height))
       
        while(cap.isOpened()):
            ret, image_src = cap.read()
            if ret:
    #image_src = cv2.imread(image_path)
                image_src = image_src[0: cap_height, 0 : cap_width]
                ta = time.time()
                for i in range(2):  # This 'for' loop is for speed check
                                    # Because the first iteration is usually longer
                    boxes = detect_yolo(context_yolo, buffers_yolo, image_src, image_size, num_classes)
                lane_img = detect_lane(context_lanenet, buffers_lanenet, image_src)
                class_names = load_class_names(namesfile)
                yolo_img = plot_boxes_cv2(image_src, boxes[0], savename=None, class_names=class_names)
                yolo_lane_img = cv2.addWeighted(lane_img.astype(np.uint8) ,0.15 ,yolo_img.astype(np.uint8) ,0.85 ,0)

                tb = time.time()

                print('-----------------------------------')
                print('    TRT inference fps: %f' % (1/(tb - ta)))
                print('-----------------------------------')
                print("yolo_lane_img : ",yolo_lane_img.shape)
                
                out.write(yolo_lane_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()
def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def detect_yolo(context, buffers, image_src, image_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    inputs[0].host = img_in

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)


    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)


    boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)
    return boxes

def detect_lane(context, buffers, image_src):
    IN_IMAGE_H, IN_IMAGE_W = image_src.shape[1],image_src.shape[0]
    data_transform = transforms.Compose([
        transforms.Resize((256,  512)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dummy_input = Image.fromarray(image_src)
    dummy_input = data_transform(dummy_input)
    input_image = torch.unsqueeze(dummy_input, dim=0).numpy()
    inputs, outputs, bindings, stream = buffers
    
    inputs[0].host = input_image
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    #trt_outputs[0].reshape(1, -1, 1, 4)
    out_img = trt_outputs[2].reshape(3,256,512)*255
    out_img = out_img.transpose((1,2,0))
    return cv2.resize(out_img, dsize=(IN_IMAGE_H, IN_IMAGE_W), interpolation=cv2.INTER_AREA)



if __name__ == '__main__':
    engine_path_yolo = sys.argv[1]
    engine_path_lane = sys.argv[2]
    image_path = sys.argv[3]
    image_size = (int(sys.argv[4]), int(sys.argv[5]))
    filename = sys.argv[6]
    num_classes = sys.argv[7]
    videoout = sys.argv[8]
    main(engine_path_yolo,engine_path_lane, image_path, image_size, filename, num_classes, videoout )
