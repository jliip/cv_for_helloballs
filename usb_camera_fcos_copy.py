#!/usr/bin/env python3

import sys
import signal
import os
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import colorsys
from time import time, sleep
import serial
import ctypes
import json
import serial.tools.list_ports
import threading
import queue
import concurrent.futures
# import test3

try:
    com = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        timeout=1 
    )
    flag = com.is_open
    if(flag):
        print("open esp32 successfully")
    else:
        print("open esp32 failed")
except serial.SerialException as e:
    print(f"Failed to open serial port: {e}")
    com = None

def keyboard_listener(stop_event):
    while not stop_event.is_set():
        user_input = input()
        if user_input.strip().lower() == "q":
            stop_event.set()
            break


def signal_handler(signal, frame):
    print("\nExiting program")
    sys.exit(0)
    pass
output_tensors = None

postprocess_info = None

class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr",ctypes.c_double),
        ("virAddr",ctypes.c_void_p),
        ("memSize",ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen",ctypes.c_int),
        ("shiftData",ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",ctypes.c_int),
        ("scaleData",ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen",ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize",ctypes.c_int * 8),
        ("numDimensions",ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",hbDNNTensorShape_t),
        ("alignedShape",hbDNNTensorShape_t),
        ("tensorLayout",ctypes.c_int),
        ("tensorType",ctypes.c_int),
        ("shift",hbDNNQuantiShift_yt),
        ("scale",hbDNNQuantiScale_t),
        ("quantiType",ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize",ctypes.c_int),
        ("stride",ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",hbSysMem_t * 4),
        ("properties",hbDNNTensorProperties_t)
    ]


class FcosPostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",ctypes.c_int),
        ("width",ctypes.c_int),
        ("ori_height",ctypes.c_int),
        ("ori_width",ctypes.c_int),
        ("score_threshold",ctypes.c_float),
        ("nms_threshold",ctypes.c_float),
        ("nms_top_k",ctypes.c_int),
        ("is_pad_resize",ctypes.c_int)
    ]



libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

get_Postprocess_result = libpostprocess.FcosPostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(FcosPostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)

def limit_display_cord(coor):
    coor[0] = max(min(1920, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(1080, coor[1]), 2)
    coor[2] = max(min(1920, coor[2]), 0)
    coor[3] = max(min(1080, coor[3]), 0)
    return coor

# detection model class names
def get_classes():
    return np.array(["person", "bicycle", "car",
                     "motorcycle", "airplane", "bus",
                     "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign",
                     "parking meter", "bench", "bird",
                     "cat", "dog", "horse",
                     "sheep", "cow", "elephant",
                     "bear", "zebra", "giraffe",
                     "backpack", "umbrella", "handbag",
                     "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball",
                     "kite", "baseball bat", "baseball glove",
                     "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup",
                     "fork", "knife", "spoon",
                     "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza",
                     "donut", "cake", "chair",
                     "couch", "potted plant", "bed",
                     "dining table", "toilet", "tv",
                     "laptop", "mouse", "remote",
                     "keyboard", "cell phone", "microwave",
                     "oven", "toaster", "sink",
                     "refrigerator", "book", "clock",
                     "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"])

# bgr格式图片转换成 NV12格式
def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def draw_bboxs(image, bboxes, ori_w, ori_h, target_w, target_h, classes=get_classes()):
    """draw the bboxes in the original image and rescale the coordinates"""
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    # Scaling factors from original to target resolution
    scale_x = target_w / ori_w
    scale_y = target_h / ori_h

    for i, result in enumerate(bboxes):
        bbox = result['bbox']  # 矩形框位置信息
        score = result['score']  # 得分
        id = int(result['id'])  # id
        name = result['name']  # 类别名称
        if name != 'sports ball':
            continue
			
        # coor = limit_display_cord(bbox)
        coor = [round(i) for i in bbox]
        # Rescale the bbox coordinates
        coor[0] = int(coor[0] * scale_x)
        coor[1] = int(coor[1] * scale_y)
        coor[2] = int(coor[2] * scale_x)
        coor[3] = int(coor[3] * scale_y)
        area = abs(coor[2]-coor[0])*(coor[3]-coor[1])
        x = int(((coor[2]-coor[0])/2) + coor[0])
        y = 1080-abs(int(((coor[3]-coor[1])/2)+coor[1]))
        bbox_color = colors[id]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = name
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(bbox_mess,
                                 0,
                                 fontScale,
                                 thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(image,
                    bbox_mess, (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA)
        if(classes_name == "sports ball"):
            # print("{} is in the picture with confidence: location: x={}, y={}".format(classes_name, x, y))
            print("{} location: x={}, y={}".format(classes_name, x, y))
            if com:
                com.write("x={}, y={}\n".format(x,y).encode('utf-8'))
    #    cv2.imwrite("demo.jpg", image)
    return image

def get_display_res():
    if os.path.exists("/usr/bin/get_hdmi_res") == False:
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])


def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

def capture_frames(cap, frame_queue, stop_event, desired_fps):
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                frame_queue.put(frame)
                sleep(1 / desired_fps)  # Control the capture rate
            else:
                print("Failed to get image from usb camera")
                stop_event.set()
    finally:
        if cap.isOpened():
            cap.release()
        print("camera resource recovered!")

def process_frame(frame, models, output_tensors, fcos_postprocess_info, disp_w, disp_h):
    h, w = models[0].inputs[0].properties.shape[2], models[0].inputs[0].properties.shape[3]
    des_dim = (w, h)
    resized_data = cv2.resize(frame, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)

    outputs = models[0].forward(nv12_data)

    strides = [8, 16, 32, 64, 128]
    for i in range(len(strides)):
        if output_tensors[i].properties.quantiType == 0:
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)

        libpostprocess.FcosdoProcess(output_tensors[i], output_tensors[i + 5], output_tensors[i + 10], ctypes.pointer(fcos_postprocess_info), i)

    result_str = get_Postprocess_result(ctypes.pointer(fcos_postprocess_info))
    result_str = result_str.decode('utf-8')

    data = json.loads(result_str[14:])
    if frame.shape[0] != disp_h or frame.shape[1] != disp_w:
        frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    box_bgr = draw_bboxs(frame, data, fcos_postprocess_info.width, fcos_postprocess_info.height, disp_w, disp_h)
    return box_bgr

def process_frames(frame_queue, stop_event, video_write_queue, desired_fps):
    models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')
    print_properties(models[0].inputs[0].properties)
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)

    disp = srcampy.Display()
    disp_w, disp_h = get_display_res()
    disp.display(0, disp_w, disp_h)

    fcos_postprocess_info = FcosPostProcessInfo_t()
    fcos_postprocess_info.height = 512
    fcos_postprocess_info.width = 512
    fcos_postprocess_info.ori_height = disp_h
    fcos_postprocess_info.ori_width = disp_w
    fcos_postprocess_info.score_threshold = 0.3
    fcos_postprocess_info.nms_threshold = 0.4
    fcos_postprocess_info.nms_top_k = 5
    fcos_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
        if len(models[0].outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
        else:
            output_tensors[i].properties.quantiType = 2
            scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])
            output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for j in range(len(models[0].outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
            output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]

    start_time = time()
    image_counter = 0
    frame_skip = int(30 / desired_fps)  # Adjust frame skipping based on desired FPS
    frame_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        while not stop_event.is_set():
            if frame_queue.empty():
                continue
            frame = frame_queue.get()
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            future = executor.submit(process_frame, frame, models, output_tensors, fcos_postprocess_info, disp_w, disp_h)
            box_bgr = future.result()
            video_write_queue.put(box_bgr)

            finish_time = time()
            image_counter += 1
            if finish_time - start_time > 10:
                print(start_time, finish_time, image_counter)
                print("FPS: {:.2f}".format(image_counter / (finish_time - start_time)))
                start_time = finish_time
                image_counter = 0

    cap.release()
    cv2.destroyAllWindows()
    print("ok")

def serial_communication(com, stop_event):
    if not com:
        return
    while not stop_event.is_set():
        if com.in_waiting > 0:
            data = com.read(com.in_waiting)
            print(f"Received from serial: {data.decode('utf-8')}")
        # Add a small sleep to prevent busy-waiting
        sleep(0.1)

def show_preview(frame_queue, stop_event, models, output_tensors, fcos_postprocess_info, disp_w, disp_h):
    while not stop_event.is_set():
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        box_bgr = process_frame(frame, models, output_tensors, fcos_postprocess_info, disp_w, disp_h)
        cv2.imshow("Preview", box_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

class DroppingQueue(queue.Queue):
    def put(self, item, block=False, timeout=None):
        try:
            super().put(item, block=False)
        except queue.Full:
            # 丢弃最旧的一帧，腾出空间
            try:
                self.get_nowait()
            except queue.Empty:
                pass
            super().put(item, block=False)

def video_writer(video_write_queue, stop_event):
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (disp_w, disp_h))
    while not stop_event.is_set() or not video_write_queue.empty():
        try:
            frame = video_write_queue.get(timeout=1)
            out.write(frame)
        except queue.Empty:
            continue
    out.release()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) > 1:
        video_device = sys.argv[1]
    else:
        video_device = find_first_usb_camera()
    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)
    if(not cap.isOpened()):
        exit(-1)
    
    # keyboard_thread = threading.Thread(
    #     target=keyboard_listener,
    #     args=(stop_event,),
    #     daemon=True
    # )
    # keyboard_thread.start()
    
    
    print("Open usb camera successfully")

    output_video_save = 'CAMERA_OUT.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # frame_rate = 30
    # frame_size = (640,480)
    # out = cv2.VideoWriter(output_video_save,fourcc,frame_rate,frame_size)

    # 设置usb camera的输出图像��式为 MJPEG， 分辨率 640 x 480
    # 可以通过 v4l2-ctl -d /dev/video8 --list-formats-ext 命令查看摄像头支持的分辨率
    # 根据应用需求调整该采集图像的分辨率
    codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    #codec = cv2.VideoWriter_fourcc(*'mp4v')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    videoc = cv2.VideoWriter_fourcc('X','V','I','D')
    # out = cv2.VideoWriter('video.avi',videoc,cv2.CAP_PROP_FPS,(1920,1080))

    frame_queue = DroppingQueue(maxsize=30)     #queue.Queue(maxsize=30)
    stop_event = threading.Event()
    serial_stop_event = threading.Event()
    video_write_queue = queue.Queue()

    desired_fps = 30  # Set your desired FPS here

    disp_w, disp_h = get_display_res()
    models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')
    fcos_postprocess_info = FcosPostProcessInfo_t()
    fcos_postprocess_info.height = 512
    fcos_postprocess_info.width = 512
    fcos_postprocess_info.ori_height = disp_h
    fcos_postprocess_info.ori_width = disp_w
    fcos_postprocess_info.score_threshold = 0.3
    fcos_postprocess_info.nms_threshold = 0.4
    fcos_postprocess_info.nms_top_k = 5
    fcos_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
        if len(models[0].outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
        else:
            output_tensors[i].properties.quantiType = 2
            scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])
            output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for j in range(len(models[0].outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
            output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]

    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, stop_event, desired_fps))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, stop_event, video_write_queue, desired_fps))
    preview_thread = threading.Thread(target=show_preview, args=(frame_queue, stop_event, models, output_tensors, fcos_postprocess_info, disp_w, disp_h))
    serial_thread = None
    # calculation_thread = threading.Thread(target=test3.calculation_thread, args=(test3.shared_coordinates, test3.serial_port))
    if com:
        serial_thread = threading.Thread(target=serial_communication, args=(com, serial_stop_event))
        serial_thread.start()
    video_write_thread = threading.Thread(
        target=video_writer,
        args=(video_write_queue, stop_event)
    )
    video_write_thread.start()
    capture_thread.start()
    process_thread.start()
    # preview_thread.start()


    try:

        capture_thread.join()
        process_thread.join()
        # preview_thread.join()
        # video_write_thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        serial_stop_event.set()
        capture_thread.join()
        process_thread.join()
        if preview_thread:
            preview_thread.join()
        # if video_write_thread:
        #     video_write_thread.join()
        if serial_thread:
            serial_thread.join()
    stop_event.set()
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    print("ok")
#!/usr/bin/env python3

import sys
import signal
import os
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import colorsys
from time import time, sleep
import serial
import ctypes
import json
import serial.tools.list_ports
import threading
import queue
import concurrent.futures
# import test3

try:
    com = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        timeout=1 
    )
    flag = com.is_open
    if(flag):
        print("open esp32 successfully")
    else:
        print("open esp32 failed")
except serial.SerialException as e:
    print(f"Failed to open serial port: {e}")
    com = None

def keyboard_listener(stop_event):
    while not stop_event.is_set():
        user_input = input()
        if user_input.strip().lower() == "q":
            stop_event.set()
            break


def signal_handler(signal, frame):
    print("\nExiting program")
    sys.exit(0)
    pass
output_tensors = None

postprocess_info = None

class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr",ctypes.c_double),
        ("virAddr",ctypes.c_void_p),
        ("memSize",ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen",ctypes.c_int),
        ("shiftData",ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",ctypes.c_int),
        ("scaleData",ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen",ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize",ctypes.c_int * 8),
        ("numDimensions",ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",hbDNNTensorShape_t),
        ("alignedShape",hbDNNTensorShape_t),
        ("tensorLayout",ctypes.c_int),
        ("tensorType",ctypes.c_int),
        ("shift",hbDNNQuantiShift_yt),
        ("scale",hbDNNQuantiScale_t),
        ("quantiType",ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize",ctypes.c_int),
        ("stride",ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",hbSysMem_t * 4),
        ("properties",hbDNNTensorProperties_t)
    ]


class FcosPostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",ctypes.c_int),
        ("width",ctypes.c_int),
        ("ori_height",ctypes.c_int),
        ("ori_width",ctypes.c_int),
        ("score_threshold",ctypes.c_float),
        ("nms_threshold",ctypes.c_float),
        ("nms_top_k",ctypes.c_int),
        ("is_pad_resize",ctypes.c_int)
    ]



libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

get_Postprocess_result = libpostprocess.FcosPostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(FcosPostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)

def limit_display_cord(coor):
    coor[0] = max(min(1920, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(1080, coor[1]), 2)
    coor[2] = max(min(1920, coor[2]), 0)
    coor[3] = max(min(1080, coor[3]), 0)
    return coor

# detection model class names
def get_classes():
    return np.array(["person", "bicycle", "car",
                     "motorcycle", "airplane", "bus",
                     "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign",
                     "parking meter", "bench", "bird",
                     "cat", "dog", "horse",
                     "sheep", "cow", "elephant",
                     "bear", "zebra", "giraffe",
                     "backpack", "umbrella", "handbag",
                     "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball",
                     "kite", "baseball bat", "baseball glove",
                     "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup",
                     "fork", "knife", "spoon",
                     "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza",
                     "donut", "cake", "chair",
                     "couch", "potted plant", "bed",
                     "dining table", "toilet", "tv",
                     "laptop", "mouse", "remote",
                     "keyboard", "cell phone", "microwave",
                     "oven", "toaster", "sink",
                     "refrigerator", "book", "clock",
                     "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"])

# bgr格式图片转换成 NV12格式
def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def draw_bboxs(image, bboxes, ori_w, ori_h, target_w, target_h, classes=get_classes()):
    """draw the bboxes in the original image and rescale the coordinates"""
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    # Scaling factors from original to target resolution
    scale_x = target_w / ori_w
    scale_y = target_h / ori_h

    for i, result in enumerate(bboxes):
        bbox = result['bbox']  # 矩形框位置信息
        score = result['score']  # 得分
        id = int(result['id'])  # id
        name = result['name']  # 类别名称
        if name != 'sports ball':
            continue
			
        # coor = limit_display_cord(bbox)
        coor = [round(i) for i in bbox]
        # Rescale the bbox coordinates
        coor[0] = int(coor[0] * scale_x)
        coor[1] = int(coor[1] * scale_y)
        coor[2] = int(coor[2] * scale_x)
        coor[3] = int(coor[3] * scale_y)
        area = abs(coor[2]-coor[0])*(coor[3]-coor[1])
        x = int(((coor[2]-coor[0])/2) + coor[0])
        y = 1080-abs(int(((coor[3]-coor[1])/2)+coor[1]))
        bbox_color = colors[id]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = name
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(bbox_mess,
                                 0,
                                 fontScale,
                                 thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(image,
                    bbox_mess, (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA)
        if(classes_name == "sports ball"):
            # print("{} is in the picture with confidence: location: x={}, y={}".format(classes_name, x, y))
            print("{} location: x={}, y={}".format(classes_name, x, y))
            if com:
                com.write("x={}, y={}\n".format(x,y).encode('utf-8'))
    #    cv2.imwrite("demo.jpg", image)
    return image

def get_display_res():
    if os.path.exists("/usr/bin/get_hdmi_res") == False:
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])


def is_usb_camera(device):
    try:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

def find_first_usb_camera():
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

def capture_frames(cap, frame_queue, stop_event, desired_fps):
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                frame_queue.put(frame)
                sleep(1 / desired_fps)  # Control the capture rate
            else:
                print("Failed to get image from usb camera")
                stop_event.set()
    finally:
        if cap.isOpened():
            cap.release()
        print("camera resource recovered!")

def process_frame(frame, models, output_tensors, fcos_postprocess_info, disp_w, disp_h):
    h, w = models[0].inputs[0].properties.shape[2], models[0].inputs[0].properties.shape[3]
    des_dim = (w, h)
    resized_data = cv2.resize(frame, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)

    outputs = models[0].forward(nv12_data)

    strides = [8, 16, 32, 64, 128]
    for i in range(len(strides)):
        if output_tensors[i].properties.quantiType == 0:
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(outputs[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(outputs[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)

        libpostprocess.FcosdoProcess(output_tensors[i], output_tensors[i + 5], output_tensors[i + 10], ctypes.pointer(fcos_postprocess_info), i)

    result_str = get_Postprocess_result(ctypes.pointer(fcos_postprocess_info))
    result_str = result_str.decode('utf-8')

    data = json.loads(result_str[14:])
    if frame.shape[0] != disp_h or frame.shape[1] != disp_w:
        frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    box_bgr = draw_bboxs(frame, data, fcos_postprocess_info.width, fcos_postprocess_info.height, disp_w, disp_h)
    return box_bgr

def process_frames(frame_queue, stop_event, video_write_queue, desired_fps):
    models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')
    print_properties(models[0].inputs[0].properties)
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)

    disp = srcampy.Display()
    disp_w, disp_h = get_display_res()
    disp.display(0, disp_w, disp_h)

    fcos_postprocess_info = FcosPostProcessInfo_t()
    fcos_postprocess_info.height = 512
    fcos_postprocess_info.width = 512
    fcos_postprocess_info.ori_height = disp_h
    fcos_postprocess_info.ori_width = disp_w
    fcos_postprocess_info.score_threshold = 0.3
    fcos_postprocess_info.nms_threshold = 0.4
    fcos_postprocess_info.nms_top_k = 5
    fcos_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
        if len(models[0].outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
        else:
            output_tensors[i].properties.quantiType = 2
            scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])
            output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for j in range(len(models[0].outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
            output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]

    start_time = time()
    image_counter = 0
    frame_skip = int(30 / desired_fps)  # Adjust frame skipping based on desired FPS
    frame_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        while not stop_event.is_set():
            if frame_queue.empty():
                continue
            frame = frame_queue.get()
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            future = executor.submit(process_frame, frame, models, output_tensors, fcos_postprocess_info, disp_w, disp_h)
            box_bgr = future.result()
            video_write_queue.put(box_bgr)

            finish_time = time()
            image_counter += 1
            if finish_time - start_time > 10:
                print(start_time, finish_time, image_counter)
                print("FPS: {:.2f}".format(image_counter / (finish_time - start_time)))
                start_time = finish_time
                image_counter = 0

    cap.release()
    cv2.destroyAllWindows()
    print("ok")

def serial_communication(com, stop_event):
    if not com:
        return
    while not stop_event.is_set():
        if com.in_waiting > 0:
            data = com.read(com.in_waiting)
            print(f"Received from serial: {data.decode('utf-8')}")
        # Add a small sleep to prevent busy-waiting
        sleep(0.1)

def show_preview(frame_queue, stop_event, models, output_tensors, fcos_postprocess_info, disp_w, disp_h):
    while not stop_event.is_set():
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        box_bgr = process_frame(frame, models, output_tensors, fcos_postprocess_info, disp_w, disp_h)
        cv2.imshow("Preview", box_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

class DroppingQueue(queue.Queue):
    def put(self, item, block=False, timeout=None):
        try:
            super().put(item, block=False)
        except queue.Full:
            # 丢弃最旧的一帧，腾出空间
            try:
                self.get_nowait()
            except queue.Empty:
                pass
            super().put(item, block=False)

def video_writer(video_write_queue, stop_event):
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (disp_w, disp_h))
    while not stop_event.is_set() or not video_write_queue.empty():
        try:
            frame = video_write_queue.get(timeout=1)
            out.write(frame)
        except queue.Empty:
            continue
    out.release()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) > 1:
        video_device = sys.argv[1]
    else:
        video_device = find_first_usb_camera()
    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)
    if(not cap.isOpened()):
        exit(-1)
    
    # keyboard_thread = threading.Thread(
    #     target=keyboard_listener,
    #     args=(stop_event,),
    #     daemon=True
    # )
    # keyboard_thread.start()
    
    
    print("Open usb camera successfully")

    output_video_save = 'CAMERA_OUT.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # frame_rate = 30
    # frame_size = (640,480)
    # out = cv2.VideoWriter(output_video_save,fourcc,frame_rate,frame_size)

    # 设置usb camera的输出图像��式为 MJPEG， 分辨率 640 x 480
    # 可以通过 v4l2-ctl -d /dev/video8 --list-formats-ext 命令查看摄像头支持的分辨率
    # 根据应用需求调整该采集图像的分辨率
    codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    #codec = cv2.VideoWriter_fourcc(*'mp4v')
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    videoc = cv2.VideoWriter_fourcc('X','V','I','D')
    # out = cv2.VideoWriter('video.avi',videoc,cv2.CAP_PROP_FPS,(1920,1080))

    frame_queue = DroppingQueue(maxsize=30)     #queue.Queue(maxsize=30)
    stop_event = threading.Event()
    serial_stop_event = threading.Event()
    video_write_queue = queue.Queue()

    desired_fps = 30  # Set your desired FPS here

    disp_w, disp_h = get_display_res()
    models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')
    fcos_postprocess_info = FcosPostProcessInfo_t()
    fcos_postprocess_info.height = 512
    fcos_postprocess_info.width = 512
    fcos_postprocess_info.ori_height = disp_h
    fcos_postprocess_info.ori_width = disp_w
    fcos_postprocess_info.score_threshold = 0.3
    fcos_postprocess_info.nms_threshold = 0.4
    fcos_postprocess_info.nms_top_k = 5
    fcos_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
        if len(models[0].outputs[i].properties.scale_data) == 0:
            output_tensors[i].properties.quantiType = 0
        else:
            output_tensors[i].properties.quantiType = 2
            scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])
            output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for j in range(len(models[0].outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
            output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]

    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, stop_event, desired_fps))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, stop_event, video_write_queue, desired_fps))
    preview_thread = threading.Thread(target=show_preview, args=(frame_queue, stop_event, models, output_tensors, fcos_postprocess_info, disp_w, disp_h))
    serial_thread = None
    # calculation_thread = threading.Thread(target=test3.calculation_thread, args=(test3.shared_coordinates, test3.serial_port))
    if com:
        serial_thread = threading.Thread(target=serial_communication, args=(com, serial_stop_event))
        serial_thread.start()
    video_write_thread = threading.Thread(
        target=video_writer,
        args=(video_write_queue, stop_event)
    )
    video_write_thread.start()
    capture_thread.start()
    process_thread.start()
    # preview_thread.start()


    try:

        capture_thread.join()
        process_thread.join()
        # preview_thread.join()
        # video_write_thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        serial_stop_event.set()
        capture_thread.join()
        process_thread.join()
        if preview_thread:
            preview_thread.join()
        # if video_write_thread:
        #     video_write_thread.join()
        if serial_thread:
            serial_thread.join()
    stop_event.set()
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    print("ok")
