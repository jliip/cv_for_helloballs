#!/usr/bin/env python3

import sys
import signal
import os
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import colorsys
import time
import serial
import ctypes
import json
import math
import serial.tools.list_ports
# change
import zmq
import numpy as np

context = None
socket = None
PORT = 55555
#change

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



# com = serial.Serial(
#     port='/dev/ttyUSB0',
#     baudrate=115200,
#     bytesize=serial.EIGHTBITS,
#     parity=serial.PARITY_NONE,
#     timeout=1 
# )
# flag = com.is_open
# if(flag):
#     print("open esp32 sucessfully")
# else:
#     print("open esp32 failed")
#    com.write(str(value).encode())
#    print(f"Sent to serial: {value}")

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

class hbDNNTesorProperties_t(ctypes.Structure):
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
        ("properties",hbDNNTesorProperties_t)
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
    coor[0] = max(min(2560, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(720, coor[1]), 2)
    coor[2] = max(min(2560, coor[2]), 0)
    coor[3] = max(min(720, coor[3]), 0)
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
    scale_x = target_w /1280 #ori_w
    scale_y = target_h /720 #ori_h

    for i, result in enumerate(bboxes):
        bbox = result['bbox']  # 矩形框位置信息
        score = result['score']  # 得分
        id = int(result['id'])  # id
        name = result['name']  # 类别名称
        # if name != 'sports ball':
        #     continue
			
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
        # if(classes_name == "person"):
        #     print("{} is in the picture with confidence: location: x={}, y={}".format(classes_name,x,y))
        #     com.write("x={}, y={}\n".format(x,y).encode('utf-8'))
        if(classes_name == "sports ball"):
            print("{} is in the picture with confidence: location: x={}, y={}".format(classes_name, x, y))
            #print("coor[0]={},coor[1]={},coor[2]={},coor[3]={}".format(coor[0],coor[1],coor[2],coor[3]))
            #com.write("{} is in the picture with confidence: location: x={}, y={}\n".format(classes_name, x, y))
            #com.write("x={}, y={}\n".format(x,y).encode('utf-8')) debug
            # com_input = com.read(10)
            # if com_input:
            #   print(com_input)
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
    video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video0')]
    for dev in video_devices:
        if is_usb_camera(dev):
            return dev
    return None

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

# def frame_generator():
#     global socket, context
#     while True:
#         try:
#             data = socket.recv(flags=zmq.NOBLOCK)
#             # 解码图像数据
#             nparr = np.frombuffer(data, np.uint8)
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             if frame is not None and frame.size > 0:
#                 yield frame
#             else:
#                 print("receive but decode failed")
#                 time.sleep(0.1)
#         except zmq.Again:
#             #print("wait for frame input")
#             time.sleep(0.1)
#             continue
#         except Exception as e:
#             print(f"🚨 error: {str(e)}")
#             try:
#                 socket.disconnect(f"tcp://0.0.0.0:{PORT}")
#                 socket.connect(f"tcp://0.0.0.0:{PORT}")
#                 print("finished reconnect")
#             except Exception as reconnect_error:
#                 print(f"reconnect failed: {str(reconnect_error)}")
#                 if socket:
#                     socket.close()
#                 if context:
#                     context.term()
#                 context = zmq.Context()
#                 socket = context.socket(zmq.SUB)
#                 socket.setsockopt(zmq.CONFLATE, 1)
#                 socket.connect(f"tcp://0.0.0.0:{PORT}")
#                 socket.setsockopt(zmq.SUBSCRIBE, b'')
#             time.sleep(0.5) 


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    # 打印输出 tensor 的属性
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)

    # this the orginal code for finding camera test1
    
    if len(sys.argv) > 1:
         video_device = sys.argv[1]
    else:
        video_device = find_first_usb_camera()
    if video_device is None:
        print("No USB camera found.")
        sys.exit(-1)

    print(f"Opening video device: {video_device}")
    cap = cv2.VideoCapture(video_device)
    cap.set(3,640)
    cap.set(4,472)
    if(not cap.isOpened()):
        exit(-1)
    
    
    print("Open usb camera successfully")
    frame_num = 0
    cam_mode = 0
    cam_previus = 0
    # while True:
    #     switch_frame_interval = math.floor(frame_num / 60)
    #     if(switch_frame_interval %2 == 0):
    #         cam_mode = 0
    #     else:
    #         cam_mode = 1
    #     if(cam_previus != cam_mode):
    #         if(cam_mode):
    #             cap.set(3,640)
    #             cap.set(4,472)
    #         else:
    #             cap.set(3,640)
    #             cap.set(4,480)
    #     cam_previus = cam_mode
    #     (grabbed, frame) = cap.read()
    #     if not grabbed:
    #         continue
    #     cv2.imshow("camer raw image",frame)
    #     # cv2.waitKey(0)
    #     frame_num += 1
    #     if cv2.waitKey(1) & 0xff == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

        

    # output_video_save = 'CAMERA_OUT.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # frame_rate = 30
    # frame_size = (640,480)
    # out = cv2.VideoWriter(output_video_save,fourcc,frame_rate,frame_size)

    # 设置usb camera的输出图像格式为 MJPEG， 分辨率 640 x 480
    # 可以通过 v4l2-ctl -d /dev/video8 --list-formats-ext 命令查看摄像头支持的分辨率
    # 根据应用需求调整该采集图像的分辨率
    #codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    #codec = cv2.VideoWriter_fourcc(*'mp4v')
    #cap.set(cv2.CAP_PROP_FOURCC, codec)
    
    # test1
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # videoc = cv2.VideoWriter_fourcc('X','V','I','D')
    # test1
    
    #out = cv2.VideoWriter('video.avi',videoc,cv2.CAP_PROP_FPS,(2560,720))
    # Get HDMI display object
    # disp = srcampy.Display()
    # For the meaning of parameters, please refer to the relevant documents of HDMI display
    disp_w, disp_h = get_display_res()
    # disp.display(0, disp_w, disp_h)

    # 获取结构体信息
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
        #print(output_tensors[i].properties.tensorLayout)
        if (len( models[0].outputs[i].properties.scale_data) == 0):
            output_tensors[i].properties.quantiType = 0
        else:
            output_tensors[i].properties.quantiType = 2
            scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])
            output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        for j in range(len(models[0].outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
            output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]

    start_time = time.time()
    image_counter = 0
    frame_skip = 2
    frame_count = 0
    while True:
        #_ ,frame = cap.read() test1
        # frame = next(frame_gen)
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to get image from usb camera")
            continue
        frame_count += 1
        
        # if frame is None or frame.size == 0 or frame.shape[0] < 100:
        #         print("received useless frame")
        #         time.sleep(0.1)
        #         continue
        
        # test1
        if frame_count % frame_skip !=0:
            continue
        # test3
        switch_frame_interval = math.floor(frame_num / 60)
        if(switch_frame_interval %2 == 0):
            cam_mode = 0
        else:
            cam_mode = 1
        if(cam_previus != cam_mode):
            if(cam_mode):
                cap.set(3,640)
                cap.set(4,472)
            else:
                cap.set(3,640)
                cap.set(4,480)
        cam_previus = cam_mode
        # (grabbed, frame) = cap.read()
        # if not grabbed:
        #     continue
        #cv2.imshow("camer raw image",frame)
        frame_num += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        # test3
        
        # 把图片缩放到模型的输入尺寸
        # 获取算法模型的输入tensor 的尺寸
        h, w = models[0].inputs[0].properties.shape[2], models[0].inputs[0].properties.shape[3]
        des_dim = (w, h)
        resized_data = cv2.resize(frame,des_dim, interpolation=cv2.INTER_AREA) #des_dim

        nv12_data = bgr2nv12_opencv(resized_data)

        t0 = time.time()
        # Forward
        outputs = models[0].forward(nv12_data)
        t1 = time.time()
        # print("forward time is :", (t1 - t0))

        # Do post process
        strides = [8, 16, 32, 64, 128]
        for i in range(len(strides)):
            if (output_tensors[i].properties.quantiType == 0):
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
        t2 = time.time()
        # print("FcosdoProcess time is :", (t2 - t1))
        # print(result_str)

        # draw result
        # 解析JSON字符串
        data = json.loads(result_str[14:])

        if frame.shape[0]!=disp_h or frame.shape[1]!=disp_w:
            frame = cv2.resize(frame, (disp_w,disp_h), interpolation=cv2.INTER_AREA)
            # Choose position and text properties
        # tim = (image_counter)
        # time_text = f"{tim}"  # Format to two decimal places
        # Ensure it's a string and handle unexpected cases
        # position = (10, 30)  # Top-left corner of the video
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.7
        # font_color = (255, 255, 255)  # White color
        # font_thickness = 2
        # Put the timestamp on the frame
        # cv2.putText(frame, time_text, position, font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
        
        # Draw bboxs
        # box_bgr = draw_bboxs(frame, data)
        box_bgr = draw_bboxs(frame, data, fcos_postprocess_info.width, fcos_postprocess_info.height, disp_w, disp_h)
        #out.write(box_bgr)
        #cv2.imwrite("imf.jpg", box_bgr)
        # out.write(box_bgr)
        cv2.imshow("Detection Results", box_bgr)
        cv2.waitKey(30)
        # Convert to nv12 for HDMI display
        box_nv12 = bgr2nv12_opencv(box_bgr)
        #disp.set_img(box_nv12.tobytes())

        finish_time = time.time()
        
        
        image_counter += 1
        if finish_time - start_time >  10:
            print(start_time, finish_time, image_counter)
            print("FPS: {:.2f}".format(image_counter / (finish_time - start_time)))
            start_time = finish_time
            image_counter = 0
        # cap.release() # testfree
        # if out.isOpened():
        #     out.release()
        # disp.close()
cap.release()
# out.release() test1
cv2.destroyAllWindows()
print("ok")
