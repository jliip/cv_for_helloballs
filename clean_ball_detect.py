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

import numpy as np


#change

# try:
#     com = serial.Serial(
#         port='/dev/ttyUSB0',
#         baudrate=115200,
#         bytesize=serial.EIGHTBITS,
#         parity=serial.PARITY_NONE,
#         timeout=1 
#     )
#     flag = com.is_open
#     if(flag):
#         print("open esp32 successfully")
#     else:
#         print("open esp32 failed")
# except serial.SerialException as e:
#     print(f"Failed to open serial port: {e}")
#     com = None


last_ball_position = None
ball_detection_history = []


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

    scale_x = target_w /1280 #ori_w
    scale_y = target_h /720 #ori_h

    for i, result in enumerate(bboxes):
        bbox = result['bbox']  # 矩形框位置信息
        score = result['score']  # 得分
        id = int(result['id'])  # id
        name = result['name']  # 类别名称

        coor = [round(i) for i in bbox]

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
            print("{} is in the picture with confidence: location: x={}, y={}".format(classes_name, x, y))
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

def preprocess_image(image):
    """增强图像质量以提高检测效果"""
    try:
        # 增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 适当锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_image, -1, kernel)
        
        return sharpened
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        return image  # 出错时返回原始图像


def detect_tennis_ball(image):
    # 转换到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 网球的颜色范围 - 荧光黄/绿色
    # 这个范围需要根据您的实际环境调整
    lower_tennis = np.array([20, 100, 100])  # 偏黄绿色
    upper_tennis = np.array([35, 255, 255])  # 到更绿的色调
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_tennis, upper_tennis)
    
    # 显示掩码用于调试（可选）
    cv2.imshow("Tennis Ball Mask", mask)
    
    # 形态学操作改善掩码
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tennis_ball_detections = []
    for contour in contours:
        # 计算面积
        area = cv2.contourArea(contour)
        
        # 只考虑合理大小的区域
        # if area < 100 or area > 15000:
        #     continue
            
        # 计算圆形度
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 网球应该是相当圆的
        if circularity > 0.65:  # 略微降低阈值，因为网球可能不是完美的圆
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比 - 网球应该近似正方形
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.8:  # 允许一定的变形
                tennis_ball_detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'score': circularity,
                    'id': 32,  # sports ball ID
                    'name': 'sports ball'
                })
    cv2.imshow("Tennis Ball Detection", image)
    return tennis_ball_detections

def verify_tennis_ball_texture(image, bbox):
    """验证边界框内的区域是否具有网球的纹理特征"""
    # 确保bbox中的值是整数
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # 确保坐标在图像范围内
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 提取区域
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    
    # 调整大小以标准化
    roi = cv2.resize(roi, (64, 64))
    
    # 转换为灰度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 计算梯度（Sobel算子）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅度
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 计算纹理特征 - 梯度的标准差和平均值
    mean_gradient = np.mean(gradient_magnitude)
    std_gradient = np.std(gradient_magnitude)
    
    # 网球的纹理特征：有明显的线条但不是极端的
    # 这些阈值需要根据实际情况调整
    if 10 < mean_gradient < 60 and 15 < std_gradient < 70:
        return True
    
    return False

# def distinguish_from_court(image, bbox):
#     """区分网球和网球场"""
#     # 确保bbox中的值是整数
#     x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
#     # 确保坐标在图像范围内
#     h, w = image.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
    
#     # 提取区域
#     roi = image[y1:y2, x1:x2]
#     if roi.size == 0:
#         return False
    
#     # 转换到HSV
#     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
#     # 计算色相和饱和度的平均值
#     h, s, v = cv2.split(hsv_roi)
#     mean_hue = np.mean(h)
#     mean_saturation = np.mean(s)
#     mean_value = np.mean(v)
    
#     # 网球比场地更亮、更鲜艳
#     if mean_value > 150 and mean_saturation > 100:
#         return True
    
#     return False



if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    # 打印输出 tensor 的属性
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)


    
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
    disp_w, disp_h = get_display_res()
    # disp.display(0, disp_w, disp_h)
    # 获取结构体信息
    fcos_postprocess_info = FcosPostProcessInfo_t()
    fcos_postprocess_info.height = 512
    fcos_postprocess_info.width = 512
    fcos_postprocess_info.ori_height = disp_h
    fcos_postprocess_info.ori_width = disp_w
    fcos_postprocess_info.score_threshold = 0.15
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

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to get image from usb camera")
            continue
        frame_count += 1
        

        

        if frame_count % frame_skip !=0:
            continue

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

        frame_num += 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        h, w = models[0].inputs[0].properties.shape[2], models[0].inputs[0].properties.shape[3]
        des_dim = (w, h)
        resized_data = cv2.resize(frame,des_dim, interpolation=cv2.INTER_AREA) #des_dim

        nv12_data = bgr2nv12_opencv(resized_data)

        t0 = time.time()

        outputs = models[0].forward(nv12_data)
        t1 = time.time()
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

        data = json.loads(result_str[14:])
        tennis_ball_detections = detect_tennis_ball(preprocess_image(frame))
        ball_detected = False
        for result in data[:]:  # 使用切片创建副本以便安全移除元素
            if result['name'] == 'sports ball':
                # 验证是否为网球
                if verify_tennis_ball_texture(frame, result['bbox']): #and distinguish_from_court(frame, result['bbox']):
                    ball_detected = True
                    # 更新检测历史
                    x = int((result['bbox'][0] + result['bbox'][2]) / 2)
                    y = int((result['bbox'][1] + result['bbox'][3]) / 2)
                    last_ball_position = (x, y)
                    ball_detection_history.append(last_ball_position)
                    if len(ball_detection_history) > 5:
                        ball_detection_history.pop(0)
                else:
                    # 不是网球，从结果中移除
                    data.remove(result)

        # 4. 如果没有检测到网球，使用自定义方法
        if not ball_detected:
            tennis_ball_detections = detect_tennis_ball(frame)  # 使用处理过的帧
            if tennis_ball_detections:
                # 验证检测结果
                verified_detections = []
                for detection in tennis_ball_detections:
                    if verify_tennis_ball_texture(frame, detection['bbox']): #and distinguish_from_court(frame, detection['bbox']):
                        verified_detections.append(detection)
                
                if verified_detections:
                    data.extend(verified_detections)
                    ball_detected = True
                    
                    # 更新位置历史
                    best_detection = max(verified_detections, key=lambda x: x['score'])
                    x = int((best_detection['bbox'][0] + best_detection['bbox'][2]) / 2)
                    y = int((best_detection['bbox'][1] + best_detection['bbox'][3]) / 2)
                    last_ball_position = (x, y)
                    ball_detection_history.append(last_ball_position)
                    if len(ball_detection_history) > 5:
                        ball_detection_history.pop(0)



        if frame.shape[0]!=disp_h or frame.shape[1]!=disp_w:
            frame = cv2.resize(frame, (disp_w,disp_h), interpolation=cv2.INTER_AREA)

        box_bgr = draw_bboxs(frame, data, fcos_postprocess_info.width, fcos_postprocess_info.height, disp_w, disp_h)
        cv2.imshow("Detection Results", box_bgr)
        cv2.waitKey(30)
        box_nv12 = bgr2nv12_opencv(box_bgr)


        finish_time = time.time()
        
        
        image_counter += 1
        if finish_time - start_time >  10:
            print(start_time, finish_time, image_counter)
            print("FPS: {:.2f}".format(image_counter / (finish_time - start_time)))
            start_time = finish_time
            image_counter = 0

cap.release()

cv2.destroyAllWindows()
print("ok")
