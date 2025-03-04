import cv2
import time
import zmq
import sys
import os
# 配置参数
PORT = 55555
CAMERA_DEVICE = "/dev/video0"
ENABLE_LOCAL_PREVIEW = False
SAVE_FRAME = False
def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头设备 {CAMERA_DEVICE}")
        sys.exit(1)
    print(f"✅ camera {CAMERA_DEVICE} open")

    # 配置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 初始化 ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    try:
        socket.bind(f"tcp://0.0.0.0:{PORT}")
        print(f"✅ ZMQ  {PORT}")
    except zmq.ZMQError as e:
        print(f"❌ ZMQ fail: {str(e)}")
        sys.exit(1)

    current_cam = "left"
    last_switch = time.time()

    try:
        print("=== start ===")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_DEVICE)
                time.sleep(1)
                continue

            # 分割画面
            left_frame = frame[:, :1280]
            right_frame = frame[:, 1280:]

            # 切换逻辑
            if time.time() - last_switch >= 10:
                current_cam = "right" if current_cam == "left" else "left"
                last_switch = time.time()
                print(f"🔄 switch to {current_cam} camera")

            # 发送数据
            selected_frame = left_frame if current_cam == "left" else right_frame
            _, img_data = cv2.imencode(".jpg", selected_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            socket.send(img_data.tobytes())
            
            if SAVE_FRAME:
                timestamp = int(time.time() * 1000)
                filename = os.path.join(f"frame_{timestamp}.jpg")
                with open(filename,"wb") as f:
                    f.write(img_data.tobytes())

    except KeyboardInterrupt:
        print("user break")
    finally:
        cap.release()
        socket.close()
        context.term()
        print("✅ release")

if __name__ == '__main__':
    main()