# test_switch_output.py（修改版）
import cv2
import time

# 配置参数
TEST_DURATION = 100
OUTPUT_FILE = "switch_test_output.avi"
CAMERA_DEVICE = "/dev/video0"

# 初始化摄像头
cap = cv2.VideoCapture(CAMERA_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 配置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30.0, (1280, 720))

# 状态变量
current_cam = "left"
start_time = time.time()
last_switch = start_time

print(f"▶️ 开始测试，视频将保存至: {OUTPUT_FILE}")

while (time.time() - start_time) < TEST_DURATION:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 分割双目画面
    left_frame = frame[:, :1280]
    right_frame = frame[:, 1280:]
    
    # 3秒切换逻辑
    if time.time() - last_switch >= 30:
        current_cam = "right" if current_cam == "left" else "left"
        last_switch = time.time()
        print(f"🔄 切换到 {current_cam} 摄像头")
    
    # 选择当前画面并写入视频（移除显示部分）
    selected_frame = left_frame if current_cam == "left" else right_frame
    out.write(selected_frame)

# 释放资源
cap.release()
out.release()

print(f"✅ 测试完成！生成文件: {OUTPUT_FILE}")
print("使用以下命令下载查看：")
print(f"   scp user@rdk-ip:{OUTPUT_FILE} ./")