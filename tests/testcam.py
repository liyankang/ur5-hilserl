import cv2
import time

# 相机已经能打开了！
cap = cv2.VideoCapture("/dev/video4")

time.sleep(0.5)

# video0
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# video2 # video4
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("相机打开失败！")
    exit()

print("✅ 相机已成功连接！开始显示...")

# # 设置resize后的尺寸
# resize_width = 320
# resize_height = 240

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 读取失败")
        break
    
    # Resize图像
    # resized_frame = cv2.resize(frame, (resize_width, resize_height))
    
    # frame = frame[191:408, 547:753]   #4
    frame = frame[425:653, 421:555] #2
    # frame = frame[120:500, 500:800]   #0
    # 显示原始图像和resize后的图像
    cv2.imshow("Original", frame)
    # cv2.imshow("Resized", resized_frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
