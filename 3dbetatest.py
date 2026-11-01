import cv2
import numpy as np
import pyautogui
import math
import mediapipe as mp

# 禁用 PyAutoGUI 默认延迟与防故障保护以保证光标顺滑
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# 1. 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 获取主显示器分辨率
SCREEN_W, SCREEN_H = pyautogui.size()

# 2. 交互判定参数
CAM_W, CAM_H = 640, 480
FRAME_MARGIN = 100          # 边缘映射缓冲框大小（像素）
SMOOTHING = 5               # 平滑指数（越大越平滑，越小延迟越低）

prev_screen_x, prev_screen_y = SCREEN_W // 2, SCREEN_H // 2
is_dragging = False         # 拖拽状态标记
PINCH_THRESHOLD = 32        # 食指与大拇指捏合像素阈值

# 3. 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

WINDOW_NAME = "AI Gesture Mouse & Accessibility Guide"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

def draw_accessibility_guide(img):
    """绘制半透明无障碍功能与操作说明卡片"""
    guide_w, guide_h = 240, 210
    overlay = img.copy()
    
    # 绘制半透明圆角黑色背景
    cv2.rectangle(overlay, (15, 15), (15 + guide_w, 15 + guide_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (15, 15), (15 + guide_w, 15 + guide_h), (0, 200, 255), 1)

    # 绘制说明文字
    lines = [
        ("[ 无障碍操作指南 ]", (0, 220, 255)),
        ("1. 移动: 食指在框内平移", (255, 255, 255)),
        ("2. 单击: 食指与拇指捏合", (255, 255, 255)),
        ("3. 拖拽: 捏合保持并移动", (255, 255, 255)),
        ("4. 释放: 张开食指与拇指", (255, 255, 255)),
        ("-----------------------", (100, 100, 100)),
        ("退出: 点击右上角[X] 或 ESC", (0, 165, 255))
    ]
    
    y = 40
    for text, color in lines:
        cv2.putText(img, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 24

while cap.isOpened():
    # 监听原生窗口关闭按钮（右上角的 X 号）
    # 当用户点击 X 关闭窗口时，该属性将不再为正数
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

    ret, frame = cap.read()
    if not ret:
        break

    # 画面水平镜像翻转
    frame = cv2.flip(frame, 1)

    # 绘制映射操作有效边界框
    cv2.rectangle(
        frame,
        (FRAME_MARGIN, FRAME_MARGIN),
        (CAM_W - FRAME_MARGIN, CAM_H - FRAME_MARGIN),
        (0, 255, 255),
        2
    )

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_status = "Status: Tracking"
    status_color = (0, 255, 0)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 提取大拇指指尖 (4) 与 食指指尖 (8) 像素坐标
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            x1, y1 = int(index_tip.x * CAM_W), int(index_tip.y * CAM_H)
            x2, y2 = int(thumb_tip.x * CAM_W), int(thumb_tip.y * CAM_H)

            # 绘制指尖视觉反馈圆点
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)

            # 坐标线性插值映射到电脑屏幕
            target_x = np.interp(x1, (FRAME_MARGIN, CAM_W - FRAME_MARGIN), (0, SCREEN_W))
            target_y = np.interp(y1, (FRAME_MARGIN, CAM_H - FRAME_MARGIN), (0, SCREEN_H))

            # 一阶低通滤波平滑移动
            curr_screen_x = prev_screen_x + (target_x - prev_screen_x) / SMOOTHING
            curr_screen_y = prev_screen_y + (target_y - prev_screen_y) / SMOOTHING

            # 移动系统光标
            pyautogui.moveTo(curr_screen_x, curr_screen_y)
            prev_screen_x, prev_screen_y = curr_screen_x, curr_screen_y

            # 计算两指距离判定捏合动作
            distance = math.hypot(x2 - x1, y2 - y1)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            if distance < PINCH_THRESHOLD:
                # 捏合状态 -> 触发按下/拖拽
                cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), -1)
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                current_status = "Status: Mouse Dragging / Click"
                status_color = (0, 0, 255)
            else:
                # 松开状态 -> 释放按键
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                current_status = "Status: Hovering"
                status_color = (0, 255, 0)

    # 渲染无障碍说明书与状态栏
    draw_accessibility_guide(frame)
    cv2.putText(frame, current_status, (25, CAM_H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.imshow(WINDOW_NAME, frame)

    # 检查按键：按下键盘 ESC 键亦可退出
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

# 释放按键与系统硬件资源
if is_dragging:
    pyautogui.mouseUp()
cap.release()
cv2.destroyAllWindows()
