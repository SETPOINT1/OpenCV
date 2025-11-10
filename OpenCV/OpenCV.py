import cv2
import numpy as np
import os
from collections import deque
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

VIDEO_FOLDER = 'video'
VIDEO_FILENAME = '10349005-uhd_4096_2160_25fps.mp4'

LOWER_COLOR = np.array([0, 0, 200])
UPPER_COLOR = np.array([180, 50, 255])

RESIZE_FACTOR = 0.25
BLUR_KERNEL_SIZE = (11, 11)
MIN_RADIUS = 10

HISTORY_LEN = 200
GRAPH_WIDTH = 600
GRAPH_HEIGHT = 480

def find_object(hsv_frame, lower_bound, upper_bound, min_radius):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center, radius = None, 0
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > min_radius:
            center = (int(x), int(y))
            radius = int(r)
            
    return center, radius, mask

def create_matplotlib_graph(history, max_len, graph_width, graph_height, y_limit):
    fig, ax = plt.subplots(figsize=(graph_width / 100, graph_height / 100), dpi=100)
    ax.set_facecolor('#2c3e50')
    fig.patch.set_facecolor('#2c3e50')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Object Height (Y-Position)", color='white', fontsize=14)
    ax.set_xlabel("Time (Frames)", color='white')
    ax.set_ylabel("Y-Coordinate", color='white')
    ax.set_xlim(0, max_len)
    ax.set_ylim(y_limit, 0)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.plot(history, color='#3498db', linewidth=2, marker='o', markersize=3, markevery=[-1]) # เน้นจุดสุดท้าย

    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_image = np.asarray(buf)    
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    
    return graph_image

# --- Main Program ---
video_path = os.path.join(os.path.dirname(__file__), VIDEO_FOLDER, VIDEO_FILENAME)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open file path: {video_path}")
else:
    history = deque(maxlen=HISTORY_LEN)
    prev_time = 0
    ret, frame = cap.read()
    if ret:
        frame_height = int(frame.shape[0] * RESIZE_FACTOR)
    else:
        frame_height = 480
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End Video, Restart")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            history.clear()
            continue

        frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        center, radius, mask = find_object(hsv, LOWER_COLOR, UPPER_COLOR, MIN_RADIUS)

        if center is not None:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            history.append(center[1])
        else:
            if len(history) > 0:
                history.append(history[-1])

        graph = create_matplotlib_graph(history, HISTORY_LEN, GRAPH_WIDTH, GRAPH_HEIGHT, frame_height)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Tracking Result", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Height Graph (Matplotlib)", graph)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()