import cv2 as cv
import time
import geocoder
import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import pygame
import threading
from scipy.signal import lti, StateSpace, lsim, TransferFunction as tf
from scipy.signal import lti, StateSpace, lsim, place_poles
from control import tf as ctl_tf, ss, parallel
from tkinter import Button
from PIL import Image, ImageTk
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import matplotlib.animation as animation

# Initialize pygame mixer
pygame.mixer.init()

# Load the beep sound
beep_sound = pygame.mixer.Sound("beep.mp3")

def play_beep_twice():
    beep_sound.play()
    pygame.time.delay(300)  # Delay for 0.3 seconds
    beep_sound.play()

# Global variable to store all road disturbances
all_road_disturbances = []

def calcul_controle_strenght(depth):
    # System parameters
    m = 250  # mass of the car body (kg)
    k = 16000  # suspension stiffness (N/m)
    c = 1000  # damping coefficient (N s/m)

    # PID Controller gains
    Kp = 200   # Proportional gain
    Ki = 10    # Integral gain
    Kd = 50    # Derivative gain
        
        # Function to simulate the road disturbance with potholes
    def road_disturbance(t, pothole):
        road_profile = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        pothole_time = pothole['time']
        pothole_width = pothole['width']
        road_profile += (depth * np.exp(-((t - pothole_time) ** 2) / (2 * pothole_width ** 2)))
        return road_profile
        
        # System dynamics
    def suspension_system(state, t, m, k, c, Kp, Ki, Kd, pothole):
        x, v, xi = state
        road = road_disturbance(t, pothole)
        error = road - x
        control_force = Kp * error + Ki * xi + Kd * (-v)
        dxdt = v
        dvdt = (control_force - c * v - k * x) / m
        dxidt = error
        return [dxdt, dvdt, dxidt]
        
        # Function for suspension simulation
    def suspension_simulation_code(pothole):
        x0, v0, xi0 = 0, 0, 0
        initial_state = [x0, v0, xi0]
        t = np.linspace(0, 10, 1000)
            
        def update_plot(frame):
            time = t[:frame+1]
            road_profile = road_disturbance(time, pothole)
            solution = odeint(suspension_system, initial_state, time, args=(m, k, c, Kp, Ki, Kd, pothole))
            x = solution[:, 0]
            v = solution[:, 1]
            plt.cla()
            plt.plot(time, road_profile, 'r-', label='Road Profile')
            plt.plot(time, v, label='Suspension Reaction')
            plt.xlabel('Time (s)')
            plt.ylabel('Disturbance / Reaction')
            plt.title('Potholes Detection')
            plt.legend()

        fig = plt.figure(figsize=(12, 8))
        ani = animation.FuncAnimation(fig, update_plot, frames=len(t), interval=10)
        plt.tight_layout()
        plt.show()

    pothole = {'time': 0.2, 'depth': -1, 'width': 0.08}
    suspension_simulation_code(pothole)
 
def start_engine():
    global play_video, playing_intro, dashboard_paused
    play_video = True
    playing_intro = False
    dashboard_paused = False
    dashboard_video.set(cv.CAP_PROP_POS_FRAMES, 0)
 
def toggle_feed():
    global show_feed
    show_feed = not show_feed
 
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
 
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)
 
cap = cv.VideoCapture('test.mp4')
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
 
original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
 
intro_video = cv.VideoCapture("introduct.mp4")
dashboard_video = cv.VideoCapture("dashboard.mp4")
if not intro_video.isOpened() or not dashboard_video.isOpened():
    print("Error: Could not open video files.")
    exit()
 
play_video = False
playing_intro = True
dashboard_paused = True
 
g = geocoder.ip('me')
result_path = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
pothole_index = 0
last_detection_time = 0
 
dashboard_width = int(original_width * 1.6)
dashboard_height = int(original_height * 1.3)
 
root = tk.Tk()
root.title("DASHBOARD")
window_width = int((dashboard_width + original_width) * 0.7)
root.geometry(f"{window_width}x{dashboard_height}")
 
show_feed = False
 
feed_button = Button(root, text="Pothole Detection", command=toggle_feed)
feed_button.pack()
 
start_button = Button(root, text="Start Engine", command=start_engine)
start_button.pack()
 
label = tk.Label(root)
label.pack()
 
last_dashboard_frame = None
 
def update_frame():
    global frame_counter, starting_time, pothole_index, last_detection_time, play_video, playing_intro, last_dashboard_frame, dashboard_paused
 
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        root.after(10, update_frame)
        return
 
    frame_counter += 1
 
    if playing_intro:
        intro_ret, intro_frame = intro_video.read()
        if not intro_ret:
            playing_intro = False
            play_video = True
    else:
        intro_frame = None
 
    if play_video and not playing_intro and not dashboard_paused:
        dashboard_ret, dashboard_frame = dashboard_video.read()
        if dashboard_ret:
            last_dashboard_frame = dashboard_frame
        else:
            play_video = False
 
    if last_dashboard_frame is not None:
        dashboard_frame_resized = cv.resize(last_dashboard_frame, (window_width, dashboard_height))
    else:
        dashboard_frame_resized = np.zeros((dashboard_height, window_width, 3), dtype=np.uint8)
 
    if intro_frame is not None:
        intro_frame_resized = cv.resize(intro_frame, (window_width, dashboard_height))
        combined_frame = intro_frame_resized
    else:
        combined_frame = dashboard_frame_resized
 
    if show_feed:
        aspect_ratio = original_width / original_height
        feed_height = int(min(original_height // 3, dashboard_height) / 1.6)
        feed_width = int(feed_height * aspect_ratio)
        feed_height = int(feed_height * 1.7)
        feed_width = int(feed_width * 1.7)
        x_offset = (window_width - feed_width) // 2
        y_offset = int((dashboard_height - feed_height) * 0.68)
        frame_resized = cv.resize(frame, (feed_width, feed_height))
        combined_frame[y_offset:y_offset + feed_height, x_offset:x_offset + feed_width] = frame_resized
 
        classes, scores, boxes = model1.detect(frame_resized, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            label_text = class_name[classid] if classid < len(class_name) else "Unknown"
            x, y, w, h = box
            recarea = w * h
            area = feed_width * feed_height
 
            if len(scores) != 0 and scores[0] >= 0.7:
                if (recarea / area) <= 0.1 and y < feed_height:
                    x_combined = x_offset + x
                    y_combined = y_offset + y
                    w_combined = w
                    h_combined = h
                    depth = - h/2
                   
                    cv.rectangle(combined_frame, (x_combined, y_combined), (x_combined + w_combined, y_combined + h_combined), (0, 255, 255), 2)
                    current_time = time.time()
                    if pothole_index == 0 or (current_time - last_detection_time) >= 2:
                        cv.imwrite(os.path.join(result_path, f'pothole{pothole_index}.jpg'), frame)
                        with open(os.path.join(result_path, f'pothole{pothole_index}.txt'), 'w') as f:
                            f.write(str(g.latlng))
                        last_detection_time = current_time
                        pothole_index += 1
                       
                        if (w_combined * h_combined) >= (feed_width * feed_height) * 0.05:
                            play_beep_twice()
                        else:
                            beep_sound.play()
                       
                        # Run control strength calculation in a separate thread
                        threading.Thread(target=calcul_controle_strenght(depth)).start()
 
    combined_frame_rgb = cv.cvtColor(combined_frame, cv.COLOR_BGR2RGB)
    combined_image = Image.fromarray(combined_frame_rgb)
    combined_photo = ImageTk.PhotoImage(image=combined_image)
    label.configure(image=combined_photo)
    label.image = combined_photo
    root.after(10, update_frame)
 
update_frame()
 
def pause_dashboard():
    global dashboard_paused
    dashboard_paused = True
 
def resume_dashboard():
    global dashboard_paused
    dashboard_paused = False
 
pause_button = Button(root, text="Pause Dashboard", command=pause_dashboard)
pause_button.pack()
 
resume_button = Button(root, text="Resume Dashboard", command=resume_dashboard)
resume_button.pack()
 
root.mainloop()
 
cap.release()
intro_video.release()
dashboard_video.release()
cv.destroyAllWindows()