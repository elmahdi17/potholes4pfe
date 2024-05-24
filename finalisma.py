import cv2 as cv
import time
import geocoder
import os
import numpy as np
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the beep sound
beep_sound = pygame.mixer.Sound("beep.mp3")

# Function to play beep sound twice with a delay
def play_beep_twice():
    beep_sound.play()
    pygame.time.delay(300)  # Delay for 0.3 seconds
    beep_sound.play()

# Function to start the dashboard video
def start_engine():
    global play_video, playing_intro, dashboard_paused
    play_video = True
    playing_intro = False
    dashboard_paused = False
    # Move to the beginning of the dashboard video
    dashboard_video.set(cv.CAP_PROP_POS_FRAMES, 0)

# Function to toggle the video feed
def toggle_feed():
    global show_feed
    show_feed = not show_feed

# Reading label names from obj.names file
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Importing model weights and config file, defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # Use DirectShow backend
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Load the background videos
intro_video = cv.VideoCapture("introduct.mp4")
dashboard_video = cv.VideoCapture("dashboard.mp4")
if not intro_video.isOpened() or not dashboard_video.isOpened():
    print("Error: Could not open video files.")
    exit()

play_video = False
playing_intro = True
dashboard_paused = True

# Define parameters for result saving and getting coordinates
g = geocoder.ip('me')
result_path = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
pothole_index = 0
last_detection_time = 0

# Set video dimensions
dashboard_width = int(original_width * 1.6)  # Increase width by 20%
dashboard_height = int(original_height * 1.3)  # Increase height by 5%

# Initialize tkinter
root = tk.Tk()
root.title("DASHBOARD")

# Reduce window width by 30%
window_width = int((dashboard_width + original_width) * 0.7)
root.geometry(f"{window_width}x{dashboard_height}")

# Variable to control the video feed display
show_feed = False

# Create a button to toggle the video feed
feed_button = Button(root, text="Pothole Detection", command=toggle_feed)
feed_button.pack()

# Create a button to start the dashboard video
start_button = Button(root, text="Start Engine", command=start_engine)
start_button.pack()

# Create a label to display the combined frame
label = tk.Label(root)
label.pack()

# Variable to store the last frame of the dashboard video
last_dashboard_frame = None

# Function to update the frames
def update_frame():
    global frame_counter, starting_time, pothole_index, last_detection_time, play_video, playing_intro, last_dashboard_frame, dashboard_paused

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        root.after(10, update_frame)  # Try again after a short delay
        return

    frame_counter += 1

    # Read the next frame from the introductory video if still playing
    if playing_intro:
        intro_ret, intro_frame = intro_video.read()
        if not intro_ret:
            playing_intro = False
            play_video = True
    else:
        intro_frame = None

    # Read the next frame from the dashboard video if play_video is True
    if play_video and not playing_intro and not dashboard_paused:
        dashboard_ret, dashboard_frame = dashboard_video.read()
        if dashboard_ret:
            last_dashboard_frame = dashboard_frame
        else:
            play_video = False  # Stop playing when the video ends

    # Use the last frame if the video has ended
    if last_dashboard_frame is not None:
        dashboard_frame_resized = cv.resize(last_dashboard_frame, (window_width, dashboard_height))
    else:
        dashboard_frame_resized = np.zeros((dashboard_height, window_width, 3), dtype=np.uint8)

    # Use the introductory video frame if still playing
    if intro_frame is not None:
        intro_frame_resized = cv.resize(intro_frame, (window_width, dashboard_height))
        combined_frame = intro_frame_resized
    else:
        combined_frame = dashboard_frame_resized

    if show_feed:
        # Resize the video frame to fit within the dashboard video window
        aspect_ratio = original_width / original_height
        feed_height = int(min(original_height // 3, dashboard_height) / 1.6)  # Decrease height by 1.6 times
        feed_width = int(feed_height * aspect_ratio)

        # Make the feed 1.7 times bigger
        feed_height = int(feed_height * 1.7)
        feed_width = int(feed_width * 1.7)

        # Calculate the position to center the video feed within the dashboard video window
        x_offset = (window_width - feed_width) // 2
        y_offset = int((dashboard_height - feed_height) * 0.68)  # Move feed up by 20% relative to dashboard height

        # Resize the video frame
        frame_resized = cv.resize(frame, (feed_width, feed_height))

        # Overlay the video frame onto the dashboard video frame
        combined_frame[y_offset:y_offset + feed_height, x_offset:x_offset + feed_width] = frame_resized

        # Analyzing the stream with detection model within the resized frame
        classes, scores, boxes = model1.detect(frame_resized, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            label_text = class_name[classid] if classid < len(class_name) else "Unknown"
            x, y, w, h = box
            recarea = w * h
            area = feed_width * feed_height

            # Drawing yellow detection boxes on combined frame for detected potholes
            if len(scores) != 0 and scores[0] >= 0.7:
                if (recarea / area) <= 0.1 and y < feed_height:
                    # Adjust coordinates for combined frame
                    x_combined = x_offset + x
                    y_combined = y_offset + y
                    w_combined = w
                    h_combined = h

                    cv.rectangle(combined_frame, (x_combined, y_combined), (x_combined + w_combined, y_combined + h_combined), (0, 255, 255), 2)  # Yellow rectangle

                    current_time = time.time()
                    if pothole_index == 0 or (current_time - last_detection_time) >= 2:
                        cv.imwrite(os.path.join(result_path, f'pothole{pothole_index}.jpg'), frame)
                        with open(os.path.join(result_path, f'pothole{pothole_index}.txt'), 'w') as f:
                            f.write(str(g.latlng))
                        last_detection_time = current_time
                        pothole_index += 1
                        
                        # Check if the rectangle is big (close pothole)
                        if (w_combined * h_combined) >= (feed_width * feed_height) * 0.05:
                            # Play beep sound twice with delay
                            play_beep_twice()
                        else:
                            # Play beep sound once
                            beep_sound.play()

    # Convert the combined frame to a format suitable for tkinter
    combined_frame_rgb = cv.cvtColor(combined_frame, cv.COLOR_BGR2RGB)
    combined_image = Image.fromarray(combined_frame_rgb)
    combined_photo = ImageTk.PhotoImage(image=combined_image)

    # Update the label with the new frame
    label.configure(image=combined_photo)
    label.image = combined_photo

    # Schedule the next frame update
    root.after(10, update_frame)

# Start the frame update loop
update_frame()

# Function to pause the dashboard video
def pause_dashboard():
    global dashboard_paused
    dashboard_paused = True

# Function to resume the dashboard video
def resume_dashboard():
    global dashboard_paused
    dashboard_paused = False

# Create buttons to control dashboard video playback
pause_button = Button(root, text="Pause Dashboard", command=pause_dashboard)
pause_button.pack()

resume_button = Button(root, text="Resume Dashboard", command=resume_dashboard)
resume_button.pack()

# Start the tkinter main loop
root.mainloop()

# Release resources when the application is closed
cap.release()
intro_video.release()
dashboard_video.release()
cv.destroyAllWindows()
