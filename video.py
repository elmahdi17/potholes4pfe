#importing necessary libraries
import cv2 as cv
import time
import geocoder
import os
 
#reading label name from obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
 
#importing model weights and config file
#defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)
 
#defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture(".mp4")
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
 
result_path = "pothole_coordinates"
result_video_path = "resultat.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
result = cv.VideoWriter(result_video_path, fourcc, fps, (width, height))
 
# Path to your picture
pothole_img_path = "danger.png"
pothole_img = cv.imread(pothole_img_path, cv.IMREAD_UNCHANGED)  # Load the image with alpha channel
if pothole_img is None:
    raise FileNotFoundError("Unable to load the pothole image.")
 
#defining parameters for result saving and get coordinates
#defining initial values for some parameters in the script
g = geocoder.ip('me')
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0
 
#detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break
    #analysis the stream with detection model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height
        #drawing detection boxes on frame for detected potholes and saving coordinates txt and photo
        if len(scores) != 0 and scores[0] >= 0.7:
            if (recarea / area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label, (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
               
                # Resize the pothole image to fit within the bounding box without distorting the aspect ratio
                max_width = 100  # Adjust the maximum width as needed
                max_height = 50  # Adjust the maximum height as needed
                scale_factor = min(max_width / pothole_img.shape[1], max_height / pothole_img.shape[0])
                new_width = int(pothole_img.shape[1] * scale_factor)
                new_height = int(pothole_img.shape[0] * scale_factor)
                pothole_img_resized = cv.resize(pothole_img, (new_width, new_height))
 
                # Calculate the position to place the image above the text
                text_position = (x, y - new_height - 5)  # Adjust the vertical offset as needed
 
                # Overlay the pothole image on top of the frame with alpha blending
                for c in range(0, 3):
                    frame_alpha = pothole_img_resized[:, :, c] * (pothole_img_resized[:, :, 3] / 255.0) + frame[text_position[1]:text_position[1]+new_height, text_position[0]:text_position[0]+new_width, c] * (1.0 - pothole_img_resized[:, :, 3] / 255.0)
                    frame[text_position[1]:text_position[1]+new_height, text_position[0]:text_position[0]+new_width, c] = frame_alpha
 
                # Draw rectangles around the image and text
                cv.rectangle(frame, (text_position[0], text_position[1]), (text_position[0] + new_width, text_position[1] + new_height), (0, 255, 0), 1)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
               
                # Save coordinates and image
                cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                    f.write(str(g.latlng))
                i += 1
    #writing fps on frame
    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    #saving result
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
   
#end
cap.release()
result.release()
cv.destroyAllWindows()
 