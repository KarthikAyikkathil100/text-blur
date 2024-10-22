import cv2
import numpy as np

# Load the Caffe model
prototxt_path = "weights/deploy.prototxt.txt"
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Function to blur faces and draw rectangles
def blur_faces_and_draw_rect(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.01:  # Confidence threshold
            print('conf -> ', confidence)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Blur the face
            face = frame[startY:endY, startX:endX]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[startY:endY, startX:endX] = blurred_face
            
            # Draw a blue rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)  # Blue rectangle

    return frame

# Process the video
video_path = "inputs/part_1.mp4"  # Change this to your video file path
output_path = "outputs/part_1.mp4"  # Output video file
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def draw_bounding_box(frame, vertices):
    # Convert normalized coordinates to pixel coordinates
    height, width, _ = frame.shape
    points = [(int(v['x'] * width), int(v['y'] * height)) for v in vertices]

    # Draw the bounding box
    points = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Blur faces and draw rectangles in the current frame
    
    out.write(processed_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_path)
