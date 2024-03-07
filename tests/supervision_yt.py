'''
@Author: Youngjae


pip install supervision or pip install supervision[desktop]
pip install ultralytics
'''

from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO('yolov8m.pt')

# https://supervision.roboflow.com/latest/annotators/#boundingboxannotator
bounding_box_annotator = sv.BoundingBoxAnnotator()

label_annotator = sv.LabelAnnotator()

blur_annotator = sv.BlurAnnotator()

video_path = 'video_samples/milk-bottling-plant.mp4'
cap = cv2.VideoCapture(video_path)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        break

    results = model.predict(img)
    
    # https://supervision.roboflow.com/latest/detection/core/#detections
    detections = sv.Detections.from_ultralytics(results[0])
    
    # filtering out low confidence detections
    # detections = detections[detections.confidence > 0.5]
    
    # filtering out all classes except for the first class
    # detections = detections[detections.class_id == 0]
    
    label = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]
    
    annotated_frame = bounding_box_annotator.annotate(
        scene=img.copy(),
        detections=detections
    )
    
    # labels를 넣어야, 실제 label이 나옴
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=label
    )
    
    annotated_frame = blur_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    
    out.write(annotated_frame)
    cv2.imshow("Image", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()