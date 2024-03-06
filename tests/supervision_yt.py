from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO('yolov8m.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture()



