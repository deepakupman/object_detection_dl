from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np 
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load model
print("[INFO] Loading Model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# start video stream
print("[INFO] Starting Video stream....")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over frames 
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # extract blob
    (h, w) = frame.shape[:2]
    frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            label_idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[label_idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[label_idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label_idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press 'q' key to quit
    if key == ord('q'):
        break
    # update fps counter
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()