import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image path")
ap.add_argument("-m", "--model", required=True, help="model path")
ap.add_argument("-p", "--prototxt", required=True, help="prototxt path")
ap.add_argument("-c", "--confidence", help="confidence score", type=float, default=0.2)
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load model from disk
print("[INFO] loading model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load input image
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
image = cv2.resize(image, (300, 300))
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

# detecting objects
print("[INFO] Detecting Objects...")
net.setInput(blob)
detections = net.forward()

# loop over detections
for i in range(detections.shape[2]):
	confidence = detections[0, 0, i, 2]

	# select detection having more than threshold 
	if confidence > args["confidence"]:
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
		(startX, startY, endX, endY) = box.astype("int")

		# display prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Output", image)
cv2.waitKey(0)