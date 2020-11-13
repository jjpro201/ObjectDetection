import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

#loading image
#cap = cv2.VideoCapture('test.mp4')
img = cv2.imread("desk.jpg")

while True:
    _, img = img.read()
    height, width, _ = img.shape
    img = cv2.resize(img, None, fx=0.6, fy=0.6)

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    class_ids = []
    confidences = []
    boxes = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #number_objects_detected = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()