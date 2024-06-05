from flask import Flask, render_template, Response, jsonify
import random
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
import sqlite3

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('tourist_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tourist_counts (timestamp TEXT, count INTEGER)''')
    conn.commit()
    conn.close()

def insert_count(count):
    conn = sqlite3.connect('tourist_data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO tourist_counts (timestamp, count) VALUES (datetime('now'), ?)''', (count,))
    conn.commit()
    conn.close()

def get_counts():
    conn = sqlite3.connect('tourist_data.db')
    c = conn.cursor()
    c.execute('''SELECT timestamp, count FROM tourist_counts ORDER BY timestamp DESC LIMIT 20''')
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

init_db()

# Opening the file in read mode
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Capture the video
cap = cv2.VideoCapture("inference/videos/bar.MP4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize centroid tracker
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

ct = CentroidTracker()

# Function to count people and draw bounding boxes
def count_and_draw_people(frame):
    global ct

    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()

    rects = []

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]

            if class_list[int(clsID)] == 'person':
                startX, startY, endX, endY = box.xyxy.numpy()[0]
                rects.append((startX, startY, endX, endY))
                cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

    objects = ct.update(rects)
    people_count = len(objects)

    cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    insert_count(people_count)
    return frame, people_count

# Generator function to stream video frames
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, people_count = count_and_draw_people(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tourist_count_data')
def tourist_count_data():
    data = get_counts()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/')
def index():
    # Fetch the latest people count from the database
    data = get_counts()
    initial_count = next(iter(data.values()), 0)  # Get the most recent count or 0 if no data
    return render_template('index.html', people_count=initial_count)
