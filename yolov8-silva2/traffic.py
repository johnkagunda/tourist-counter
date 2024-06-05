import random
from flask import Flask, render_template_string
from ultralytics import YOLO
import time

app = Flask(__name__)

# Function to generate HTML content
def generate_html(traffic_light_status):
    # HTML content template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Traffic Light</title>
        <style>
            .traffic-light {{
                width: 100px;
                height: 300px;
                border: 2px solid black;
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: space-around;
            }}
            .light {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
            }}
            .red {{
                background-color: red;
            }}
            .green {{
                background-color: green;
            }}
        </style>
    </head>
    <body>
        <div class="traffic-light">
            <div class="light {'red' if traffic_light_status else 'green'}"></div>
        </div>
    </body>
    </html>
    """
    return html_content

# Example usage
if __name__ == "__main__":
    # Load a pretrained YOLOv8n model
    model = YOLO("weights/yolov8n.pt", "v8")

    # Video frame dimensions
    frame_wid = 640
    frame_hyt = 480

    # Initialize variables for traffic light control
    red_light_duration = 0
    start_time = time.time()
    alert_triggered = False

    @app.route("/")
    def home():
        global red_light_duration, alert_triggered

        # Capture frame-by-frame
        # Replace this with your code to capture frames from the video
        # For simplicity, we're just using a placeholder here
        traffic_light_status = random.choice([True, False])

        # Check if red light duration exceeds 5 minutes (300 seconds)
        if traffic_light_status:
            red_light_duration += 1
            if red_light_duration >= 300 and not alert_triggered:
                print("Alert: Red light has been on for more than 5 minutes!")
                alert_triggered = True  # Set flag to prevent repeated alerts
        else:
            red_light_duration = 0  # Reset red light duration if not red

        # Generate HTML content based on traffic light status
        html_content = generate_html(traffic_light_status)

        return html_content

    app.run(debug=True)
