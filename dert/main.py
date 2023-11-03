from time import sleep
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
import cv2
from PIL import Image
import pyttsx3


engine = pyttsx3.init()

# Load the image processing model and the object detection model
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
# model = model.to("cuda")  # Move the model to GPU if available

# Initialize the video capture from the default camera (usually camera index 0)
# cap = cv2.VideoCapture("test_2.mp4")
cap = cv2.imread("test-1.png")

frame_count = 0
skip_frames = 15  # Skip some frames for faster processing

while True:
    ret, frame = True, cap
    frame_count += 1

    if frame_count % skip_frames == 0:
        frame_count = 0

        # Resize the frame to reduce resolution (optional)
        frame = cv2.resize(frame, (1024, 640))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Process the image using the image processor
        inputs = image_processor(images=pil_image, return_tensors="pt")
        # inputs = inputs.to("cuda")  # Move the inputs to GPU if available

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        # Initialize variables to keep track of object positions
        object_in_middle = False
        object_on_left = False
        object_on_right = False
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_str = model.config.id2label[label.item()]
            confidence = round(score.item(), 3)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_str}: {confidence}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            # draw path ROI
            # Define the dimensions and positions of the three rectangles
            width, height = len(frame[0]), len(frame)
            rect_widths = [int(width * 0.5) // 5, 3 * int((width * 0.5) // 5), int(width * 0.5) // 5]
            rect_height = int(0.7 * height)
            y_position = int(0.2 * height)  # Center vertically at 20% of the image height

            # Calculate the x-positions of the three rectangles
            x_positions = [
                int((width - sum(rect_widths)) / 2),  # Center the first rectangle
                int((width - sum(rect_widths)) / 2) + rect_widths[0],  # Start of the second rectangle
                int((width - sum(rect_widths)) / 2) + rect_widths[0] + rect_widths[1],  # Start of the third rectangle
            ]

            # Define rectangle colors (BGR format)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
            # Draw the three rectangles on the image
            for i in range(3):
                cv2.rectangle(
                    frame,
                    (x_positions[i], y_position),
                    (x_positions[i] + rect_widths[i], y_position + rect_height),
                    colors[i],
                    thickness=1, 
                )
            # end ROI

            # detect the location of objects
            middle_rect = (
                (x_positions[1], y_position),  # Top-left corner
                (x_positions[1] + rect_widths[1], y_position + rect_height),  # Bottom-right corner
            )

            # Calculate the width of the object's bounding box
            object_width = box[2] - box[0]

            # print(object_width, rect_widths[1] / 2)


            # Check if the object's bounding box is within the middle rectangle
            if (
                    middle_rect[0][0] < box[0] < middle_rect[1][0] and
                    middle_rect[0][1] < box[1] < middle_rect[1][1] and
                    middle_rect[0][0] < box[2] < middle_rect[1][0] and
                    middle_rect[0][1] < box[3] < middle_rect[1][1] and
                    object_width > (rect_widths[1] / 2)
            ):
                object_in_middle = True
            elif box[2] < middle_rect[0][0]:
                object_on_left = True
            elif box[0] > middle_rect[1][0]:
                object_on_right = True

            # Determine recommendations based on object positions
            if object_in_middle:
                recommendation = "Stop! An "+ label_str +" is in front of you."
            elif object_on_left and object_on_right:
                recommendation = "Move forward cautiously. Objects on both sides."
            elif object_on_left:
                recommendation = "Move right. There's an "+ label_str +" on the left."
            elif object_on_right:
                recommendation = "Move left. There's an "+ label_str +" on the right."
            else:
                recommendation = "Safe to move forward."

            # Speak the recommendation
            engine.say(recommendation)
            print(recommendation)
            engine.runAndWait()
            # end
        sleep(3.3)
        cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
