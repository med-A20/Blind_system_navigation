import argparse
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
import cv2
import numpy as np

from PIL import Image

def process_image(image_path, output_video_path=None):
    # Load the image processing model and the object detection model
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = model.to("cuda") if torch.cuda.is_available() else model

    # Initialize the input source (image or video)
    if image_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        input_source = cv2.imread(image_path)
    else:
        input_source = cv2.VideoCapture(image_path)

    # Initialize the output video writer if provided
    if output_video_path:
        frame_width = int(input_source.get(3))
        frame_height = int(input_source.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    while True:
        if isinstance(input_source, np.ndarray):
            frame = input_source
            ret = True
        else:
            ret, frame = input_source.read()

        if not ret:
            break

        # Resize the frame to reduce resolution (optional)
        frame = cv2.resize(frame, (1024, 480))


        # draw path ROI
        # Define the dimensions and positions of the three rectangles
        width, height = len(frame[0]), len(frame)
        rect_widths = [int(width * 0.4) // 5, 3 * int((width * 0.4) // 5), int(width * 0.4) // 5]
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
                thickness=1,  # Fill the rectangle
            )
        # end ROI

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Process the image using the image processor
        inputs = image_processor(images=pil_image, return_tensors="pt")
        inputs = inputs.to("cuda") if torch.cuda.is_available() else inputs

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_str = model.config.id2label[label.item()]
            confidence = round(score.item(), 3)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_str}: {confidence}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        # Write frame to the output video if provided
        if output_video_path:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if isinstance(input_source, cv2.VideoCapture):
        input_source.release()

    if output_video_path:
        out.release()
        print(f"Processed video saved to {output_video_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with DETR")
    parser.add_argument("input_path", help="Path to input image or video file.")
    parser.add_argument("-o", "--output_video", help="Path to save the output video (optional).")
    args = parser.parse_args()

    process_image(args.input_path, args.output_video)
