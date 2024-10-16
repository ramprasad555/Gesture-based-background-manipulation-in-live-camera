import torch
import numpy as np
import cv2 as cv
from torchvision import models
import torchvision.transforms as T

class HumanSegmenter:
    def __init__(self, alpha=0.8):
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.alpha = alpha

    def segment_human(self, img):
        input_tensor = self.transforms(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        return output_predictions.numpy()

    def colorize_segmentation(self, segmentation_mask: np.ndarray, img: np.ndarray, background_color=(0, 255, 0)) -> np.ndarray:
        colorized_mask = img.copy()
        background = np.zeros_like(img)
        background[:] = background_color

        # Set non-human regions to the specified background color
        colorized_mask[segmentation_mask != 15] = background[segmentation_mask != 15]

        return colorized_mask

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        segmentation_mask = self.segment_human(img)
        colorized_mask = self.colorize_segmentation(segmentation_mask, img)
        blended_image = cv.addWeighted(img, 1 - self.alpha, colorized_mask, self.alpha, 0)

        return blended_image

if __name__ == "__main__":
    # Create an instance of the HumanSegmenter
    segmenter = HumanSegmenter()

    # Open a video capture object
    cap = cv.VideoCapture(0)  # Use 0 for the default camera, change it if you have multiple cameras

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform segmentation and display the result
        result = segmenter.detect_and_draw(frame)

        cv.imshow("Segmented Image", result)

        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv.destroyAllWindows()
