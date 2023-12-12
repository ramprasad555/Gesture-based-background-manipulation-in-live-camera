import torch
import numpy as np
import cv2 as cv
from torchvision import models
import torchvision.transforms as T
import matplotlib.cm as cm

class DeepLabV3Segmenter:
    def __init__(self, alpha=0.8):
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.alpha = alpha

    def segment_image(self, img: np.ndarray) -> np.ndarray:
        torch_img = self.transforms(img)
        with torch.no_grad():
            result = self.model(torch_img.unsqueeze(0))

        class_predictions = torch.argmax(result['out'][0], dim=0).cpu().numpy()
        output_image = self.colorize_segmentation(class_predictions)

        if not np.array_equal(output_image.shape[:2], img.shape[:2]):
            output_image = cv.resize(output_image, (img.shape[1], img.shape[0]))

        return output_image

    def colorize_segmentation(self, segmentation_mask: np.ndarray) -> np.ndarray:
        colorized_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
        for class_id in np.unique(segmentation_mask):
            if class_id == 0:  # Background
                continue
            color = self.get_color(class_id)
            colorized_mask[segmentation_mask == class_id] = color

        return colorized_mask

    def get_color(self, class_id: int) -> np.ndarray:
        return np.array(cm.get_cmap('tab20')(class_id / 20.0)[:-1]) * 255

    def detect_and_change_background(self, img: np.ndarray, background_color=(0, 255, 0)) -> np.ndarray:
        segmented_image = self.segment_image(img)

        # Create a mask for the person
        person_mask = np.all(segmented_image == [0, 0, 0], axis=-1)

        # Replace the background with the specified color
        img[person_mask] = background_color

        blended_image = cv.addWeighted(img, 1 - self.alpha, segmented_image, self.alpha, 0)

        return blended_image

def draw_balloons(frame, balloon_centers, balloon_radius=20, balloon_color=(255, 0, 0)):
    for center in balloon_centers:
        cv.circle(frame, center, balloon_radius, balloon_color, -1)

if __name__ == "__main__":
    # Create an instance of the DeepLabV3Segmenter
    segmenter = DeepLabV3Segmenter()

    # Open a video capture object
    cap = cv.VideoCapture(0)  # Use 0 for the default camera, change it if you have multiple cameras

    balloon_centers = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform segmentation without changing the background
        segmented_image = segmenter.segment_image(frame)

        # Draw balloons on the frame
        draw_balloons(frame, balloon_centers)

        # Display the result
        cv.imshow("Segmented Image", frame)

        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Generate new balloon positions
        balloon_centers = [(x, y - 5) for x, y in balloon_centers]

        # Add a new balloon at the top of the frame
        balloon_centers.append((np.random.randint(frame.shape[1]), frame.shape[0]))

    # Release the capture object and close all windows
    cap.release()
    cv.destroyAllWindows()
