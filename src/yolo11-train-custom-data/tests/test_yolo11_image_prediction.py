from ultralytics import YOLO
from utils import Logger

LOGGER = Logger()


class Test:
    """Test class for running YOLO inference on a single image."""

    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.image = "https://ultralytics.com/images/bus.jpg"

    def test(self):
        LOGGER.log("Running test")
        # Run batched inference on a list of images
        results = self.model([self.image])  # return a result objects
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk
        LOGGER.log("Test completed")
