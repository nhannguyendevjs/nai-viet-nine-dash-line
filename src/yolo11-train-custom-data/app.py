from tests import test_yolo11_image_prediction
from utils import Logger

LOGGER = Logger()


def run():
    LOGGER.log("Program started")
    test = test_yolo11_image_prediction.Test()
    test.test()
    LOGGER.log("Program finished")


if __name__ == "__main__":
    run()
