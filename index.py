

from ultralytics.models.yolo.model import YOLO


if __name__ == "__main__":

    model = YOLO("yolo11n.pt")

    model.track("0",show=True)
    #model.track("test.jpg",save=True)