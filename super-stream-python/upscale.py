import cv2
from cv2 import dnn_superres
import time
# attempt using open cv2

TEST_IMAGE_PATH = "./inputs/" + "frame0.png"
ESPCN_PATH = "./models/ESPCN/" + "ESPCN_x2.pb"
FSRCNN_PATH = "./models/FSRCNN/" + "FSRCNN-small_x2.pb"

MODEL_TYPE = "fsrcnn"  # "espcn" or "fsrcnn"
SCALE_FACTOR = 2


def upscale(sr, image):
    result = sr.upsample(image)
    # cv2.imwrite('./outputs'+'espcn_x2.png', result)
    return result


def resize(image, factor):
    return cv2.resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)))


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()  # returns frameGrabbed, frameDecoded
    frame_count = 0
    start = time.time()

    sr = dnn_superres.DnnSuperResImpl_create()
    # image = cv2.imread(TEST_IMAGE_PATH)
    path = ESPCN_PATH if MODEL_TYPE == 'espcn' else FSRCNN_PATH
    # sr.readModel(path)
    # sr.setModel("espcn", 2)
    # path = FSRCNN_PATH
    sr.readModel(path)
    sr.setModel(MODEL_TYPE, SCALE_FACTOR)

    while (cap.isOpened()):
        _, frame = cap.read()

        predicted = upscale(sr, frame)
        # refactor = resize(predicted, 0.5)
        cv2.imshow('predicted', predicted)

        # cv2.imshow('actual', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
        if time.time() - start > 1:
            start = time.time()
            print("FPS: "+str(frame_count))
            frame_count = 0
    cap.release()
    cv2.destroyAllWindows()
