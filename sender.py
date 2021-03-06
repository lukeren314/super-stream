import cv2
import numpy as np
import socket
import struct
import math
import time

class FrameSegment(object):
    """ 
    Object to break down image frame segment
    if the size of image exceed maximum datagram size 
    """
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown
    def __init__(self, sock, port, addr="127.0.0.1"):
        self.s = sock
        self.port = port
        self.addr = addr

    def udp_frame(self, img):
        """ 
        Compress image and Break down
        into data segments 
        """
        # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        compress_img = cv2.imencode('.jpg', img)[1]
        dat = compress_img.tobytes()
        size = len(dat)
        print("packet size: "+str(size))
        count = math.ceil(size/(self.MAX_IMAGE_DGRAM))
        array_pos_start = 0
        while count:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            self.s.sendto(struct.pack("B", count) +
                dat[array_pos_start:array_pos_end], 
                (self.addr, self.port)
                )
            array_pos_start = array_pos_end
            count -= 1


def main():
    """ Top level main function """
    # Set up UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 12345

    fs = FrameSegment(s, port)
    start = time.time()
    frameCount = 0
    cap = cv2.VideoCapture(0)
    _, frame = cap.read() # returns frameGrabbed, frameDecoded

    while (cap.isOpened()):
        _, frame = cap.read()
        fs.udp_frame(frame)
        frameCount += 1
        if time.time() - start > 1:
            start = time.time()
            print("FPS: "+str(frameCount))
            frameCount = 0
    cap.release()
    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()