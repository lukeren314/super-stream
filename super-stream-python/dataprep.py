import cv2
vidcap = cv2.VideoCapture('test\WIN_20201029_12_36_58_Pro.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("inputs/frame%d.png" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1
print('done')