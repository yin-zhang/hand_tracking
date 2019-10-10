import numpy as np
import cv2
from hand_tracker import HandTracker

det = HandTracker('models/palm_detection_without_custom_op.tflite',
                  'models/hand_landmark_3d.tflite',
                  'data/anchors.csv',
                  box_shift=-0.5, box_enlarge=2.6)
in_bgr = cv2.imread('data/test_img1.jpg')
in_rgb = in_bgr[:,:,::-1]
list_keypoints, list_bbox = det(in_rgb)

out_img = np.copy(in_bgr)

# point size
ps = int(np.ceil(min(out_img.shape[0], out_img.shape[1]) / 256))

if list_keypoints is not None:
    for idx in range(len(list_keypoints)):
        keypoints = list_keypoints[idx]
        bbox = list_bbox[idx]
        for i in range(4):
            j = (i + 1) % 4
            p0 = (int(bbox[i,0]+0.5),int(bbox[i,1]+0.5))
            p1 = (int(bbox[j,0]+0.5),int(bbox[j,1]+0.5))
            cv2.line(out_img, p0, p1, (0,0,255), ps)
        for i in range(keypoints.shape[0]):
            p = (int(keypoints[i,0]+0.5),int(keypoints[i,1]+0.5))
            cv2.circle(out_img, p, ps, (255,0,0), ps)
    cv2.imwrite('out.jpg', out_img)
