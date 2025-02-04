import csv
import cv2
import numpy as np
import tensorflow as tf

class HandTracker():
    r"""
    Class to use Google's Mediapipe HandTracking pipeline from Python.
    Multiple hands are supported.
    Any any image size and aspect ratio are supported.
    Both 2d and 3d joints are supported.

    Args:
        palm_model: path to the palm_detection.tflite
        joint_model: path to the hand_landmark.tflite
        anchors_path: path to the csv containing SSD anchors
    Ourput:
        (21,3) or (21,2) array of hand joints (depending on
        whether joint_model is 2d or 3d)
    Examples::
        >>> det = HandTracker(path1, path2, path3)
        >>> input_img = np.random.randint(0,255, 256*256*3).reshape(256,256,3)
        >>> keypoints, bbox = det(input_img)
    """

    def __init__(self, palm_model, joint_model, anchors_path,
                 box_enlarge=2.6, box_shift=-0.5):
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge

        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_joint = tf.lite.Interpreter(joint_model)
        self.interp_joint.allocate_tensors()
        
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
        
        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']

        # 90° rotation matrix used to create the alignment trianlge        
        self.R90 = np.r_[[[0,1],[-1,0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
                        [128, 128],
                        [128,   0],
                        [  0, 128]
                    ])
        self._target_box = np.float32([
                        [  0,   0, 1],
                        [256,   0, 1],
                        [256, 256, 1],
                        [  0, 256, 1],
                    ])
    
    def _get_triangle(self, ctr0, kp0, kp2, side=1, yshift=0):
        """get a triangle used to calculate Affine transformation matrix"""

        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)

        dir_v_r = dir_v @ self.R90.T

        ctr = ctr0 - yshift * dir_v
        
        return np.float32([ctr, ctr + dir_v*side/2, ctr + dir_v_r*side/2])

    @staticmethod
    def _triangle_to_bbox(source):
        # plain old vector arithmetics
        bbox = np.c_[
            [source[2] - source[0] + source[1]],
            [source[1] + source[0] - source[2]],
            [3 * source[0] - source[1] - source[2]],
            [source[2] - source[1] + source[0]],
        ].reshape(-1,2)
        return bbox
    
    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')
    
    @staticmethod
    def _sim(box0, box1, do_iou=True):
        x0,y0,w0,h0 = box0
        x1,y1,w1,h1 = box1
        xmin = max(x0-w0/2,x1-w1/2)
        ymin = max(y0-h0/2,y1-h1/2)
        xmax = min(x0+w0/2,x1+w1/2)
        ymax = min(y0+h0/2,y1+h1/2)
        i = max(0, xmax - xmin) * max(0, ymax - ymin)
        area0 = w0 * h0
        area1 = w1 * h1
        if do_iou:
            u = area0 + area1 - i
        else:
            # modified jaccard
            u = area1
        return i / (u + 1e-6)
    
    def predict_joints(self, img_norm):
        self.interp_joint.set_tensor(
            self.in_idx_joint, img_norm.reshape(1,256,256,3))
        self.interp_joint.invoke()

        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        return joints.reshape(21,-1)

    def non_maximum_suppression(self, reg, anchors, scores,
                                weighted=True, sim_thresh=0.3, max_results=-1):

        sorted_idxs = scores.argsort()[::-1].tolist()

        abs_reg = np.copy(reg)
        
        # turn relative bbox/keyp into absolute bbox/keyp
        for idx in sorted_idxs:
            center = anchors[idx,:2] * 256
            for j in range(2):
                abs_reg[idx,j] = center[j] + abs_reg[idx,j]
                abs_reg[idx,(j+4)::2] = center[j] + abs_reg[idx,(j+4)::2]

        remain_idxs = sorted_idxs
        output_regs = abs_reg[0:0,:]

        while len(remain_idxs) > 0:
            # separate remain_idxs into candids and remain
            candids = []
            remains = []
            idx0 = remain_idxs[0]
            for idx in remain_idxs:
                sim = self._sim(abs_reg[idx0,:4], abs_reg[idx,:4], do_iou=False)
                if sim >= sim_thresh:
                    candids.append(idx)
                else:
                    remains.append(idx)

            # compute weighted bbox/keyp
            if not weighted:
                weighted_reg = abs_reg[idx0,:]
            else:
                weighted_reg = 0
                total_score = 0
                for idx in candids:
                    total_score += scores[idx]
                    weighted_reg += scores[idx] * abs_reg[idx,:]
                weighted_reg /= total_score

            # add a new instance
            output_regs = np.concatenate((output_regs, weighted_reg.reshape(1,-1)), axis=0)
            
            remain_idxs = remains

            if max_results > 0 and output_regs.shape[0] >= max_results:
                break

        return output_regs

    def detect_hand(self, img_norm):
        assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (256, 256, 3),\
        "img_norm shape must be (256, 256, 3)"

        # predict hand location and 7 initial landmarks
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0,:,0]
        out_scr = self._sigm(out_clf)

        # finding the best prediction
        detection_mask = out_scr > 0.7
        filtered_detect = out_reg[detection_mask]
        filtered_anchors = self.anchors[detection_mask]
        filtered_scores = out_scr[detection_mask]

        if filtered_detect.shape[0] == 0:
            return None, None

        # perform non-maximum suppression
        candidate_detect = self.non_maximum_suppression(filtered_detect, filtered_anchors, filtered_scores)

        list_sources = []
        list_keypoints = []

        for idx in range(candidate_detect.shape[0]):

            # bounding box center offsets, width and height
            cx,cy,w,h = candidate_detect[idx, :4]

            # 7 initial keypoints
            keypoints = candidate_detect[idx,4:].reshape(-1,2)
        
            # now we need to move and rotate the detected hand for it to occupy a
            # 256x256 square
            # line from wrist keypoint to middle finger keypoint
            # should point straight up
            center = np.float32([cx,cy])
            side = max(w,h) * self.box_enlarge
            yshift = h * self.box_shift
            source = self._get_triangle(center, keypoints[0], keypoints[2], side, yshift)

            list_sources.append(source)
            list_keypoints.append(keypoints)
            
        return list_sources, list_keypoints

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad_all = (shape.max() - shape[:2]).astype('uint32')
        pad = pad_all // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad_all[0]-pad[0]), (pad[1],pad_all[1]-pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)
        
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad


    def __call__(self, img):
        img_pad, img_norm, pad = self.preprocess_img(img)
        
        list_sources, list_keypoints = self.detect_hand(img_norm)
        if list_sources is None:
            return None, None

        list_joints = []
        list_bbox = []
        for i in range(len(list_sources)):

            source = list_sources[i]
            keypoints = list_keypoints[i]

            # calculating transformation from img_pad coords
            # to img_landmark coords (cropped hand image)
            scale = max(img.shape) / 256
            Mtr = cv2.getAffineTransform(
                source * scale,
                self._target_triangle
            )

            img_landmark = cv2.warpAffine(
                self._im_normalize(img_pad), Mtr, (256,256)
            )
        
            joints = self.predict_joints(img_landmark)
        
            # adding the [0,0,1] row to make the matrix square
            Mtr = self._pad1(Mtr.T).T
            Mtr[2,:2] = 0
            
            Minv = np.linalg.inv(Mtr)

            # projecting 2d keypoints back into original image coordinate space
            kp_orig = (self._pad1(joints[:,:2]) @ Minv.T)[:,:2]
            box_orig = (self._target_box @ Minv.T)[:,:2]
            kp_orig -= pad[::-1]
            box_orig -= pad[::-1]

            joints[:,:2] = kp_orig
            if joints.shape[1] == 3:
                # scale Z coordinate proportionally
                joints[:,2] = joints[:,2] * np.sqrt(cv2.determinant(Minv))

            list_joints.append(joints)
            list_bbox.append(box_orig)
            
        return list_joints, list_bbox
