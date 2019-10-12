import argparse
import sys
import numpy as np
import subprocess as sp
import cv2
from hand_tracker import HandTracker

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump weights from a Caffe model'
    )
    parser.add_argument(
        '--video',
        dest='video',
        help='video file path',
        default=None,
        type=str
    )
    parser.add_argument(
        '--outdir',
        dest='outdir',
        help='output directory',
        default='outputs',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)

def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def draw_hands(img, kpts, boxs):
    out_img = img.copy()
    for idx in range(len(kpts)):
        kpt = kpts[idx]
        box = boxs[idx]
        for i in range(4):
            j = (i + 1) % 4
            p0 = (int(box[i,0]+0.5),int(box[i,1]+0.5))
            p1 = (int(box[j,0]+0.5),int(box[j,1]+0.5))
            cv2.line(out_img, p0, p1, (0,0,255), 2)
        for i in range(kpt.shape[0]):
            p = (int(kpt[i,0]+0.5),int(kpt[i,1]+0.5))
            cv2.circle(out_img, p, 2, (255,0,0), 2)
    return out_img

def main(args):

    det = HandTracker('models/palm_detection_without_custom_op.tflite',
                      'models/hand_landmark_3d.tflite',
                      'data/anchors.csv',
                      box_shift=-0.5, box_enlarge=2.6)

    video_name = args.video
    out_dir = args.outdir
    
    print('Processing {}'.format(video_name))
    for frame_i, img in enumerate(read_video(video_name)):
        img_rgb = img[:,:,::-1]
        kpts, boxs = det(img_rgb)
        out_file = out_dir + '/' + str(frame_i) + '.jpg'
        if boxs is None:
            out_img = img
        else:
            out_img = draw_hands(img, kpts, boxs)
        cv2.imwrite(out_file, out_img)

if __name__ == '__main__':
    args = parse_args()
    main(args)

