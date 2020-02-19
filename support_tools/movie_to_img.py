"""
Convert movie into images.
"""
import os
import argparse
import cv2


img_name_tmplt = 'frame_{:04d}.png'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('movie', type=str,
                        help='Path to the movie file.')
    parser.add_argument('img_dir', type=str,
                        help='Path to the directory saved images.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.movie)

    frame_counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.flip(frame, -1)
        frame = frame[:, 310:1390, :]
        frame = cv2.resize(frame, (640, 640))
        frame = frame.transpose(1, 0, 2)[:, ::-1]
        save_path = os.path.join(args.img_dir, img_name_tmplt.format(frame_counter))
        cv2.imwrite(save_path, frame)
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
