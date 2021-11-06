import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFile
import glob
import random
import cv2
import scipy.io as sio
import os
from centerface.centerface import CenterFace
import time
import matplotlib.pyplot as plt
import face_recognition


__version__ = '0.3.0'


def main():
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    parser.add_argument('--face_model', default='DLib', help='CenterFace / DLib')
    parser.add_argument('--pic_path',
                        default='/home/tducnguyen/NguyenTran/Project/26_Pig_Monitor/Code/face-mask/input/',
                        help='Picture path.')
    parser.add_argument('--mask_path',
                        default='/home/tducnguyen/NguyenTran/Project/26_Pig_Monitor/Code/face-mask/face_mask/images/',
                        help='Picture path.')
    parser.add_argument('--save_path',
                        default='/home/tducnguyen/NguyenTran/Project/26_Pig_Monitor/Code/face-mask/output/',
                        help='Output path.')
    parser.add_argument('--camera', default=False , help='Input from camera')
    parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')

    args = parser.parse_args()

    if args.camera:
        print("Read from camera")
        cap = cv2.VideoCapture(0)
        mask_pths = glob.glob(args.mask_path + '*.png')
        face_model = args.face_model

        while True:
            ret, frame = cap.read()

            FaceMasker(np.array(frame), mask_pths[0], args.show, args.model, save_pth=None, face_model=face_model,
                       image_flg=False).mask()

            cv2.imshow('frame', FaceMasker._face_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        img_pths = glob.glob(args.pic_path + '*.jpg')
        mask_pths = glob.glob(args.mask_path + '*.png')
        save_pth = args.save_path
        face_model = args.face_model
        for img in img_pths:
            msk_id = random.randint(0, len(mask_pths)-1)
            FaceMasker(img, mask_pths[msk_id], args.show, args.model, save_pth, face_model).mask()
            # for msk in mask_pths:
        #     FaceMasker(img, msk, args.show, args.model, save_pth).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog', save_pth=None, face_model='CenterFace', image_flg = True):
        self.image_flg = image_flg
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self._mask_name = mask_path.split('/')[-1].split('.')[-2]
        self.save_pth = save_pth
        self.face_model = face_model
        if self.face_model == 'CenterFace':
            self.centerface = CenterFace(landmarks=True)

    def mask(self):
        if self.face_model == 'DLib':
            if self.image_flg:
                face_image_np = face_recognition.load_image_file(self.face_path)
            else:
                face_image_np = self.face_path
            face_locations = face_recognition.face_locations(face_image_np, model=self.model)
            face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
            self._face_img = Image.fromarray(face_image_np)
            self._mask_img = Image.open(self.mask_path)

            found_face = False
            for face_landmark in face_landmarks:
                # check whether facial features meet requirement
                skip = False
                for facial_feature in self.KEY_FACIAL_FEATURES:
                    if facial_feature not in face_landmark:
                        skip = True
                        break
                if skip:
                    continue

                # mask face
                found_face = True
                self._mask_face(face_landmark)

            if found_face:
                if self.show:
                    self._face_img.show()

                # save
                self._save()
            else:
                print('Found no face.')

        else:
            face_image = cv2.imread(self.face_path)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            h, w = face_image.shape[:2]
            face_locations, face_landmarks = self.centerface(face_image, h, w, threshold=0.35)
            self._face_img = Image.fromarray(face_image)
            self._mask_img = Image.open(self.mask_path)

            for f_id in range(face_landmarks.shape[0]):
                face_landmark = face_landmarks[f_id, :]
                face_location = face_locations[f_id, :]
                self._mask_face_CF(face_landmark)
                self._save()

    def _mask_face_CF(self, face_landmark: dict):
        left_eye = (int(face_landmark[0]), int(face_landmark[1]))
        right_eye = (int(face_landmark[2]), int(face_landmark[3]))
        nose = (int(face_landmark[4]), int(face_landmark[5]))
        left_mouth = (int(face_landmark[6]), int(face_landmark[7]))
        right_mouth = (int(face_landmark[8]), int(face_landmark[9]))

        middle_eye = (int(left_eye[0] + abs(nose[0] - left_eye[0])/2), min(left_eye[1], right_eye[1]))
        middle_mouth = (int(left_mouth[0] + abs(nose[0] - left_mouth[0])/2), min(left_mouth[1], right_mouth[1]))

        middle_top_mask = (int((middle_eye[0] + nose[0])/2), int(abs(nose[1] - middle_eye[1])/2 + middle_eye[1]))
        middle_bottom_mask = (int((left_mouth[0] + right_mouth[0])/2), int(middle_mouth[1] + (middle_mouth[1] - nose[1])*2))


        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(middle_bottom_mask[1] - middle_top_mask[1])

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = abs(middle_mouth[0] - left_mouth[0])*2.5
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = abs(right_mouth[0] - middle_mouth[0])*2.5
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size).convert("RGBA")
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(middle_mouth[1] - middle_top_mask[1], middle_mouth[0] - middle_top_mask[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (middle_bottom_mask[0] + middle_top_mask[0]) // 2
        center_y = (middle_bottom_mask[1] + middle_top_mask[1]) // 2
        # center_x = middle_mouth
        # center_y = middle_mouth

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size).convert("RGBA")
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        path_splits = os.path.splitext(self.face_path)
        file_name = path_splits[0].split('/')[-1]
        new_face_path = self.save_pth + file_name + '-with-mask-{}'.format(self._mask_name) + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    main()
