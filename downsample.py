import os
import cv2



def prepare_images(path, factor):
    for file in os.listdir(path):
        img = cv2.imread(path + '/' + file)
        print(img.shape)
        h, w, _ = img.shape
        new_height = h / factor
        new_width = w / factor
        img = cv2.resize(img, (int(new_width), int(new_height)), interpolation = cv2.INTER_CUBIC)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
        print('Saving {}'.format(file))
        cv2.imwrite((write_path + file), img)

new_path = os.path.join(os.getcwd(),'hr_image')
write_path = os.path.join(os.getcwd(), 'lr_image/')
print(write_path)
prepare_images(new_path, 3)
