import os
import cv2

def crop_image():
    directory = os.path.join(os.getcwd(),'hr_image/HR.bmp')
    img = cv2.imread(directory, cv2.IMREAD_UNCHANGED)
 
    print('Original Dimensions : ',img.shape)
 
    width = 64
    height = 64 # keep original height
    dim = (width, height)
 
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
    print('Resized Dimensions : ',resized.shape)
 
    cv2.imwrite((new_path + "/HR.bmp"), resized)

def prepare_images(path, factor):
    for file in os.listdir(path):
        img = cv2.imread(path + '/' + file)
        print(img.shape)
        h, w, _ = img.shape
        new_height = h / factor
        new_width = w / factor
        img = cv2.resize(img, (int(new_width), int(new_height)), interpolation = cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
        print('Saving {}'.format(file))
        cv2.imwrite((write_path + file), img)

new_path = os.path.join(os.getcwd(),'hr_image')
write_path = os.path.join(os.getcwd(), 'lr_image/')
crop_image()
print(write_path)
prepare_images(new_path, 3)
