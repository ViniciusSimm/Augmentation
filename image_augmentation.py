# para inserir o augmentation diretamente no modelo:
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/ungraded_labs/C2_W2_Lab_1_cats_v_dogs_augmentation.ipynb

# para gerar imagens alteradas e salvar no HD:
# https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5

import cv2
import random
import os

class Transformation():
    def __init__(self,path):
        self.path = path
        self.img = cv2.imread(path)
        x,y,_ = self.img.shape
        self.shape = (y,x)

    def resize_img(self,img):
        return cv2.resize(img, self.shape, interpolation = cv2.INTER_AREA)

    def fill(self, img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def horizontal_shift(self,ratio = 0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return self.img
        ratio = random.uniform(-ratio, ratio)
        h, w = self.img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = self.img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = self.img[:, int(-1*to_shift):, :]
        img = self.fill(img, h, w)
        img = self.resize_img(img)
        return img
    
    def vertical_shift(self, ratio = 0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return self.img
        ratio = random.uniform(-ratio, ratio)
        h, w = self.img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = self.img[:int(h-to_shift), :, :]
        if ratio < 0:
            img = self.img[int(-1*to_shift):, :, :]
        img = self.fill(img, h, w)
        img = self.resize_img(img)
        return img

    def zoom(self, value):
        if value > 1 or value < 0:
            print('Value for zoom should be less than 1 and greater than 0')
            return self.img
        value = random.uniform(value, 1)
        h, w = self.img.shape[:2]
        h_taken = int(value*h)
        w_taken = int(value*w)
        h_start = random.randint(0, h-h_taken)
        w_start = random.randint(0, w-w_taken)
        img = self.img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
        img = self.fill(img, h, w)
        img = self.resize_img(img)
        return img

    def horizontal_flip(self, flag):
        if flag:
            img = self.resize_img(self.img)
            return cv2.flip(img, 1)
        else:
            return self.img

    def vertical_flip(self, flag):
        if flag:
            img = self.resize_img(self.img)
            return cv2.flip(img, 0)
        else:
            return self.img

    def rotation(self, angle):
        angle = int(random.uniform(-angle, angle))
        h, w = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(self.img, M, (w, h))
        img = self.resize_img(img)
        return img

    def gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

# t = Transformation('images/exemplo_1.jpg')
# img = t.gray()
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":

    MAIN_FOLDER = 'images'
    paths = os.listdir(MAIN_FOLDER)

    for path in paths:
        parts = path.split('.')


        name = str(MAIN_FOLDER+'/'+path)
        transformation = Transformation(name)


        # Horizontal Shift
        img = transformation.horizontal_shift(ratio=0.1)
        save_path = str(MAIN_FOLDER+'/'+parts[0]+'_horizontalshift'+'.'+parts[1])
        cv2.imwrite(save_path,img)


        # Vertical Shift
        img = transformation.vertical_shift(ratio=0.1)
        save_path = str(MAIN_FOLDER+'/'+parts[0]+'_verticalshift'+'.'+parts[1])
        cv2.imwrite(save_path,img)


        # Zoom
        img = transformation.zoom(value=0.9)
        save_path = str(MAIN_FOLDER+'/'+parts[0]+'_zoom'+'.'+parts[1])
        cv2.imwrite(save_path,img)


        # Horizontal Flip
        img = transformation.horizontal_flip(True)
        save_path = str(MAIN_FOLDER+'/'+parts[0]+'_horizontalflip'+'.'+parts[1])
        cv2.imwrite(save_path,img)


        # Vertical Flip
        img = transformation.vertical_flip(True)
        save_path = str(MAIN_FOLDER+'/'+parts[0]+'_verticalflip'+'.'+parts[1])
        cv2.imwrite(save_path,img)


        # Rotation
        img = transformation.rotation(angle=30)
        save_path = str(MAIN_FOLDER+'/'+parts[0]+'_rotation'+'.'+parts[1])
        cv2.imwrite(save_path,img)

