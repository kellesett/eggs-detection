import torch
torch.classes.__path__ = []
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


class Model:
    def __init__(self):
        pass

    def detect(self, image):
        pass


class WrapperYOLO(Model):
    def __init__(self, path='yolo11n.pt', conf=0.6):
        super().__init__()
        self.model = YOLO(path)
        self.conf=conf

    def detect(self, image):
        results = self.model.predict(image, save=False, imgsz=1024, conf=self.conf)  # return a list of Results objects

        # Process results list
        boxes = results[0].boxes  # Boxes object for bounding box outputs
        results[0].save(filename="tmp.jpg")  # save to disk

        altered = Image.open('tmp.jpg')
        fig, ax = plt.subplots(1, 1)
        ax.imshow(altered)
        ax.axis(False)
        
        # Return dummy counts
        white = torch.where(boxes.cls == 0, 1, 0).sum()
        red = torch.where(boxes.cls == 1, 1, 0).sum()
        return [fig], white, red
    

class MorphBasedDetector(Model):
    def __init__(self):
        super().__init__()

    def calc(self, img):
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        canny_gray = cv2.Canny(gray, 10, 50, edges=True, L2gradient=True)
        _, binary = cv2.threshold(canny_gray, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        
        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
        ax[0].set_title(f'Изначальное изображение\n')
        ax[0].imshow(gray, cmap='gray')
        ax[1].set_title(f'После применения\nфильтра Кэнни')
        ax[1].imshow(binary, cmap='gray')
        ax[2].set_title(f'После дилатации\n')
        ax[2].imshow(dilated, cmap='gray')
        ax[3].set_title(f'После эрозии\n')
        ax[3].imshow(eroded, cmap='gray')

        for i in range(4):
            ax[i].axis(False)
        return fig, cv2.bitwise_not(eroded)
    
    def mask2class(self, img, mask):
        img, mask = np.array(img), np.array(mask)
        masked = cv2.bitwise_and(img, img, mask=mask[:, :, None])
    
        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
        ax[0].set_title(f'Исходное изображение')
        ax[0].imshow(img)
        ax[1].set_title(f'Выделенные яйца')
        ax[1].imshow(masked)
        
        # Для белых
        background = np.zeros(img.shape, dtype=np.uint8)
        lower_white = np.array([0, 0, 130])
        upper_white = np.array([255, 255, 255])    

        white_mask = cv2.inRange(masked, lower_white, upper_white)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        eroded = cv2.erode(white_mask, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        white_contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_white_contours = [contour for contour in white_contours if len(contour) > 200]

        white_mask = cv2.drawContours(background, filtered_white_contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_RGB2GRAY)

        white_masked = cv2.bitwise_and(img, img, mask=white_mask[:, :, None])
        white_masked = cv2.drawContours(img, filtered_white_contours, -1, (0, 0, 255), 10)
        
        white_cnt = len(filtered_white_contours)
        ax[2].set_title(f'{white_cnt} белых яиц найдено')
        ax[2].imshow(white_masked)
    
        lower_red = np.array([1, 1, 1])
        upper_red = np.array([255, 255, 130])    
        
        # Для коричневых яиц
        background = np.zeros(img.shape, dtype=np.uint8)
        red_mask = cv2.inRange(masked, lower_red, upper_red)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        eroded = cv2.erode(red_mask, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        red_contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_red_contours = [contour for contour in red_contours if len(contour) > 200]

        red_mask = cv2.drawContours(background, red_contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        red_mask = cv2.cvtColor(red_mask, cv2.COLOR_RGB2GRAY)

        red_masked = cv2.bitwise_and(img, img, mask=red_mask[:, :, None])
        red_masked = cv2.drawContours(img, filtered_red_contours, -1, (0, 255, 0), 10)
        
        red_cnt = len(filtered_red_contours)
        ax[3].set_title(f'{red_cnt} коричневых яиц найдено')
        ax[3].imshow(red_masked)

        for i in range(4):
            ax[i].axis(False)
        return fig, white_cnt, red_cnt

    def detect(self, image):
        fig1, mask = self.calc(image)
        fig2, white_cnt, red_cnt = self.mask2class(image, mask)
        return [fig1, fig2], white_cnt, red_cnt
    

class SegmentBaseDetector(Model):
    def __init__(self):
        super().__init__()
    
    def binarize(self, img, lower, upper):
        mask = cv2.inRange(img, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if len(contour) > 200]

        background = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        filtered_contours = self.filter(filtered_contours)
        contoured_mask = cv2.drawContours(background, filtered_contours, -1, (0, 255, 0), thickness=cv2.FILLED)
        return contoured_mask

    def filter(self, contours):
        lengths = np.array([cv2.arcLength(contour, True) for contour in contours])
        areas = np.array([cv2.contourArea(contour) for contour in contours])
        ratios = areas / lengths ** 2 * 4 * np.pi
        
        filtered_contours = [contour for contour, ratio, area in zip(contours, ratios, areas) if (
            ratio > 0.5 and area > 0. and area < 200000.
        )]
        return filtered_contours

    def detect(self, img):
        img = np.array(img)
        it = range(0, 255, 15)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        tmp = np.zeros(img.shape, np.uint8)
        for i, value in enumerate(it):
            lower = np.array([value])
            upper = np.array([value + 30])
            for channel in range(3):
                bin_= self.binarize(img[:, :, channel], lower, upper)
                tmp = tmp + bin_

        ax[0].set_title('Исходное изображение')
        ax[0].imshow(img)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bin_tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        eroded = cv2.erode(bin_tmp, kernel)

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if len(contour) > 200]

        background = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        filtered_contours = self.filter(filtered_contours)
        contoured_mask = cv2.drawContours(background, filtered_contours, -1, (0, 255, 0), thickness=cv2.FILLED)

        mask = cv2.cvtColor(contoured_mask, cv2.COLOR_RGB2GRAY)

        ax[1].set_title('Полученная маска')
        ax[1].imshow(contoured_mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        white_cnt = []
        red_cnt = []
        for i, cnt in enumerate(contours):
            background = np.zeros_like(mask)
            cnt_mask = cv2.drawContours(background, [cnt], -1, color=255, thickness=-1)
            mean_val = cv2.mean(img, mask=cnt_mask)
            if mean_val[2] < 175.:
                red_cnt.append(cnt)
            else:
                white_cnt.append(cnt)

        masked_img = cv2.bitwise_and(img, img, mask=mask)
        contoured_img = cv2.drawContours(img, white_cnt, -1, color=(0, 0, 255), thickness=10)
        contoured_img = cv2.drawContours(img, red_cnt, -1, color=(0, 255, 0), thickness=10)

        ax[2].set_title('Выделенные яйца разных цветов')
        ax[2].imshow(contoured_img)

        for i in range(3):
            ax[i].axis(False)
        white_res = len(white_cnt)
        red_res = len(red_cnt)
        return [fig], white_res, red_res