import os

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import torch.nn as nn


# 加载maskrcnn模型进行目标检测
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)


class MRCNN:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self.model.eval()
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def get_prediction(self, img_path, threshold):
        # self.__int__()
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        # print('pred')
        # print(pred)
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        # print("masks>0.5")
        # print(pred[0]['masks'] > 0.5)
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        # print("this is masks")
        # print(masks)
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t + 1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return masks, pred_boxes, pred_class

    def random_colour_masks(self, image):
        colours = [[255, 255, 255], [255, 255, 255], [255, 255, 255]]
        # colours = [[25, 25, 255], [25, 25, 25], [25, 25, 25]]

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, len(colours) / 3)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask

    def isexist_dir(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

    def instance_segmentation_api_orig(self, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        masks, boxes, pred_cls = self.get_prediction(img_path, threshold)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            rgb_mask = self.random_colour_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)

        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def instance_segmentation_api(self, img_path, save_path='./outout', threshold=0.5, rect_th=3):
        masks, boxes, pred_cls = self.get_prediction(img_path, threshold)
        self.isexist_dir(save_path)
        is_person = [True if cls in 'person' else False for cls in pred_cls]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        full_rgb_mask = np.zeros(img.shape, dtype=np.uint8)
        for i in range(len(masks)):
            if is_person[i] == True:
                rgb_mask, randcol = self.random_colour_masks(masks[i]), (
                    random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(rgb_mask, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                              thickness=rect_th)  # add mask bboxes
                full_rgb_mask = cv2.bitwise_or(full_rgb_mask, rgb_mask)  # save full mask img
        masked_img = cv2.bitwise_and(img, full_rgb_mask)
        # save masked image
        masked_img = Image.fromarray(masked_img)
        masked_img.save(save_path + "/" + "{}_masked.jpg".format(img_path.split('/')[-1].split('.')[0]))
        # show result image
        plt.figure(figsize=(20, 30))
        plt.imshow(masked_img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

if __name__ == “main”:
    maskedCNN = MRCNN()
    maskedCNN.instance_segmentation_api(img_path='./input/image2.jpg')
