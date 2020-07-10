import pytesseract
import cv2
from pytesseract import Output
from config import citizenship_np,citizenship_mapper,citizenship_eng
from utils import get_grayscale, thresholding, remove_noise, morph_closing, imshow, morph_opening,inside_box,lies_insame_line
import numpy as np


def ocr(image,rgb, language,draw=False):
    imager =cv2.pyrDown(image)
    if language=='nep':
        boxes = pytesseract.image_to_data(imager, output_type=Output.DICT, lang='nep')
    if language =='eng':
        boxes = pytesseract.image_to_data(imager, output_type=Output.DICT, lang='eng')
    boxes_list = []
    bboxes = {}
    n_boxes = len(boxes['text'])
    for i in range(n_boxes):
        if int(boxes['conf'][i]) > 60:
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            tuple1 = (x, y, x + w, y + h)

            # print(tuple1)
            boxes_list.append(tuple1)
            text = boxes['text'][i]
            # print(boxes)
            bboxes[tuple1] = text
            imager = cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 0), 2)
    if draw:
        imshow("output", imager, 0)
    # print(boxes_list)
    return bboxes, boxes_list


def distance(box1, box2):
    box1_ymin, box1_xmin, box1_ymax, box1_xmax = box1
    box2_ymin, box2_xmin, box2_ymax, box2_xmax = box2
    x_distance = min(abs(box1_xmin - box2_xmin), abs(box1_xmin - box2_xmax), abs(box1_xmax - box2_xmin),
                     abs(box1_xmax - box2_xmax))
    y_distance = min(abs(box1_ymin - box2_ymin), abs(box1_ymin - box2_ymax), abs(box1_ymax - box2_ymin),
                     abs(box1_ymax - box2_ymax))
    dist = x_distance + y_distance
    return dist


def merge_boxes(box1, box2):
    box1_ymin, box1_xmin, box1_ymax, box1_xmax = box1
    box2_ymin, box2_xmin, box2_ymax, box2_xmax = box2
    return [min(box1_ymin, box2_ymin),
            min(box1_xmin, box2_xmin),
            max(box1_ymax, box2_ymax),
            max(box1_xmax, box2_xmax)]


def pre_process(image, draw=False):
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    noise = remove_noise(thresh)
    opening = morph_opening(noise)
    closing = morph_closing(noise)
    if draw:
        imshow("original_image", image)
        imshow("thresh", thresh)
        imshow("noise", noise)
        imshow("opening", opening)
        imshow("closing", closing, wait=0)
    return gray


def Convert(lst):
    it = iter(lst)
    res_dct = dict(zip(it, it))
    return res_dct


def get_boxes_contours(la, draw= False):
    # lar = cv2.imread(la)
    contour_box_list = []
    rgb = cv2.pyrDown(la)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        tupl = (x, y, w + x, h + y)
        contour_box_list.append(tupl)
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        cropped = rgb[y:y + h, x:x + w]

        # text = pytesseract.image_to_string(cropped, lang='nep')
        # print(text)
    if draw:
        imshow('rects', rgb, 0)
        cv2.waitKey(0)
    return rgb, contour_box_list


def parse_citizenship(filename, draw = False,front = True):
    close_dist = 20
    final_dict = {}
    # copy =cv2.imread(filename)
    original = cv2.pyrDown(filename)
    new_boxes_list = []
    new_boxes_dict = {}
    # image = cv2.imread(filename)
    rgb, contour_box_list = get_boxes_contours(filename, draw=draw)

    pre_processed_image = pre_process(filename, draw=draw)
    if front :
        bboxes_text, boxes_list = ocr(pre_processed_image, rgb,language='nep', draw=draw)
    if not front:
        bboxes_text, boxes_list = ocr(pre_processed_image, rgb, language='eng',draw=draw)
    unmerged_box_dict= {}

    for i in range(0, len(boxes_list)):
        for j in range(i + 1, len(boxes_list)):
            d = distance(boxes_list[i], boxes_list[j])
            unmerged_box_dict[boxes_list[i]]= bboxes_text[boxes_list[i]]

            if d < close_dist:
                if lies_insame_line(boxes_list[i],boxes_list[j],thresh=10):
                    new_box = merge_boxes(boxes_list[i], boxes_list[j])

                    new_boxes_list.append(new_box)
                    l, t, r, b = new_box
                    image = cv2.rectangle(original, (l, t), (r, b), (93, 25, 111), 2)
                    new_boxes_dict[tuple(new_box)] = bboxes_text[boxes_list[i]] + " " + bboxes_text[boxes_list[j]]



    if draw:
        imshow("mergerd", image, 0)
        cv2.waitKey(0)
    new_boxes_dict.update(unmerged_box_dict)
    result_dict = {}

    keys_list = list(new_boxes_dict.keys())
    values_list= list(unmerged_box_dict.keys())
    result_values=[]
    key_value=[]
    if  front:
        for i in range(len(keys_list)):
            for j in range(i + 1, len(keys_list)):
                new_list = []
                if lies_insame_line(keys_list[i], keys_list[j], thresh = 10):
                    if keys_list[i][0] < keys_list[j][0]:
                        if new_boxes_dict[keys_list[i]].strip() in citizenship_np:
                            if new_boxes_dict[keys_list[j]] not in citizenship_np:
                                a= new_boxes_dict[keys_list[i]].strip()
                                if a in result_dict:
                                    val = result_dict[new_boxes_dict[keys_list[i]]] + new_boxes_dict[keys_list[j]]

                                else:
                                    result_dict[new_boxes_dict[keys_list[i]]] = new_boxes_dict[keys_list[j]]
    if not front:
        for i in range(len(keys_list)):
            for j in range(i + 1, len(keys_list)):
                new_list = []
                if lies_insame_line(keys_list[i], keys_list[j], thresh=10):
                    if keys_list[i][0] < keys_list[j][0]:
                        if new_boxes_dict[keys_list[i]].strip() in citizenship_eng:
                            if new_boxes_dict[keys_list[j]] not in citizenship_eng:
                                a = new_boxes_dict[keys_list[i]].strip()
                                if a in result_dict:
                                    val = result_dict[new_boxes_dict[keys_list[i]]] + new_boxes_dict[keys_list[j]]

                                else:
                                    result_dict[new_boxes_dict[keys_list[i]]] = new_boxes_dict[keys_list[j]]
    new={}
    rows= result_dict
    name_map = citizenship_mapper
    for key, value in rows.items():
        for key1, value1 in name_map.items():
            if key == key1:
                new[name_map[key1]] = rows[key1]





    return new



if __name__ == '__main__':
    # filename = "images/citizenship_front.jpg"
    filename = "../data/images/vge.jpg"
    filename1 = "images/back.jpg"

    parse_citizenship(filename=filename, draw=False,front = True)

    # parse_citizenship(filename=filename1,draw=False,front = False)


