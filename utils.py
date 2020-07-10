# get grayscale image
import cv2
import numpy as np
import math
from scipy import ndimage


BLUR_KERNEL_SIZE = (5, 5)



def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dilation_followed_by_erosion(image):
    erosion_elem = 0
    erosion_size = 0
    dilation_elem = 0
    dilation_size = 0
    max_elem = 2
    max_kernel_size = 21


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


def remove_noise1(image):
    return cv2.fastNlMeansDenoising(image, 9, 13)

def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        8: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 31)
    }
    return switcher.get(argument, "Invalid method")

def thresholding(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 31)
    # return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # return cv2.threshold(image,127,255,cv2.THRESH_BINARY)

def threshold1(image):
    img = cv2.bilateralFilter(image, 9, 75, 75)
    return None

def rotate_image(image):
    img_edges = cv2.Canny(image, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=5, maxLineGap=300)
    angles = []
    # print(lines[0])
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(image, median_angle)

    # print("Angle is {}".format(median_angle))

    # fixes = cv2.imwrite('rotated.jpg', img_rotated)
    # cv2.namedWindow("h", cv2.WINDOW_NORMAL)
    # cv2.imshow("h", img_rotated)
    # cv2.waitKey(0)
    return img_rotated
# thresholding

def apply_filters(image, denoise=False):


    if denoise:
        denoised_gray = cv2.fastNlMeansDenoising(image, None, 9, 13)
        source_blur = cv2.GaussianBlur(denoised_gray, BLUR_KERNEL_SIZE, 3)

    else:
        source_blur = cv2.GaussianBlur(image, (3, 3), 3)
    source_thresh = cv2.adaptiveThreshold(source_blur, 255, 0, 1, 5, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    source_eroded = cv2.erode(source_thresh, kernel, iterations=1)
    source_dilated = cv2.dilate(source_eroded, kernel, iterations=1)

    return source_dilated


def morph_opening(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    return opening


def morph_closing(image):
    kernel = np.ones((4, 4), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # dilation = cv2.dilate(image, kernel, iterations=1)
    # closing = cv2.erode(dilation, kernel, iterations=1)
    return closing


def overlap(bb0, bb1):
    x0, y0, w0, h0 = bb0[:4]
    x1, y1, w1, h1 = bb1[:4]
    xa = max(x0, x1)
    ya = max(y0, y1)
    xb = min(x0 + w0, x1 + w1)
    yb = min(y0 + h0, y1 + h1)
    if xa > xb or ya > yb:
        return 0
    i = (xb - xa) * (yb - ya)
    u = (w0 * h0) + (w1 * h1) - i
    return i / u


def lies_insame_line(box1, box2, thresh):
    if abs(box1[1] - box2[1]) < thresh or abs(box2[3] - box1[3]) < thresh:
        return True
    else:
        return False


def inside_box(outer, inner):
    x1, y1, w1, h1 = outer
    x2, y2, w2, h2 = inner
    if x1 < x2 and y1 < y2:
        if x1 + w1 > x2 + w2 and y1 + h1 > y2 + h2:
            # if w1 > w2 and h1>h2:
            return True

        #     print("completely inside")
        # else:
        #     print("partially inside")


def imshow(name, image, wait=None):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    if wait is not None:
        cv2.waitKey(wait)


def get_boxes_contours(la, draw=False):
    lar = cv2.imread(la)
    contour_box_list = []
    rgb = cv2.pyrDown(lar)
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