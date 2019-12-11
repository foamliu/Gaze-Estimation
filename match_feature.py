# -*- encoding=utf-8 -*-
import sys
import warnings

import cv2
import pandas as pd

from GmsMatcher import *

# reload(sys)
# sys.setdefaultencoding('utf-8')

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def checksim(T):
    T = np.transpose(T)
    res = 1
    if T[0, 0] < 0 or T[0, 0] > 3 or T[1, 1] < 0 or T[1, 1] > 3 \
            or T[2, 0] > 600 or T[2, 0] < -30 or T[2, 1] > 600 or T[2, 1] < -30:
        res = 0
    return res


def CheckLabelId(image1, image2):
    w, h = (720, 960)
    MIN_MATCH_COUNT = 10

    detector = cv2.xfeatures2d.SURF_create(800)
    gray_demo = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_imag = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    (demopoints, demoscriptors) = detector.detectAndCompute(gray_demo, None)
    (imgpoints, imgscriptors) = detector.detectAndCompute(gray_imag, None)

    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(demoscriptors, imgscriptors, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([demopoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([imgpoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if len(H) == 0 or H is None:
            H = []
            rectify_image = []
            score = 0
        else:
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            dst1 = np.float32([dst[0], dst[3], dst[1], dst[2]])
            pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
            M = cv2.getPerspectiveTransform(dst1, pts1)
            rectify_image = cv2.warpPerspective(image2, M, (w, h))
            # cv2.imwrite('1111.jpg',rectify_image)
            score = float(len(good) * 50) / float(len(matches))
        if len(H) == 0:
            result = 0
            inv = 0
        else:
            res = checksim(H)
            top_left = dst1[0][0]
            top_right = dst1[1][0]
            bottom_left = dst1[2][0]
            bottom_right = dst1[3][0]
            img_weight = np.sqrt(np.sum(np.square(top_left - top_right)))
            img_height = np.sqrt(np.sum(np.square(bottom_left - bottom_right)))
            if img_weight < 900 or img_weight > 2000 or img_height < 900 or img_height > 2000:
                inv = 0
                result = 0
                score = -1
            elif top_left[0] > bottom_right[0] and top_left[1] > bottom_right[1] \
                    and 900 < img_weight < 2000 and 900 < img_height < 2000:
                inv = 1
                result = 0
                score = 0
            else:
                if res == 1:
                    result = 1
                    inv = 0
                else:
                    inv = 0
                    result = 0
                    score = 0
    else:
        result = 0
        score = 0
        inv = 0
        rectify_image = []
    return result, inv, score, rectify_image


def ReCheckLabelId(image1, photo_img):
    orb = cv2.ORB_create(1000)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    gms = GmsMatcher(orb, matcher)
    (demopoints, demoscriptors) = gms.detect_fearure(image1)
    (imgpoints, imgscriptors) = gms.detect_fearure(photo_img)

    gms_matches, matchnum = gms.compute_matches(demopoints, demoscriptors, imgpoints, imgscriptors)

    result = float(len(gms_matches)) / 1000

    return result


def Recognition(image1, image2):
    result, inv, score, rectify_image = CheckLabelId(image1, image2)
    if inv == 1:
        res = 'check'
    elif result == 1 and inv == 0:
        res = 'ok'
    elif result == 0 and score == -1:
        res = 'check'

    else:
        matchresult = ReCheckLabelId(image1, rectify_image)
        if matchresult >= 0.20:
            res = 'ok'
        else:
            res = 'check'
    return res, inv


if __name__ == '__main__':
    imagepath1 = r'data\data\frame\cron20190326\A000136389\0.jpg'
    imagepath2 = r'data\data\frame\cron20190326\A000136389\1.jpg'
    image1 = cv2.imread(imagepath1)
    image1 = cv2.resize(image1, (720, 960))
    image2 = cv2.imread(imagepath2)
    result, inv = Recognition(image1, image2)
    print(result, inv)
