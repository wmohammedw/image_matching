import cv2
import numpy as np
from flask import Flask, request, jsonify
import jsonpickle

app = Flask(__name__)

imgs = []
img1, img2 = None, None


@app.route('/enterData/', methods=['POST', 'GET'])
def enterData():
    global imgs
    r = request
    imgs.append(r.data)

    return 'success'


@app.route('/isItSimilar/', methods=['POST', 'GET'])
def isItSimilar():
    global img1, img2, imgs
    FLAN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    img1 = cv2.imdecode(np.fromstring(
        imgs[0], np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.fromstring(
        imgs[1], np.uint8), cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    kyp1, des1 = sift.detectAndCompute(img1, None)
    kyp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    counter = 0
    fail = 0

    for m1, m2 in matches:
        if m1.distance < 0.6 * m2.distance:
            good_matches.append([m1])
            counter += 1
        else:
            fail += 1

    flann_matches = cv2.drawMatchesKnn(
        img1, kyp1, img2, kyp2, good_matches, None, flags=2)
    # confidence = 100 - ((counter / fail) * 100)
    confidence = ((counter / len(matches)))

    # response = jsonify({
    #     'confidence': confidence,
    # })
    # response.headers.add("Access-Control-Allow-Origin", "*")
    # return response
    imgs = []
    return jsonpickle.encode({'img_matches': flann_matches, 'confidence': confidence})


if __name__ == '__main__':
    app.run()
