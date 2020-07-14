#!/usr/bin/python
# -*- coding: utf-8 -*-
import base64
import sys
from os.path import join, dirname, realpath
from flask import Flask, request, jsonify
import cifar_cnn.details as details
from nas_cnn.prediction import get_class
from prediction import get_next

sys.path.append('.')
app = Flask(__name__)
UPLOADS_PATH = join(dirname(realpath(__file__)))
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH
app.config['JSON_AS_ASCII'] = False
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


@app.route('/')
def index():
    return "index page"


@app.route('/hello/<name>', methods=['GET', 'POST'])
def hello(name):
    return jsonify("hello to ," + name)


@app.route('/getDetail/<string:kind>', methods=['GET', 'POST'])
def get_details(kind):
    detail = details.details[kind]
    res = {"status": 200, "kind": kind, "detail": detail}
    return jsonify(res)


@app.route('/getKind', methods=['GET', 'POST'])
def get_kind():
    if request.method == 'POST':
        print(request)
        f = request.files['image'].read()
        with open('./image.png', "wb") as img:
            img.write(f)
        # 还需要输入网络，做回归一化

    # im = cv2.imread('./image.png')
    # im = cv2.resize(im, dsize=(32, 32))  # 缩放到32*32 大小
    # # 转换格式  1,1,32,32
    # trans = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #     ])
    # im = trans(im)
    # im = im.unsqueeze(0)
    # cv2.imwrite('image.jpg', im)
    # 转换格式  1,1,32,32
    # cv2.imshow('图像', im)
    # cv2.waitKey(0)
    # print(im)
    # kinds = kind(im)

    kinds = get_class(image_path='./image.png')
    kinds = str(kinds)
    image_path = './images/' + kinds + '.png'
    with open(image_path, 'rb') as f:
        image_res = base64.b64encode(f.read()).decode()
    detail = details.details[kinds]
    res = {"status": 200, "kind": kinds, "detail": detail, "image": image_res}
    return jsonify(res)


@app.route('/getNext/<string:word>', methods=['POST', 'GET'])
def get_text(word):
    try:
        result = get_next(word)
        return jsonify({'result': result})
    except:
        return jsonify({'result': None})


if __name__ == "__main__":
    app.run('0.0.0.0', 19010, debug=True)
