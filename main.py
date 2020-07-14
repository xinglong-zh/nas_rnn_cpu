# coding=utf-8
from flask import Flask, request
from prediction import get_next

app = Flask(__name__)


# 访问根路径
@app.route('/get_next', methods=['POST', 'GET'])
def main():
    request_value = request.args.get("word")
    print("单词是:" + request_value)
    result_value = get_next(request_value)
    if result_value == None:
        return "无返回"
    str_value = " "
    for i in range(len(result_value)):
        str_value = str_value + result_value[i] + "   "

    return '返回结果包括:' + str_value


# 执行py文件时，运行flask对象
if __name__ == '__main__':
    app.run('0.0.0.0', 19010, debug=True)
