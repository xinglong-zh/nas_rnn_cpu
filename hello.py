from flask import Flask,jsonify
app = Flask(__name__)


@app.route('/hello/<name>', methods=['GET', 'POST'])
def hello(name):
    return jsonify("hello to ," + name)


if __name__ == "__main__":

    app.run('0.0.0.0', 19010, debug=True)