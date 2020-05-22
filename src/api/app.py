from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_pymongo import PyMongo

import numpy as np
import cv2
from skimage import io

from api.settings import SHARED_COMPONENTS

from api.inference import classify_ballot_paper


def init_app():
    '''Initialize flask app'''
    flask_app = Flask(__name__)
    # flask_app = create_routes(flask_app)

    flask_app.config.from_pyfile('config/config.cfg')
    CORS(flask_app)

    mongo = PyMongo(flask_app)

    mongo_db = mongo.db
    SHARED_COMPONENTS['db'] = mongo_db
    return flask_app


# def create_routes(app):
#     '''
#     Create routes for app
#     '''

#     app.add_url_rule(rule='/home', methods=['GET'], view_func=home_view)

#     app.add_url_rule(rule='/api/v1/predict',
#                      methods=['POST'], view_func=classify_ballot_paper)

#     return app


app = init_app()


@app.route('/home', methods=['GET', 'POST'])
def home_view():
    '''Home view'''
    print(request.method)
    if request.method == 'GET':
        html_file = open('./api/templates/home.html')
        return Response(html_file.read())
    elif request.method == 'POST':
        print('request files: ', request.files)
        if 'myfile' in request.files:
            img_file = request.files['myfile']

            print(img_file)
            # nparr = np.fromstring(data, np.uint8)

            img = io.imread(img_file)

            print(img.shape)

            pred_class = classify_ballot_paper(img)
            print(f'class: {pred_class}')

        html_file = open('./api/templates/home.html')
        return Response(html_file.read())


@app.route('/classify', methods=['POST'])
def classifier():
    '''classifier api'''
    img_file = request.files['myfile']

    print(img_file)
    # nparr = np.fromstring(data, np.uint8)

    img = io.imread(img_file)

    print(img.shape)
