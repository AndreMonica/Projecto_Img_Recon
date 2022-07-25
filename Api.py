from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
import Main

# API image submit
# https://medium.com/csmadeeasy/send-and-receive-images-in-flask-in-memory-solution-21e0319dcc1
# https://flask-restful.readthedocs.io/en/latest/
# https://flask-restful.readthedocs.io/en/latest/quickstart.html#a-minimal-api

app = Flask(__name__)

@app.route('/maskImage' , methods=['POST'])
def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################
	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})

@app.route('/test' , methods=['GET','POST'])
def test():
    print("log: got at test" , file=sys.stderr)
    Main.get_data_train('./img/teeth/test_set')
    return jsonify({'status':'succces'})

if __name__ == '__main__':
	app.run(debug = True)
 
