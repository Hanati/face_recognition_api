from flask import Flask
from flask_restplus import Resource, Api, reqparse
import werkzeug, os
import face_recognition
import pickle
import numpy as np
from flask_cors import CORS


fin = open('/data/user_vec.pickle', 'rb')
users = pickle.load(fin)

def find_user(v1, users):
    min_user = None
    min_dist = None
    min_tf = None
    for key,val in users.items():
        dist=face_recognition.face_distance([np.array(val)],v1)
        t_or_f=face_recognition.compare_faces([np.array(val)],v1)
        if min_dist==None:
            min_dist=dist
            min_user=key
            min_tf=1*t_or_f[0]
        else:
            if dist<min_dist:
                min_dist=dist
                min_user=key
                min_tf=1*t_or_f[0]
        
    return {'user':min_user,'t_or_f':int(min_tf),'dist':float(min_dist)}


app = Flask(__name__)
CORS(app)

api = Api(app)
UPLOAD_FOLDER = '/data/'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

@api.route('/hello')

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


@api.route('/loadimage')
class PhotoUpload(Resource):
    @api.expect(parser)
    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        photo = data['file']

        if photo:
            filename = 'your_image.jpg'
            photo.save(os.path.join(UPLOAD_FOLDER,filename))
            tmp_img = face_recognition.load_image_file(os.path.join(UPLOAD_FOLDER,filename))
            tmp_loc = face_recognition.face_locations(tmp_img, model = 'cnn')
            tmp_encode = face_recognition.face_encodings(tmp_img, known_face_locations=tmp_loc)[0]
            return find_user(tmp_encode, users)
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
               }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')