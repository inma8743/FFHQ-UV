from flask import Flask
from RGB_Fitting.views.face_reconstruction import face_reconstruction_blueprint

app = Flask(__name__)

app.register_blueprint(face_reconstruction_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
