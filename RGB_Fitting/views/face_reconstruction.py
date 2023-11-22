from flask import Blueprint, request
from ..services.process_data import process_data

face_reconstruction_blueprint = Blueprint('face_reconstruction', __name__)

@face_reconstruction_blueprint.route('/reconstruct', methods=['POST'])
def reconstruct_face():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # 특징점 추출
    process_data(file)

    # 결과 반환
