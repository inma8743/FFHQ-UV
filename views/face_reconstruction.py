from flask import Blueprint, request, jsonify
from services.process_data import process_data
# from services.rgb_fitting import perform_rgb_fitting

face_reconstruction_blueprint = Blueprint('face_reconstruction', __name__)

@face_reconstruction_blueprint.route('/reconstruct', methods=['POST'])
def reconstruct_face():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # 특징점 추출
    processing_data = process_data(file)

    # 결과 반환
