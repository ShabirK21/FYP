from quart import Quart, request, redirect, url_for, jsonify
from quart.templating import render_template
from quart_auth import (
    AuthUser, current_user, login_required, login_user, logout_user, QuartAuth
)
from tortoise.transactions import in_transaction
from tortoise.contrib.quart import register_tortoise
import os
import base64
from models.models import User, Image, PredictionResult
from werkzeug.utils import secure_filename
import requests
import numpy as np
import json
import cv2
import numpy as np
from io import BytesIO
from aiohttp import ClientSession
import asyncio
from datetime import datetime
import mimetypes

app = Quart(__name__)
app.secret_key = 'a4656a7e5388916912f36da6d620897fd8e066e642ed288395073ce67d1088f6'
QuartAuth(app)

@app.route('/')
async def root():
    return await render_template('login.html', current_user=current_user)

@app.route('/home')
async def home():
    user = await User.get(id=current_user.auth_id)
    username = user.username
    return await render_template('homepage.html', current_user=current_user, username=username)

@app.route('/register', methods=['GET', 'POST'])
async def register():
    if request.method == 'POST':
        form = await request.form
        username = form.get('username')
        password = form.get('password')
        email = form.get('email')

        # Create a new user and save it to the database
        user = User(username=username, email=email)
        user.set_password(password)
        await user.save()

        # Log the user in
        login_user(user)

        # Redirect to the home page
        return redirect(url_for('home'))

    # Render the registration template
    return await render_template('register.html', current_user=current_user)

@app.route('/login', methods=['GET', 'POST'])
async def login():
    if request.method == 'POST':
        form = await request.form
        username = form.get('username')
        password = form.get('password')
        user = await User.get(username=username)
        password_check = user.check_password(password)

        if user is not None and password_check:
            login_user(user)
            return redirect(url_for('home'))

    return await render_template('login.html', current_user=current_user)

@app.route('/logout', methods=['GET', 'POST'])
async def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
async def upload():
    user = await User.get(id=current_user.auth_id)
    files = (await request.files).getlist('imageUpload[]')
    
    for file in files:
        filename = secure_filename(file.filename) 
        file_path = f'uploads/{user.id}/{filename}'

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        image = Image(user=user, path=file_path)
        await image.save()

        await file.save(file_path)

    return redirect(url_for('home'))

@app.route('/view-images', methods=['GET', 'POST'])
@login_required
async def view_images():
    user = await User.get(id=current_user.auth_id)
    if request.method == 'GET':
        images = await Image.filter(user=user)
        image_data = []
        for image in images:
            img = cv2.imread(image.path)
            img_resized = cv2.resize(img, (256, 256))
            
            # Convert the resized image to a base64-encoded string
            _, buffer = cv2.imencode('.jpg', img_resized)  # Save to buffer as JPG
            encoded_string = base64.b64encode(buffer).decode()  # Encode buffer to base64
            
            image_name = os.path.basename(image.path)
            image_data.append({'data': f'data:image/jpeg;base64,{encoded_string}', 'name': image_name})
        
        return jsonify(image_data)
    
    elif request.method == 'POST':
        data = await request.form
        image_name = data.get('name')
        image = await Image.get(user=user, path=os.path.join('uploads', str(user.id), image_name))
        
        img = cv2.imread(image.path)
        img_resized = cv2.resize(img, (256, 256))
        
        # Convert the resized image to a base64-encoded string
        _, buffer = cv2.imencode('.jpg', img_resized)
        encoded_string = base64.b64encode(buffer).decode()  # Encode buffer to base64
        
        return jsonify({'data': f'data:image/jpeg;base64,{encoded_string}', 'name': image_name})

@app.route('/view-results', methods=['GET', 'POST'])
@login_required
async def view_results():
    user = await User.get(id=current_user.auth_id)
    if request.method == 'GET':
        results = await PredictionResult.filter(user_id=user.id).all()
        prediction_values = [result.prediction_value for result in results]
        return jsonify(prediction_values)

    if request.method == 'POST':
        data = await request.form
        prediction_value = data.get('prediction_value')

        # Fetch a single prediction result
        result = await PredictionResult.filter(user_id=user.id, prediction_value=prediction_value).first()

        if result is None:
            return jsonify({"error": "No result found"}), 404

        # Helper function to encode image file to base64
        def encode_image_to_base64(file_path):
            with open(file_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Encode images to base64
        original_image_base64 = encode_image_to_base64(result.original_image)
        result_image_base64 = encode_image_to_base64(result.result_path)

        # Determine MIME type for the images
        mime_type_original = mimetypes.guess_type(result.original_image)[0] or 'image/png'
        mime_type_result = mimetypes.guess_type(result.result_path)[0] or 'image/png'

        # Prepare response data
        response_data = {
            'prediction_value': result.prediction_value,
            'classification': result.classification,
            'probability': result.probability*100,
            'original_image': f'data:{mime_type_original};base64,{original_image_base64}',
            'predicted_image': f'data:{mime_type_result};base64,{result_image_base64}'
        }

        return jsonify(response_data)



@app.route('/predict-image', methods=['GET', 'POST'])
@login_required
async def predict_image():
    user = await User.get(id=current_user.auth_id)
    if request.method == 'POST':
        data = await request.form
        image_name = data.get('image')
        prediction_value = data.get('prediction_value')

        # Construct the image path
        image_path = os.path.join('uploads', str(user.id), image_name)
        
        # Fetch the image from the database (validate if the image exists)
        image = await Image.get(user=user, path=image_path)
        if not image:
            return jsonify({'error': 'Image not found'}), 404

        # Perform classification and segmentation
        classification_result, confidence = await classify_image(image_name, user)
        predicted_image, file_path= await segment_image(image_name, user)

        # Save the prediction result in the database
        prediction_result = PredictionResult(user=user,
                                             prediction_value=prediction_value,
                                             original_image=image_path,
                                             result_path=file_path,
                                             classification=classification_result,
                                             probability=confidence,
                                             timestamp=datetime.now())
        await prediction_result.save()

        mime_type, _ = mimetypes.guess_type(image.path)
        if not mime_type:
            mime_type = 'application/octet-stream'

        # Encode the image to base64
        with open(image.path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Return the JSON response with base64-encoded image
        return jsonify({
            'original_image': f'data:{mime_type};base64,{encoded_string}',
            'predicted_image': f'data:{mime_type};base64,{predicted_image}',
            'classification': classification_result,
            'probability': str(confidence * 100),
        })

async def classify_image(image_name, user):
    image = await Image.get(user=user, path=os.path.join('uploads', str(user.id), image_name))
    img_array = preprocess_image_classification(image.path)

    data = json.dumps({"signature_name": "serving_default", "instances": img_array.tolist()})
    async with ClientSession() as session:
        async with session.post('http://tensorflow_serving:8501/v1/models/multiclass_classification:predict', data=data) as response:
            response_json = await response.json()
            predictions = response_json['predictions'][0]
            probabilities = np.array(predictions)
            confidence = np.max(probabilities) 
            class_index = np.argmax(probabilities)
            confidence = round(confidence, 2)
            class_mapping = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
            result = class_mapping.get(class_index, 'Unknown')
            return result, confidence

async def segment_image(image_name, user):
    # Load the original image
    image_path = os.path.join('uploads', str(user.id), image_name)
    image = cv2.imread(image_path)
    img_array = preprocess_image_segmentation(image_path)

    # Prepare the payload
    payload = {
        "signature_name": "serving_default",
        "instances": img_array.tolist()
    }
    headers = {"Content-Type": "application/json"}
    
    # Send request to TensorFlow Serving
    async with ClientSession() as session:
        async with session.post('http://tensorflow_serving:8501/v1/models/image_segmentation:predict',
                                data=json.dumps(payload), headers=headers) as response:
            response_json = await response.json()
            pred_mask = np.array(response_json.get('predictions', [])[0], dtype=np.float32)

    # Process the predicted mask
    processed_mask = (pred_mask > 0.5).astype(np.float32)  # Apply thresholding

    # Resize the predicted mask to the original image size
    processed_mask_resized = cv2.resize(processed_mask.squeeze(), (image.shape[1], image.shape[0]))  # Resize mask

    # Convert the grayscale mask to a 3-channel mask for overlay
    processed_mask_colored = np.stack([processed_mask_resized] * 3, axis=-1)  # Shape: (height, width, 3)

    # Normalize the mask to [0, 1] range for blending
    processed_mask_normalized = processed_mask_colored / np.max(processed_mask_colored)  # Shape: (height, width, 3)

    # Create a colored overlay with green color
    green_overlay = np.zeros_like(image)
    green_overlay[processed_mask_resized > 0.5] = [0, 255, 0]  # Green mask

    # Blend the mask with the original image
    alpha = 0.5  # Adjust the alpha value for transparency
    overlay = cv2.addWeighted(image, 1 - alpha, green_overlay, alpha, 0)

    # Save the overlay image
    file_path = f'results/{user.id}/{image_name}_result.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, overlay)

    # Convert the image to base64 for the API response
    _, buffer = cv2.imencode('.png', overlay)
    predicted_image = base64.b64encode(buffer).decode('utf-8')
    
    return predicted_image, file_path

def preprocess_image_classification(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def preprocess_image_segmentation(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

register_tortoise(
    app,
    db_url='sqlite://db.sqlite3',
    modules={'models': ['models.models']},
    generate_schemas=True,
)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

