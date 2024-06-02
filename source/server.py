import torch
from PPO.Agent import Agent
from UNet.UNet import UNetResnet
from CAE.maxPooling import MaxPooling
from flask import Flask, request, jsonify
from Environment import Environment
import cv2
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "source/UNet/models/UNetResNet_model_seg_v3_30.pt"
modelSegmentation = UNetResnet()
modelSegmentation.load_model(model_path)
modelSegmentation.to(device)
env = Environment()
maxPooling = MaxPooling()
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=6, cuda=True, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)


app = Flask(__name__)

@app.route('/', methods=['GET'])
def process_request():
    return jsonify({'message': 'API is running'})

@app.route('/observation', methods=['POST'])
def get_observation():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

    image = request.files['image']

    image_data = image.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    img = torch.from_numpy(np.array(img) / 255.0).float().permute(2, 0, 1)
    #print("ImageShape: ", img.shape, "ImageType: ", img.dtype, "ImageMax: ", img.max(), "ImageMin: ", img.min())
    img = img.clone().detach().unsqueeze(0).to(device)
    modelSegmentation.eval()
    #with torch.no_grad():
    output = modelSegmentation(img)[0]
    mask_img = torch.argmax(output, axis=0) #type: ignore
    img_poo = maxPooling(mask_img)
    return jsonify({
        "image": img_poo.tolist()
    })

@app.route('/chooseAction', methods=['POST'])
def choose_action():
    if 'image' not in request.json:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    img = np.array(request.json['image'])
    img_poo = torch.from_numpy(img).to(torch.float32)
    action, probs, value = agent.choose_action(img_poo)
    return jsonify({
        "action": action,
        "probs": probs,
        "value": value
    })

@app.route('/remember', methods=['POST'])
def remember():
    if 'state' not in request.json or 'action' not in request.json or 'probs' not in request.json \
            or 'value' not in request.json or 'reward' not in request.json or 'done' not in request.json:
        return jsonify({'error': 'No se proporcionó la información necesaria'}), 400
    observation = np.array(request.json['state'])
    action = request.json['action']
    prob = request.json['probs']
    val = request.json['value']
    reward = request.json['reward']
    done = request.json['done']
    agent.remember(observation, action, prob, val, reward, done)
    return jsonify({'message': 'Datos almacenados correctamente'}), 200

@app.route('/learn', methods=['POST'])
def learn():
    agent.learn()
    return jsonify({'message': 'Modelo actualizado correctamente'}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)