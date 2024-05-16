import torch
from PPO.Agent import Agent
from Yolo.yolo_seg import YOLOSeg
from CAE.maxPooling import MaxPooling
from flask import Flask, request, jsonify
from Environment import Environment
import cv2
import numpy as np
import time

model_path = "source/Yolo/runs/segment/train3/weights/best.onnx"
modelSegmentation = YOLOSeg(model_path, conf_thres=0.7, iou_thres=0.5)
env = Environment()
maxPooling = MaxPooling()
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=11, cuda=True, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)


app = Flask(__name__)

@app.route('/', methods=['GET'])
def process_request():
    return jsonify({'message': 'API is running'})

@app.route('/observation', methods=['POST'])
def get_observation():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    image_data = image.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    img_seg = modelSegmentation(img)
    img_seg_torch = torch.from_numpy(img_seg).unsqueeze(0)
    img_poo = maxPooling(img_seg_torch)
    img_poo = img_poo.to(torch.float32)
    return jsonify({
        "image": img_poo.tolist()
    })

@app.route('/step', methods=['POST'])
def step():
    if 'action' not in request.json:
        return jsonify({'error': 'No se proporcionó ninguna acción'}), 400

    action = request.json['action']
    observation, reward, done = env.step(action)
    return jsonify({
        "observation": observation.tolist(),
        "reward": reward,
        "done": done,
    })

@app.route('/chooseAction', methods=['POST'])
def choose_action():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    image_data = image.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    img_seg = modelSegmentation(img)
    img_seg_torch = torch.from_numpy(img_seg).unsqueeze(0)
    img_poo = maxPooling(img_seg_torch)
    img_poo = img_poo.to(torch.float32)
    action, probs, value = agent.choose_action(img_poo)
    return jsonify({
        "action": action,
        "probs": probs,
        "value": value
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)