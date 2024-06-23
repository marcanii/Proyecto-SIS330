import torch
from PPO.Agent import Agent
from UNet.UNet import UNetResnet
from CAE.maxPooling import MaxPooling
from flask import Flask, request, jsonify
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "source/UNet/models/UNetResNet_model_seg_v3_30.pt"
modelSegmentation = UNetResnet()
modelSegmentation.load_model(model_path)
modelSegmentation.to(device)
maxPooling = MaxPooling()
maxPooling.to(device)
batch_size = 4
n_epochs = 10
alpha = 0.0003
agent = Agent(n_actions=7, cuda=True, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

app = Flask(__name__)

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    x_input = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    x_input.div_(255.0)
    return x_input

@app.route('/', methods=['GET'])
def process_request():
    return jsonify({'message': 'API is running'})

@app.route('/observation', methods=['POST'])
def get_observation():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

    image = request.files['image']
    image_data = image.read()
    x_input = process_image(image_data)
    
    with torch.no_grad():
        output = modelSegmentation(x_input)[0]
        mask_img = output.argmax(dim=0, keepdim=True).unsqueeze(0).float()

    img_poo = maxPooling(mask_img).squeeze().cpu().numpy()
    reward, done = agent.calculateReward(img_poo)
    return jsonify({
        "image": img_poo.tolist(),
        "reward": reward,
        "done": done
    })

@app.route('/chooseAction', methods=['POST'])
def choose_action():
    if 'image' not in request.json:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    img = np.array(request.json['image'])
    img_poo = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32)
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
    print("State: ", observation.shape, "Action: ", action, "Probs: ", prob, "Vals: ", val, "Reward: ", reward, "Done: ", done)
    return jsonify({'message': 'Datos almacenados correctamente'}), 200

@app.route('/learn', methods=['POST'])
def learn():
    agent.learn()
    return jsonify({'message': 'Modelo actualizado correctamente'}), 200

@app.route('/saveModels', methods=['POST'])
def save_models():
    if request.json['save'] == True:
        agent.save_models()
        return jsonify({'message': 'Modelos guardados correctamente'}), 200
    
    return jsonify({'error': 'No se proporcionó el modelo a guardar'}), 400

@app.route('/loadModels', methods=['POST'])
def load_models():
    if request.json['load'] == True:
        agent.load_models()
        return jsonify({'message': 'Modelos cargados correctamente'}), 200
    
    return jsonify({'error': 'No se proporcionó el modelo a cargar'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)