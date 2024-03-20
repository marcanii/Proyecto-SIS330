import numpy as np
from PPO.Agent import Agent

# Función para simular un paso de tiempo en el entorno del robot
def simulate_step():
    # Aquí simularías la interacción del robot con el entorno en un solo paso de tiempo
    # Esto incluiría la lectura de los sensores, toma de decisiones, actuación en el entorno, etc.
    # Por simplicidad, aquí simplemente se generan observaciones aleatorias

    # Generar observaciones aleatorias
    proximity_sensors = np.random.rand(4)  # Lecturas aleatorias de los sensores de proximidad
    wheel_speeds = np.random.rand(4)  # Velocidades aleatorias de las ruedas
    linear_velocity = np.random.rand()  # Velocidad lineal aleatoria
    angular_velocity = np.random.rand()  # Velocidad angular aleatoria
    camera_image = np.random.rand(64, 64, 3)  # Imagen aleatoria de la cámara (64x64 píxeles RGB)

    return proximity_sensors, wheel_speeds, linear_velocity, angular_velocity, camera_image

# Función para procesar y normalizar las observaciones
def preprocess_observation(observation):
    normalized_proximity_sensors = observation['proximity_sensors'] / 5.0  # Normalizar los sensores de proximidad al rango [0, 1]
    normalized_wheel_speeds = observation['wheel_speeds'] / 2.0  # Normalizar las velocidades de las ruedas al rango [0, 1]
    normalized_linear_velocity = (observation['linear_velocity'] + 1) / 2.0  # Normalizar la velocidad lineal al rango [0, 1]
    normalized_angular_velocity = (observation['angular_velocity'] + 1) / 2.0  # Normalizar la velocidad angular al rango [0, 1]

    # No normalizamos la imagen de la cámara, ya que la normalización se suele hacer durante el preprocesamiento de la imagen

    # Combinar todas las observaciones normalizadas en un ndarray
    processed_observation = np.concatenate((normalized_proximity_sensors, normalized_wheel_speeds, [normalized_linear_velocity, normalized_angular_velocity]))

    return processed_observation

# Función para simular un episodio completo en el entorno del robot
def simulate_episode():
    observations = []
    actions = []
    probs = []
    vals = []
    rewards = []
    dones = []

    # Número de pasos de tiempo en el episodio
    num_steps = 100

    for _ in range(num_steps):
        # Simular un paso de tiempo en el entorno
        proximity_sensors, wheel_speeds, linear_velocity, angular_velocity, camera_image = simulate_step()

        # Observaciones
        observation = {
            'proximity_sensors': proximity_sensors,
            'wheel_speeds': wheel_speeds,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'camera_image': camera_image
        }

        # Preprocesar y normalizar las observaciones
        processed_observation = preprocess_observation(observation)

        # Recolectar las observaciones
        observations.append(processed_observation)

        # Simular una acción del agente (por ahora, solo se elige aleatoriamente)
        action = np.random.choice(['left', 'right', 'up', 'down', 'stay'])
        actions.append(action)

        # Simular probabilidades (no se utilizan realmente para ahora)
        probs.append(np.random.rand())

        # Simular valores (no se utilizan realmente para ahora)
        vals.append(np.random.rand())

        # Simular recompensas (por ahora, solo se generan aleatoriamente)
        reward = np.random.rand()
        rewards.append(reward)

        # Simular flag de done (por ahora, no se utiliza realmente)
        done = False
        dones.append(done)

    return observations, actions, probs, vals, rewards, dones

if __name__ == '__main__':
    # Parámetros de entrenamiento
    n_epochs = 128
    batch_size = 64
    n_steps = 0
    N = 20
    learn_iters = 0
    score_history = []
    # Crear el agente PPO
    actions = ["left", "right", "up", "down", "stay"]
    input_dims = 6  # Modificar según las dimensiones de las observaciones preprocesadas
    agent = Agent(n_actions=len(actions), batch_size=batch_size, n_epochs=n_epochs, input_dims=input_dims)

    for i in range(n_epochs):
        observation = simulate_episode()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = simulate_step()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    
    print('Entrenamiento completado')
