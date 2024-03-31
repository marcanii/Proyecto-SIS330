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
    camera_image = np.random.rand(59, 105, 3)  # Imagen aleatoria de la cámara (64x64 píxeles RGB)

    return proximity_sensors, wheel_speeds, linear_velocity, angular_velocity, camera_image

# Función para procesar y normalizar las observaciones
def preprocess_observation(observation):
    normalized_proximity_sensors = observation['proximity_sensors'] / 5.0  # Normalizar los sensores de proximidad al rango [0, 1]
    normalized_wheel_speeds = observation['wheel_speeds'] / 2.0  # Normalizar las velocidades de las ruedas al rango [0, 1]
    normalized_linear_velocity = (observation['linear_velocity'] + 1) / 2.0  # Normalizar la velocidad lineal al rango [0, 1]
    normalized_angular_velocity = (observation['angular_velocity'] + 1) / 2.0  # Normalizar la velocidad angular al rango [0, 1]

    # No normalizamos la imagen de la cámara, ya que la normalización se suele hacer durante el preprocesamiento de la imagen

    # Combinar todas las observaciones normalizadas en un ndarray
    processed_observation = np.concatenate((normalized_proximity_sensors, normalized_wheel_speeds, [normalized_linear_velocity, normalized_angular_velocity], observation['camera_image'].flatten()))

    return processed_observation

if __name__ == '__main__':
    # Parámetros de entrenamiento
    n_epochs = 128
    batch_size = 64
    n_actions = 5  # Número de acciones posibles (left, right, up, down, stay)
    input_dims = 4 + 4 + 1 + 1 + 64 * 64 * 3  # Dimensiones de las observaciones preprocesadas

    # Crear el agente PPO
    agent = Agent(n_actions=n_actions, batch_size=batch_size, n_epochs=n_epochs, input_dims=input_dims)

    for i in range(n_epochs):
        # Simular un episodio
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

            # Elegir una acción
            action, prob, val = agent.choose_action(processed_observation)
            actions.append(action)
            probs.append(prob)
            vals.append(val)

            # Simular recompensas (por ahora, solo se generan aleatoriamente)
            reward = np.random.rand()
            rewards.append(reward)

            # Simular flag de done (por ahora, no se utiliza realmente)
            done = False
            dones.append(done)

        # Almacenar las memorias del episodio
        for observation, action, prob, val, reward, done in zip(observations, actions, probs, vals, rewards, dones):
            agent.remember(observation, action, prob, val, reward, done)

        # Aprender de las memorias recolectadas
        agent.learn()

        # Imprimir progreso
        print(f'Epoch {i+1}/{n_epochs}')

    print('Entrenamiento completado')
    agent.save_models()