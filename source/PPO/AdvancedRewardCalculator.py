import numpy as np
from scipy.ndimage import center_of_mass

class AdvancedRewardCalculator:
    def __init__(self, image_shape=(60, 108), history_length=10):
        self.image_shape = image_shape
        self.history_length = history_length
        self.position_history = []
        self.previous_center = None
        self.cumulative_reward = 0

    def calculate_reward(self, masks):
        done = False
        total_area = np.prod(self.image_shape)
        camino = np.sum(masks == 2)
        obs = np.sum(masks == 1)
        
        # 1. Recompensa base por área de camino
        reward_camino_base = 2 * (camino / total_area)
        
        # 2. Recompensa por posición central en el camino
        center_reward = self.calculate_center_reward(masks)
        
        # 3. Recompensa por continuidad del camino
        continuity_reward = self.calculate_continuity_reward(masks)
        
        # 4. Penalización por obstáculos
        obstacle_penalty = self.calculate_obstacle_penalty(masks)
        
        # 5. Recompensa por progreso
        progress_reward = self.calculate_progress_reward(masks)
        
        # 6. Penalización por zigzag
        zigzag_penalty = self.calculate_zigzag_penalty()
        
        # Cálculo de la recompensa total
        reward = (
            reward_camino_base + 
            center_reward + 
            continuity_reward + 
            obstacle_penalty + 
            progress_reward + 
            zigzag_penalty
        )
        
        # Condiciones de finalización
        if obs > camino or camino == 0:
            done = True
            reward -= 3  # Penalización ajustada por salirse completamente del camino
        elif camino >= total_area * 0.75:
            done = True
            reward += 3  # Bonificación ajustada por completar el recorrido
        
        return reward, done

    def calculate_center_reward(self, masks):
        road_mask = masks == 2
        if np.sum(road_mask) == 0:
            return -1  # Penalización si no hay camino visible
        
        center_y, center_x = center_of_mass(road_mask)
        ideal_center_y = self.image_shape[0] / 2
        
        # Recompensa basada en qué tan cerca está el centro del camino del centro de la imagen
        distance_from_center = abs(center_y - ideal_center_y) / ideal_center_y
        return (1 - distance_from_center) * 1.5  # Ajuste del factor para mayor recompensa

    def calculate_continuity_reward(self, masks):
        road_mask = masks == 2
        continuity = np.sum(road_mask[-10:, :]) / (10 * self.image_shape[1])
        return continuity  # Ajuste para balancear

    def calculate_obstacle_penalty(self, masks):
        obs_mask = masks == 1
        total_area = np.prod(self.image_shape)
        obstacle_ratio = np.sum(obs_mask) / total_area
        return -3 * obstacle_ratio  # Penalización ajustada proporcional al área de obstáculos

    def calculate_progress_reward(self, masks):
        road_mask = masks == 2
        current_center = center_of_mass(road_mask)
        
        if self.previous_center is not None:
            progress = current_center[1] - self.previous_center[1]
            self.previous_center = current_center
            return 3 * (progress / self.image_shape[1])  # Ajuste para balancear y normalizado por el ancho de la imagen
        
        self.previous_center = current_center
        return 0

    def calculate_zigzag_penalty(self):
        if len(self.position_history) < 3:
            return 0
        
        # Calcular la variación en la posición horizontal
        positions = np.array(self.position_history)
        horizontal_variation = np.std(positions[:, 1])
        
        # Penalizar si la variación es alta
        return -3 * (horizontal_variation / self.image_shape[1])

    def update_history(self, masks):
        road_mask = masks == 2
        current_center = center_of_mass(road_mask)
        self.position_history.append(current_center)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
