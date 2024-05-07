import torch
from PPO.Agent import Agent
import torchvision.models as models
from RobotController.MotorController import MotorController
import time

if __name__ == '__main__':
    # agent = Agent(5, 2*60*108)
    # action, probs, value = agent.choose_action(torch.randn(1, 2, 60, 108))
    # print("Action: ", action)
    # print("Probs: ", probs)
    # print("Value: ", value)
    motor_controller = MotorController()
    motor_controller.setupMotors()
    motor_controller.turnLeft(speed=0.5)
    time.sleep(2)
    motor_controller.turnRight(speed=0.5)
    time.sleep(2)
    motor_controller.stop()