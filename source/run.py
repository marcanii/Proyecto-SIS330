import random
import time
from RobotController.MotorController import MotorController

if __name__ == '__main__':
    motor_controller = MotorController()
    motor_controller.setupMotors()
    while True:
        speed = random.uniform(0.4, 0.8)
        direction = random.randrange(0, 10)

        if direction == 0:
            motor_controller.forward(speed)
        elif direction == 1:
            motor_controller.backward(speed)
        elif direction == 2:
            motor_controller.left()
        elif direction == 3:
            motor_controller.right()
        elif direction == 4:
            motor_controller.diagonalForwardLeft(speed)
        elif direction == 5:
            motor_controller.diagonalForwardRight(speed)
        elif direction == 6:
            motor_controller.diagonalBackwardLeft(speed)
        elif direction == 7:
            motor_controller.diagonalBackwardRight(speed)
        elif direction == 8:
            motor_controller.stop()
        elif direction == 9:
            motor_controller.turnLeft(speed)
        elif direction == 10:
            motor_controller.turnRight(speed)
        
        time.sleep(1)

        if KeyError == 27 or KeyError == ord('q'):
            break
        
    print("Programa finalizado...")
        