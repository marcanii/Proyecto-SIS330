import time
import traitlets
from traitlets.config.configurable import SingletonConfigurable
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from Robot.Motor import Motor

class MotorController(SingletonConfigurable):
    left_front_motor = traitlets.Instance(Motor)
    right_front_motor = traitlets.Instance(Motor)
    left_back_motor = traitlets.Instance(Motor)
    right_back_motor = traitlets.Instance(Motor)
    
    # config
    i2c_bus = traitlets.Integer(default_value=1).tag(config=True)
    right_back_motor_channel = traitlets.Integer(default_value=1).tag(config=True)
    right_back_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    left_back_motor_channel = traitlets.Integer(default_value=4).tag(config=True)
    left_back_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    right_front_motor_channel = traitlets.Integer(default_value=2).tag(config=True)
    right_front_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    left_front_motor_channel = traitlets.Integer(default_value=3).tag(config=True)
    left_front_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(MotorController, self).__init__(*args, **kwargs)
        self.motor_driver = Adafruit_MotorHAT(i2c_bus=self.i2c_bus)
        self.left_front_motor = Motor(self.motor_driver, self.left_front_motor_channel, "LF", alpha=self.left_front_motor_alpha)
        self.right_front_motor = Motor(self.motor_driver, self.right_front_motor_channel, "RF", alpha=self.right_front_motor_alpha)
        self.left_back_motor = Motor(self.motor_driver, self.left_back_motor_channel, "LB", alpha=self.left_back_motor_alpha)
        self.right_back_motor = Motor(self.motor_driver, self.right_back_motor_channel, "RB", alpha=self.right_back_motor_alpha)

    def setupMotors(self):
        print("Configurando motores...")
        self.left_front_motor.value = 0
        self.right_front_motor.value = 0
        self.left_back_motor.value = 0
        self.right_back_motor.value = 0
                   
    def forward(self, speed=1.0):
        print("Avanzando...")
        self.left_front_motor.value = speed
        self.right_front_motor.value = speed
        self.left_back_motor.value = speed
        self.right_back_motor.value = speed

    def backward(self, speed=1.0):
        print("Retrocediendo...")
        self.left_front_motor.value = -speed
        self.right_front_motor.value = -speed
        self.left_back_motor.value = -speed
        self.right_back_motor.value = -speed

    def stop(self):
        print("Deteniendo motores...")
        self.left_front_motor.value = 0
        self.right_front_motor.value = 0
        self.left_back_motor.value = 0
        self.right_back_motor.value = 0

    def left(self, speed=1.0):
        print("Izquierda...")
        self.left_front_motor.value = -speed
        self.right_front_motor.value = speed
        self.left_back_motor.value = speed
        self.right_back_motor.value = -speed

    def right(self, speed=1.0):
        print("Derecha...")
        self.left_front_motor.value = speed
        self.right_front_motor.value = -speed
        self.left_back_motor.value = -speed
        self.right_back_motor.value = speed

    def diagonalForwardLeft(self, speed=1.0):
        print("Diagonal Adelante Izquierda...")
        self.left_front_motor.value = 0
        self.right_front_motor.value = speed
        self.left_back_motor.value = speed
        self.right_back_motor.value = 0

    def diagonalForwardRight(self, speed=1.0):
        print("Diagonal Adelante Derecha...")
        self.left_front_motor.value = speed
        self.right_front_motor.value = 0
        self.left_back_motor.value = 0
        self.right_back_motor.value = speed

    def diagonalBackwardLeft(self, speed=1.0):
        print("Diagonal Atras Izquierda...")
        self.left_front_motor.value = -speed
        self.right_front_motor.value = 0
        self.left_back_motor.value = 0
        self.right_back_motor.value = -speed

    def diagonalBackwardRight(self, speed=1.0):
        print("Diagonal Atras Derecha...")
        self.left_front_motor.value = 0
        self.right_front_motor.value = -speed
        self.left_back_motor.value = -speed
        self.right_back_motor.value = 0
    
    def turnLeft(self, speed=1.0):
        print("Girando Izquierda...")
        self.left_front_motor.value = -speed
        self.right_front_motor.value = speed
        self.left_back_motor.value = -speed
        self.right_back_motor.value = speed
    
    def turnRight(self, speed=1.0):
        print("Girando Derecha...")
        self.left_front_motor.value = speed
        self.right_front_motor.value = -speed
        self.left_back_motor.value = speed
        self.right_back_motor.value = -speed