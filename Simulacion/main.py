import pybullet as p
import pybullet_data
import time

# Conectar al motor de física de PyBullet
physicsClient = p.connect(p.GUI)

# Configurar el motor de física
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Cargar un modelo de caja
plane = p.loadURDF("plane.urdf")

# Cargar el modelo URDF del robot Husky
husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Configurar las articulaciones del robot Husky
num_joints = p.getNumJoints(husky)
for i in range(num_joints):
    joint_info = p.getJointInfo(husky, i)
    if joint_info[2] == p.JOINT_PRISMATIC:
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=0, force=50)
    elif joint_info[2] == p.JOINT_REVOLUTE:
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=0, force=50)

# Configurar la velocidad lineal y angular del robot Husky
vel_lineal = 0.5
vel_angular = 0

# Configurar la fuerza de las ruedas del robot Husky
forces = [0]*num_joints
for i in range(4):
    forces[i] = vel_lineal
    forces[i+4] = vel_angular

# Configurar el control de las articulaciones del robot Husky
p.setJointMotorControlArray(husky, range(num_joints), p.VELOCITY_CONTROL, forces=forces)

# Simular la caída de la caja
while True:
    p.stepSimulation()
    time.sleep(1./240.)


# Desconectar del motor de física de PyBullet
p.disconnect()
