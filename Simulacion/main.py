import pybullet as p
import time
import pybullet_data

# Conectar al motor de física de PyBullet
physicsClient = p.connect(p.GUI)

# Configurar el motor de física
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Cargar un modelo de caja
plane = p.loadURDF("plane.urdf")
boxId = p.loadURDF("r2d2.urdf")

# Posicionar la caja en el espacio
p.resetBasePositionAndOrientation(boxId, [0, 0, 1], [0, 0, 0, 1])

# Simular la caída de la caja
while True:
    p.stepSimulation()
    time.sleep(1./240.)

# Desconectar del motor de física de PyBullet
p.disconnect()
