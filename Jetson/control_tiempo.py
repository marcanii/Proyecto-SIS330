from Agent import Agent
import time

if __name__ == "__main__":
    agent = Agent()
    if agent.loadModels():
        print("Modelos cargados correctamente...")
    else:
        print("Error al cargar los modelos...")

    while True:
        initial = time.time()
        state, _, _ = agent.observation()
        action, _, _ = agent.chooseAction(state)
        agent.takeAction(action)
        final = time.time() - initial
        if 2.0 < final < 4.0:
            print("Tiempo:", final, "CORRECTO")
        else:
            print("Tiempo:", final, "INCORRECTO")
            break