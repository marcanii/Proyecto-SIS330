import threading
import requests

class AutonomousDriving:
    def __init__(self):
        self.url_inference = "https://dhbqf30k-5000.brs.devtunnels.ms/inference"
        self.is_running = False
        self.thread = None

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        while self.is_running:
            response = requests.post(self.url_inference)
            action = response.json()['action']
            