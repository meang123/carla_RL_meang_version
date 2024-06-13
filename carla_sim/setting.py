

HOST = "localhost"
PORT = 2000
TIMEOUT = 60.0 # 1 분

CAR_NAME = 'model3'
EPISODE_LENGTH = 120
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = True
FPS=15

RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'

#VAE Bottleneck
LATENT_DIM = 64 #95

import carla


class ClientConnection:
    def __init__(self,town):
        self.client = None
        self.town = town

    def setup(self):
        try:

            # conncet carla server
            self.client = carla.Client(HOST,PORT)
            self.client.set_timeout(TIMEOUT)
            self.world = self.client.load_world(self.town)
            
            self.world.set_weather(carla.WeatherParameters.ClearNoon)

        except Exception as e:
            print(f"Faile connect carla server {e}")
