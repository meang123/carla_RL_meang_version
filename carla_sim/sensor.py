import math
import time
import weakref  # 약한 참조는 메모리 관리와 관련된 문제를 해결하거나
                # 캐시와 같은 메모리 민감한 애플리케이션에서 유용하게 사용될 수 있습니다.
from carla_sim.setting import *
import pygame
import numpy as np
import carla
# Segmentation camera
class CameraSensor():

    def __init__(self,vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image:CameraSensor._get_front_camera_data(weak_self,image))

    # vehicle에 센서 부착 하는 함수
    def _set_camera_sensor(self,world):
        # segment camera buleprint
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return front_camera


    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        image.convert(carla.ColorConverter.CityScapesPalette)
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]


        array = array.copy()

        self.front_camera.append(array)



"""
이 코드는 차량 시뮬레이션에서 실시간으로 카메라 영상을 처리하고 화면에 출력하는 기능을 제공합니다.
 Placeholder는 NumPy 배열을 재구성하고 변환하는 중간 단계를 나타내며, 
이미지 데이터를 올바른 형식으로 변환하는 역할을 합니다.
"""
# ---------------------------------------------------------------------|
# ------------------------------- ENV CAMERA | RGB CAMERA
# ---------------------------------------------------------------------|




class CameraSensor_RGB:

    def __init__(self, vehicle):

        #pygame.init()
        #self.display = pygame.display.set_mode((160, 80),pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        #self.surface = None
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor_RGB._get_third_person_camera(weak_self, image))

    # Third camera is setup and provide the visual observations for our environment.

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'160')
        thrid_person_camera_bp.set_attribute('image_size_y', f'80')
        thrid_person_camera_bp.set_attribute("fov",f"110")
        #time.sleep(0.0001)
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):

        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        array = array.copy()

        self.front_camera.append(array)
        # self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        # self.display.blit(self.surface, (0, 0))
        # pygame.display.flip()



# ---------------------------------------------------------------------|
# ------------------------------- COLLISION SENSOR|
# ---------------------------------------------------------------------|

# It's an important as it helps us to tract collisions
# It also helps with resetting the vehicle after detecting any collisions
class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        #time.sleep(0.0001)
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        time.sleep(1)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)




