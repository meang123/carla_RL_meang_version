"""


HOST = "localhost"
PORT = 2000
TIMEOUT = 60.0 # 1 분

CAR_NAME = 'model3'
EPISODE_LENGTH = 120
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = True


RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'



continuous action space만 고려 할것이다
"""
import time
import random
import numpy as np
import pygame
import gymnasium
import gym
from gym.utils import seeding
import torch

from carla_sim.sensor import CameraSensor, CollisionSensor,CameraSensor_RGB
from carla_sim.setting import *
import os
from vae.vae_model import VAE
from torchvision import transforms
from carla_sim.hud import HUD


def load_vae():
    model_dir = os.path.join('/home/ad05/meang_rl_carla/RL_algorithm_pycharm (2)/RL_algorithm_pycharm/vae', 'best.tar')
    model = VAE(LATENT_DIM)
    if os.path.exists(model_dir):
        state = torch.load(model_dir)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        return model
    raise Exception("Error - VAE model does not exist")

# numpy -> tensor include 정규화 ToTensor 사용 이유
def preprocess_frame(frame):

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess(frame).unsqueeze(0)
    return frame




class CarlaEnv(gym.Env):
    def __init__(self,town,checkpoint_frequency=100,eval=False):

        self.fps=FPS
        self.episode_idx = -2
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(TIMEOUT)
        self.town = "Town02" # town

        self.world = self.client.load_world(self.town)


        self.world.set_weather(carla.WeatherParameters.CloudyNoon)

        self.observation_space = self.create_observation_space()
        self.action_space = self.create_action_space()


        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.vae = load_vae()

        self.total_reward = 0
        self.reward = 0
        self.step_count=0
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.setting = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.initial_start = True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.eval = eval


        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        #-----



        # observation -> vae,steer,throttle nomalize velocity, normalize distance, normalize angle
    # compute angle,distance ,speed steer,throttle ---> final action steer throttle
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]
    def create_observation_space(self):
        try:
            observation_space={}
            observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(LATENT_DIM,), dtype=np.float32)
            # steer,throttle ,velocity, normalize distance,  angle
            observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0, 0.0, -3.14]), high=np.array([1.0, 1.0, 120.0, 1.0, 3.14]),dtype=np.float32)

            return gym.spaces.Dict(observation_space)




        except Exception as e:

            print(f"observation_space setting problem!!!!  {e}")

            raise NotImplementedError()


    def create_action_space(self):
        try:
            return gym.spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]),dtype=np.float32)
        except Exception as e:
            print(f"action_space problem!!!!  {e}")

            raise NotImplementedError()


    #Clean up method
    # def remove_sensors(self):
    #     self.camera_obj = None
    #     self.collision_obj = None
    #     self.lane_invasion_obj = None
    #     self.env_camera_obj = None
    #     self.front_camera = None
    #     self.collision_history = None
    #     self.wrong_maneuver = None


    def remove_sensors(self):
        try:
            if self.camera_obj and self.camera_obj.sensor.is_alive:
                self.camera_obj.sensor.stop()
                self.camera_obj.sensor.destroy()
                self.camera_obj = None
            if self.collision_obj and self.collision_obj.sensor.is_alive:
                self.collision_obj.sensor.stop()
                self.collision_obj.sensor.destroy()
                self.collision_obj = None
            # if self.lane_invasion_obj and self.lane_invasion_obj.sensor.is_alive:
            #     self.lane_invasion_obj.sensor.stop()
            #     self.lane_invasion_obj.sensor.destroy()
            #     self.lane_invasion_obj = None
            # if self.env_camera_obj and self.env_camera_obj.sensor.is_alive:
            #     self.env_camera_obj.sensor.stop()
            #     self.env_camera_obj.sensor.destroy()
            #     self.env_camera_obj = None

        except Exception as e:
            print(f"Error while removing sensors: {e}")

    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


# 두 벡터 사이의 각도 계산
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle

# 점과 선분간의 거리 계산
    def distance_to_line(self, A, B, p):
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    """
    이 메소드는 CARLA의 위치, 회전 벡터를 numpy 배열로 변환합니다. 
    이는 벡터 연산을 쉽게 하기 위해 사용됩니다.
    """

    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])



    def reset(self):
        time.sleep(1)
        try:


            if len(self.actor_list)!=0:

                #self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                for actor in self.actor_list:
                    if actor.is_alive:

                        actor.destroy()

                time.sleep(0.2)

                self.actor_list.clear()

            if len(self.sensor_list)!=0:
                # list안에 있는 내용들 다 없에기

                #self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                for sensor in self.sensor_list:
                    if sensor.is_alive:

                        sensor.destroy()
                    time.sleep(0.2)

                # list 없에기
                self.sensor_list.clear()


            self.remove_sensors()
            vehicle_bp = self.get_vehicle(CAR_NAME)


            # Town2 기준
            transform = self.map.get_spawn_points()[1] # Town 2 is 1
            self.total_distance = 780 # 차량이 주행할 총 길이 설정

    #다른 town설정
            # elif:
            #     pass




            # town2 위치에 차량 bp spawn 하기
            self.vehicle = self.world.try_spawn_actor(vehicle_bp,transform)
            #time.sleep(1)

            self.actor_list.append(self.vehicle) # 차량 actor 뿐아니라 다른 actor들도 이 리스트에 들어감



            # initialize pygame for visualization
            if self.display_on:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((1120, 560), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(1120, 560)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)




# ------------------------------------------------------------------------
            self.extra_info =[]
            self.total_reward=0
            self.episode_idx += 1
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw # 차량의 방향 주행중 방향을 추적하는 역할
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0

            # speed min max target
            self.target_speed = 30  # km/h
            self.max_speed = 50.0
            self.min_speed = 10.0

            self.max_distance_from_center = 3

            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0


            # 초기 상태에서 시작 하는 경우
            if self.initial_start:
                self.current_waypoint_index = 0
                self.route_waypoints = list() # waypoint 경로를 생성하고 싶기 때문에 이를 위한 리스트
                # 현재 차량 기준으로 가장 가까운 waypoint 찾는 코드 -> 도로 위에서 벗어나지 않기 위해서임
                # project to road  : 도로 벗어남에 관계 없이 가장 가까운 waypoint반환
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)

# total distance에 따른 way point 생성
                for x in range(self.total_distance):

                    if self.town=='Town02':
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1] # next(1.0) 현재 waypoint에서 1m앞에 waypoint 찾는다
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    # elif other town need to define total distance

                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint


            else:
                # teleport vehicle to last checkpoint
                # step 함수에서 checkpoint가 계산된다
                if self.checkpoint_waypoint_index !=0:
                    print(f"check point activated {self.checkpoint_waypoint_index}")

                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            # # ----------------------Setting RGB camera sensor (VAE is rgb to seg pretrained model)---------------------------------------------



            self.camera_obj = CameraSensor_RGB(self.vehicle)#CameraSensor(self.vehicle)
            while (len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            self.image_obs = self.camera_obj.front_camera.pop(-1) # 가장 마지막 segmentation image
                # print(f"IM DEBUGER in reset {self.image_obs},-----------")
            self.sensor_list.append(self.camera_obj.sensor)  # segmentation camera 추가됨

            # Collision sensor

            self.collision_obj = CollisionSensor(self.vehicle)


            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)


            """
            주행 성능 평가 하고 reward 계산하는데 사용된다 
            경로 이탈 충돌등에 따라 에피소드 종료하는데 사용됨 
            """

            nav_obs=[self.previous_steer,self.throttle,self.velocity,self.distance_from_center,self.angle]


            self.collision_history.clear()

            self.episode_start_time = time.time()

            #observation = self.encoded_state(nav_obs)
            # observation = {'vae_latent': observation['vae_latent'],
            #                'vehicle_measures': observation['vehicle_measures']}
            return self.encoded_state(nav_obs) # observation 즉 sate임


        except Exception as e:
            print(f"Reset part problem {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def step(self,action_idx):
        try:
            self.timesteps+=1
            self.initial_start=False

            velocity = self.vehicle.get_velocity() # 차량 속도 얻기
            self.velocity = np.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)*3.6 # covert km/h

            #self.observation = self._get_observation()
            #only deal with continuous action space no option for discrite action space

            steer = float(action_idx[0])
            steer = max(min(steer, 1.0), -1.0)
            throttle = float((action_idx[1] + 1.0) / 2)
            throttle = max(min(throttle, 1.0), 0.0)

            # 이전 값을 90%유지 새로운 값을 10%유지 한게 appy control한것이다
            self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.9 + steer * 0.1,throttle=self.throttle * 0.9 + throttle * 0.1))
            self.previous_steer = steer
            self.throttle = throttle


            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data # list

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()

            self.distance_traveled += self.previous_location.distance(self.location)
            self.previous_location = self.location

            # keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index


            """
            wp : 다음 waypoint 객체 
            dot 연산은 차량이 다음 waypoint를 지나쳤는지 여부를 결정하는데 사용된다 
                : 차량의 위치 벡터와 웨이포인트의 전방 벡터간의 내적 계산 
                만약 다음 웨이포인트 지났으면 waypoint_index +1 한다 
                
            """
            for _ in range(len(self.route_waypoints)):
                # 다음 waypoint 설정
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]

                """
                get_forward_vector()는 waypoint의 전방 벡터를 반환 
                self.location- wp.transform.location은 웨이포인트에서 차량 위치로 향하는 벡터 계산 
                
                다음 웨이포인트 전방 벡터와 이전 웨이포인트에서 차량 위치 향하는 벡터가 서로 같은 방향이면 내적 양수이다 
                --> 다음 웨이포인트로 잘 넘어가는지 추적하는 기능이다 
                """
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                             self.vector(self.location - wp.transform.location)[:2])


                if dot > 0.0:
                    waypoint_index+=1
                else:
                    break

            self.current_waypoint_index = waypoint_index # 업데이트



            # Calculate deviation from center of the lane
            """
            차량이 차선 중앙으로부터 얼마나 벗어났는지 계산 
            self.route_waypoints:
            
            경로를 구성하는 웨이포인트의 리스트입니다. 이 리스트는 초기화 시에 설정되며, 차량이 주행할 경로를 정의합니다.
            self.current_waypoint_index:
            
            차량이 현재 경로에서 위치한 웨이포인트의 인덱스입니다. 이 인덱스는 차량의 진행 상황에 따라 업데이트됩니다.
            self.current_waypoint:
            
            self.route_waypoints 리스트에서 현재 웨이포인트를 나타냅니다.
            self.current_waypoint_index를 사용하여 리스트에서 현재 웨이포인트를 가져옵니다.
            self.next_waypoint:
            
            self.route_waypoints 리스트에서 다음 웨이포인트를 나타냅니다.
            self.current_waypoint_index + 1을 사용하여 리스트에서 다음 웨이포인트를 가져옵니다.
            
            
            self.distance_from_center:
            
            차량이 현재 경로의 중앙으로부터 얼마나 벗어나 있는지를 나타내는 값입니다. 이 값은 self.distance_to_line() 메소드를 사용하여 계산됩니다.
            
            현재 웨이포인트 위치와 다음 웨이포인트 위치와 차량의 현재 위치
            
            
            
            cf : 현재 웨이포인트와 차량의 위치는 일반적으로 일치하지 않습니다.
            현재 차량 위치 점과  현 웨이포인트와 다음 웨이포인트의 선분간의 거리 계산한다 
            
            
            self.center_lane_deviation:
            
            주행 동안 누적된 차선 중심으로부터의 편차를 저장합니다. 이는 전체 주행 중 차량의 경로 이탈을 평가하는 데 사용됩니다.
            
            """

            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]

            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))

            self.center_lane_deviation += self.distance_from_center


            """
            차량의 현재 진행 방향과 경로 상의 웨이포인트 방향간의 각도 차이 계산하는 역할 
            -> 차량이 올바른 방향으로 가는지 확인 할수있다 
            
            fwd : 차량의 현재 속도 벡터 
            wp_fwd : 현재 웨이포인트의 전방 방향 벡터 (현 웨이포인트 진행방향)
            self.angle : 차량의 진행 방향과 현재 웨이포인트의 전방 방향간의 각도 차이 나타낸다 
                        -> 차량의 진행 방향이 경로의 방향과 얼마나 일치 하는지 평가 한다 
                        
            """

            # Get angle difference between closest waypoint and vehicle forward vector

            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd) # 두 벡터간의 각도 차이 계산


            """
            current_waypoint_index가 250이고, checkpoint_frequency가 100이라면:
            (250 // 100) * 100 = 2 * 100 = 200
            즉, 200번째 웨이포인트가 체크포인트가 됩니다.
            
            -> 주행중 일정 간격으로 설정된 체크포인트로 돌아갈수있도록 하기 위함이다 
            """
            if not self.initial_start and self.checkpoint_frequency is not None:
                self.checkpoint_waypoint_index = (self.current_waypoint_index//self.checkpoint_frequency)* self.checkpoint_frequency




            # Reward 계산

            done =False
            #reward =0
            self.reward=0

            time.sleep(0.00002)
            # Interpolated from 1 when centered to 0 when 3 m from center
            # 차량이 차선 중심으로부터 얼마나 벗어나 있는지 나타낸다
            # 경로 중심에서 벗어날수록 작아진다
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)

            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            # 차량의 진행 방향이 경로의 방향과 얼마나 일치 하는지 나타내는 보상 요소
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0) # np.deg2rad(20)






            time.sleep(0.0002)

            distanced_factor = (self.distance_traveled/100)*0.6

            # Reward 증가
            if not done:
                # 속도가 느리면 속도 높이도록 보상 줄인다
                if self.velocity < self.min_speed:
                    self.reward += (self.velocity / self.min_speed) * centering_factor * angle_factor#*distanced_factor


                # 차량속도가 목표 속도보다 높으면 속도를 낮추도록 보상 줄인다
                elif self.velocity > self.target_speed:
                    self.reward += (1.0 - ((self.velocity - self.target_speed) / (self.max_speed - self.target_speed)))*centering_factor * angle_factor#*distanced_factor
                    #self.reward += distanced_factor
                # min~ target의 범위에 있으면 보상을 최대화 한다
                else:
                    self.reward += 1.0 * centering_factor * angle_factor
                    #self.reward += distanced_factor
                # positive reward에 대해서만 계산
                self.reward += distanced_factor

            time.sleep(0.00002)
            # 충돌시
            if len(self.collision_history)!=0:
                done =True
                self.reward = -200
                print("COLLISSION!\n")

            # center 벗어난 경우
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                self.reward = -200
                print("OFF TRACK!\n")

            # 차량이 10초의 시간동안 거의 움직이지 않을때 에피소드 종료 하는것이다
            # episode_start_time + 10은 시작하고 10초 지난 시점을 의미한다
            elif self.episode_start_time+10 < time.time() and self.velocity <1.0:
                done =True
                self.reward = -200
                print("not move in time!\n")

            # 속도가 최고 속도보다 빠른 경우
            elif (self.velocity > self.max_speed) and not self.eval:
                self.reward = -180
                done = True
                print("TOO FAST!\n")

            #self.reward += distanced_factor

            self.total_reward += self.reward



            # 시뮬레이션 단계가 7500넘으면 종료
            if self.timesteps >= 7500:
                done = True

            # 현재 waypoint가 경로 waypoint 끝에 다다르면 사실상 경로가 끝났음을 의미한다
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.initial_start = True
                if self.checkpoint_frequency is not None:
                    # 체크 포인트 빈도가 전체 거리의 절반보다 작은 경우 체크 포인트 빈도 증가
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0


            #
            # # # 이미지 데이터 확보 하는 작업
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)


            self.image_obs = self.camera_obj.front_camera.pop(-1)
            #


            normalized_velocity = self.velocity / self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20)) #np.deg2rad(20)

            # 정규화한 값으로 업데이트
            navigation_obs = [self.previous_steer, self.throttle, normalized_velocity, normalized_distance_from_center, normalized_angle]



            encoded_state = self.encoded_state(navigation_obs)
            time.sleep(0.0002)
            self.observation_decode = self.decode_vae_state(encoded_state['vae_latent'])


            self.step_count+=1
            if self.display_on:
                pygame.event.pump()

                self.render()

            # Remove everything that has been spawned in the env
            if done:
                # 누적된 차선 중심으로부터의 편차를 주행 시간으로 나누어 평균 편차 구한다 -> 다음 학습에 사용하기 위해 계싼
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps

                # 주행 동안 커버한 거리(웨이포인트 수)를 계산 -> 실제로 얼마나 주행하였는지 평가 할수있다

                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)

                for sensor in self.sensor_list:
                    sensor.destroy()

                self.remove_sensors()

                for actor in self.actor_list:
                    actor.destroy()


            info = {
                "reward" : self.reward,
                "total_reward" : self.total_reward,
                "mean_reward":self.total_reward/self.step_count,
                "center_lane_deviation":self.center_lane_deviation,
                "distance_covered": self.distance_covered


            }
            #self.total_reward
            return encoded_state, self.reward, done, info

        except Exception as e:
            print(f"Step part problem {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()



    def render(self, mode="human"):
        if mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])

        self.clock.tick()
        self.hud.tick(self.world,self.clock)

        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.reward,
            f"steer : {self.previous_steer}",
            f"throttle : {self.throttle}",
            f"velocity : {self.velocity}",


            "Distance traveled: % 7d m" % self.distance_traveled,
            "normalized Center deviance:   % 7.2f m" % (self.distance_from_center/self.max_distance_from_center),
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation),
            "distance_covered:      % 7.2f km/h" % (self.distance_covered),
            "distance:        % 7.2f" % self.distance_traveled,
        ])
        obs_h, obs_w = self.image_obs.shape[:2]


        # obs render position
        pos_observation = (500,250)
        self.display.blit(pygame.surfarray.make_surface(self.image_obs.swapaxes(0, 1)), pos_observation)

        # vae render position
        pos_vae_decoded = (self.display.get_size()[0] - 2 * obs_w - 10, 10)

        self.display.blit(pygame.surfarray.make_surface(self.observation_decode.swapaxes(0, 1)), pos_vae_decoded)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()


    # Reward 계산까지 다하고 나서 호출 가능
    # normalized value list
    def encoded_state(self,nav_obs):

        encoded_state = {}
        with torch.no_grad():
            frame = preprocess_frame(self.image_obs) # self.image_obs
            mu, logvar = self.vae.encode(frame)
            vae_latent = self.vae.reparameterize(mu, logvar)[0].cpu().detach().numpy().squeeze()

        encoded_state['vae_latent'] = vae_latent

        vehicle_measures = []
        for measures in nav_obs:
            vehicle_measures.append(measures)

        encoded_state['vehicle_measures'] = vehicle_measures

        return encoded_state


    def decode_vae_state(self,z):

        with torch.no_grad():
            sample = torch.tensor(z)
            sample = self.vae.decode(sample).cpu()
            generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0)) * 255
        return generated_image



## 확장할수있다

# # -------------------------------------------------
# # Creating and Spawning Pedestrians in our world |
# # -------------------------------------------------
#
#     # Walkers are to be included in the simulation yet!
#     def create_pedestrians(self):
#         try:
#
#             # Our code for this method has been broken into 3 sections.
#
#             # 1. Getting the available spawn points in  our world.
#             # Random Spawn locations for the walker
#             walker_spawn_points = []
#             for i in range(NUMBER_OF_PEDESTRIAN):
#                 spawn_point_ = carla.Transform()
#                 loc = self.world.get_random_location_from_navigation()
#                 if (loc != None):
#                     spawn_point_.location = loc
#                     walker_spawn_points.append(spawn_point_)
#
#             # 2. We spawn the walker actor and ai controller
#             # Also set their respective attributes
#             for spawn_point_ in walker_spawn_points:
#                 walker_bp = random.choice(
#                     self.blueprint_library.filter('walker.pedestrian.*'))
#                 walker_controller_bp = self.blueprint_library.find(
#                     'controller.ai.walker')
#                 # Walkers are made visible in the simulation
#                 if walker_bp.has_attribute('is_invincible'):
#                     walker_bp.set_attribute('is_invincible', 'false')
#                 # They're all walking not running on their recommended speed
#                 if walker_bp.has_attribute('speed'):
#                     walker_bp.set_attribute(
#                         'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
#                 else:
#                     walker_bp.set_attribute('speed', 0.0)
#                 walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
#                 if walker is not None:
#                     walker_controller = self.world.spawn_actor(
#                         walker_controller_bp, carla.Transform(), walker)
#                     self.walker_list.append(walker_controller.id)
#                     self.walker_list.append(walker.id)
#             all_actors = self.world.get_actors(self.walker_list)
#
#             # set how many pedestrians can cross the road
#             #self.world.set_pedestrians_cross_factor(0.0)
#             # 3. Starting the motion of our pedestrians
#             for i in range(0, len(self.walker_list), 2):
#                 # start walker
#                 all_actors[i].start()
#             # set walk to random point
#                 all_actors[i].go_to_location(
#                     self.world.get_random_location_from_navigation())
#
#         except:
#             self.client.apply_batch(
#                 [carla.command.DestroyActor(x) for x in self.walker_list])
#
#
# # ---------------------------------------------------
# # Creating and Spawning other vehciles in our world|
# # ---------------------------------------------------
#
#
#     def set_other_vehicles(self):
#         try:
#             # NPC vehicles generated and set to autopilot
#             # One simple for loop for creating x number of vehicles and spawing them into the world
#             for _ in range(0, NUMBER_OF_VEHICLES):
#                 spawn_point = random.choice(self.map.get_spawn_points())
#                 bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
#                 other_vehicle = self.world.try_spawn_actor(
#                     bp_vehicle, spawn_point)
#                 if other_vehicle is not None:
#                     other_vehicle.set_autopilot(True)
#                     self.actor_list.append(other_vehicle)
#             print("NPC vehicles have been generated in autopilot mode.")
#         except:
#             self.client.apply_batch(
#                 [carla.command.DestroyActor(x) for x in self.actor_list])
#
#
# # ----------------------------------------------------------------
# # Extra very important methods: their names explain their purpose|
# # ----------------------------------------------------------------
#
#     # Setter for changing the town on the server.
#     def change_town(self, new_town):
#         self.world = self.client.load_world(new_town)
#
#
#     # Getter for fetching the current state of the world that simulator is in.
#     def get_world(self) -> object:
#         return self.world
#

























