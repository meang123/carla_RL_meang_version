import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_sim.environment import *
from callback_function import *
import argparse
import sys

def run_eval(env, model, model_path=None, record_video=False):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))

    # vec_env = model.get_env()
    state = env.reset()
    rendered_frame = env.render(mode="rgb_array")

    # Init video recording
    if record_video:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0
    # While non-terminal state
    print("Episode ", episode_idx)
    saved_route = False
    while episode_idx < 4:
        env.extra_info.append("Evaluation")
        action, _states = model.predict(state, deterministic=True)
        state, reward, dones, info = env.step(action)
        if env.step_count >=6000000 and env.current_waypoint_index == 0:
            dones = True


        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if record_video:
            video_recorder.add_frame(rendered_frame)
        if dones:
            state = env.reset()
            episode_idx += 1
            saved_route = False
            print("Episode ", episode_idx)

    # Release video
    if record_video:
        video_recorder.release()

def main():
    model_path="/home/ad05/meang_rl_carla/RL_algorithm_pycharm (2)/RL_algorithm_pycharm/tensorboard/stablebaseline3_PPO_/model_200000_steps.zip"
    env = CarlaEnv(town="Town02",eval=True)

    model = PPO.load(model_path,env=env, verbose=1,device='cuda')
    run_eval(env, model, model_path, record_video=True)

if __name__ == "__main__":
    try:
        main()
        print("im in 22 ")
    except KeyboardInterrupt:
        print("runner falied")
        sys.exit()