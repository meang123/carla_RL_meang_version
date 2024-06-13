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

log_dir ="after_tensorboard"
os.makedirs(log_dir, exist_ok=True)

algo_param={
    "SAC": dict(
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
    ),
    "PPO": dict(
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    )
}
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timestep to train for")
    parser.add_argument("--reload_model", type=bool, default=False, help="Path to a model to reload")
    parser.add_argument("--num_checkpoints", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()

    total_timesteps = args.total_timesteps
    seed = args.seed
    load_model = args.reload_model
    env = CarlaEnv(town="Town02")

    #n_actions = env.action_space.shape[-1]
    #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    #model = SAC('MultiInputPolicy', env,verbose=1, seed=seed, tensorboard_log=log_dir, device='cuda',**algo_param["SAC"])

    #model = TD3('MultiInputPolicy', env, action_noise=action_noise,verbose=1, seed=seed, tensorboard_log=log_dir,device='cuda')
    model = SAC.load("/home/ad05/meang_rl_carla/RL_algorithm_pycharm (2)/RL_algorithm_pycharm/tensorboard/stablebaseline3_SAC_/model_300000_steps.zip",env=env, verbose=1, seed=seed, tensorboard_log="tensorboard",device='cuda')

    model_name = "stablebaseline3_SAC_"

    model_dir = os.path.join(log_dir, model_name)
    new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps,
                callback=[TensorboardCallback(1), CheckpointCallback(
                    save_freq=total_timesteps // args.num_checkpoints,
                    save_path=model_dir,
                    name_prefix="model")], reset_num_timesteps=False)

if __name__ == "__main__":
    try:
        main()
        print("im in 22 ")
    except KeyboardInterrupt:
        print("runner falied")
        sys.exit()