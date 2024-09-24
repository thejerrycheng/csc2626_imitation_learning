import gym
from full_state_car_racing_env import FullStateCarRacingEnv
import scipy
import scipy.misc
import imageio
import numpy as np
import argparse
import torch
import os

from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

def run(steering_network, args):
    # Initialize the car racing environment
    env = FullStateCarRacingEnv()
    state = env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None

    cumulative_cross_track_error = 0  # Initialize cumulative cross-track error

    for t in range(args.timesteps):
        env.render()
        # Get the current state and expert action from the environment
        state, expert_action, reward, done, info = env.step(learner_action)
        if done:
            break

        # Track cross-track error
        cross_track_error = info.get('cross_track_error', 0.0)
        cumulative_cross_track_error += abs(cross_track_error)

        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]    # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]

        if args.expert_drives:
            learner_action[0] = expert_steer
        else:
            # Get the learner's action using the steering network
            learner_action[0] = steering_network.predict(state, device=DEVICE)

        # Set throttle and brake according to expert's action
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        # Save expert demonstration images if requested
        if args.save_expert_actions:
            image_filename = os.path.join(args.out_dir, f'expert_{args.run_id}_{t}_{expert_steer}.jpg')
            imageio.imwrite(image_filename, state)

    env.close()
    return cumulative_cross_track_error

def collect_expert_demos(learner, iteration, data_dir='./dagger_dataset/train'):
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Initialize the car racing environment
    env = FullStateCarRacingEnv()
    state = env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None

    timestep = 0
    done = False
    
    while not done:
        # Get the current state and expert action from the environment
        state, expert_action, reward, done, info = env.step(learner_action)

        # The learner's policy decides the next action
        learner_action[0] = learner.predict(state, device=DEVICE)
        learner_action[1] = expert_action[1]  # Maintain the expert throttle
        learner_action[2] = expert_action[2]  # Maintain the expert brake

        # Save the state image and expert steering command
        image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{expert_action[0]}.jpg'
        imageio.imwrite(image_filename, state)

        timestep += 1
    
    env.close()

def evaluate_cross_track_error(learner):
    env = FullStateCarRacingEnv()
    state = env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    cumulative_cross_track_error = 0
    timestep = 0
    done = False
    
    while not done:
        state, expert_action, reward, done, info = env.step(learner_action)

        # Get the learner's steering command
        learner_action[0] = learner.predict(state, device=DEVICE)
        learner_action[1] = expert_action[1]  # Maintain the expert throttle
        learner_action[2] = expert_action[2]  # Maintain the expert brake

        # Track the cross-track error
        cross_track_error = info.get('cross_track_error', 0.0)
        cumulative_cross_track_error += abs(cross_track_error)
        timestep += 1

    env.close()
    return cumulative_cross_track_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", help="directory in which to save the expert's data", default='./dagger_dataset/train')
    parser.add_argument("--save_expert_actions", type=str2bool, help="save the images and expert actions in the training set",
                        default=False)

    parser.add_argument("--expert_drives", type=str2bool, help="should the expert steer the vehicle?", default=False)
    parser.add_argument("--run_id", type=int, help="Id for this particular data collection run (e.g. dagger iterations)", default=0)
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run, up to one full loop of the track", default=100000)
    parser.add_argument("--learner_weights", type=str, help="filename from which to load learner weights for the steering network",
                        default='')
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)

    args = parser.parse_args()

    steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    if args.learner_weights:
        steering_network.load_weights_from(args.learner_weights)

    cumulative_cross_track_error = run(steering_network, args)
    print(f'Cumulative Cross-Track Error: {cumulative_cross_track_error}')