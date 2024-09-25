import gym
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import random  # Import the random module for environment randomization
from full_state_car_racing_env import FullStateCarRacingEnv
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE

def visualize_dagger_policy(args):
    # Create directories for saving visualizations if they don't exist
    if not os.path.exists(args.visualize_dir):
        os.makedirs(args.visualize_dir)

    # Load the current policy from the specified weights
    learner = DiscreteDrivingPolicy(n_classes=20).to(DEVICE)  # Ensure n_classes matches the trained model
    learner.load_weights_from(args.weights)

    # Initialize the car racing environment with a random seed
    seed = random.randint(0, 10000)  # Generate a random seed
    env = FullStateCarRacingEnv(seed=seed)  # Pass the random seed to the environment
    print(f"Initialized environment with random seed: {seed}")

    state = env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    timestep = 0
    done = False
    current_time = 0.0

    # PID history reset
    env.orientation_pid.erase_history()
    env.distance_pid.erase_history()
    env.gas_pid.erase_history()

    while not done and timestep < args.num_timesteps:
        env.render()

        # Calculate cross-track error
        _, cross_track_error, _ = env.get_cross_track_error(env.car, env.track)

        # Determine whether to use the expert or learner policy based on cross-track error
        if abs(cross_track_error) > args.cte_threshold:
            # Use the expert action if the car is deviating too much
            expert_action, eh, cte = env.get_expert_action(env.car, env.track)
            learner_action = expert_action
            print(f"Timestep {timestep}: Switching to expert policy due to deviation (CTE: {cross_track_error:.2f}).")

            # Update PID controllers
            current_time += 1.0 / env.metadata.get("render_fps", 50)
            if len(env.orientation_pid.timestamps_of_errors) >= 1:
                last_timestamp = env.orientation_pid.timestamps_of_errors[-1]
                if current_time > last_timestamp:
                    env.orientation_pid.update(eh, current_time)
                    env.distance_pid.update(cte, current_time)
                    env.gas_pid.update(eh, current_time)
            else:
                env.orientation_pid.update(eh, current_time)
                env.distance_pid.update(cte, current_time)
                env.gas_pid.update(eh, current_time)
        else:
            # Use the learner's policy
            learner_action[0] = learner.predict(state, DEVICE)
            if abs(learner_action[0]) <= 0.06:
                learner_action[0] = 0
            learner_action[1] = expert_action[1] if expert_action is not None else 0.1
            learner_action[2] = expert_action[2] if expert_action is not None else 0.0

        # Print the action taken by the learner
        print(f"Timestep {timestep}: Using learner policy. The cross track error is {cross_track_error:.2f}, learner prediction is {learner_action[0]:.2f}")

        # Take a step in the environment
        next_state, _, reward, done, info = env.step(learner_action)

        # Save the frame as an image
        image_filename = os.path.join(args.visualize_dir, f'frame_{timestep}.jpg')
        imageio.imwrite(image_filename, state)

        state = next_state
        timestep += 1

    env.close()
    print(f"Visualization frames saved in {args.visualize_dir}")

    # Create a video from the saved frames
    create_video(args.visualize_dir, 'dagger_visualization.mp4')
    print('Video visualization saved as dagger_visualization.mp4')


def create_video(image_dir, output_video_path, fps=30):
    # Get list of images
    image_files = sorted([img for img in os.listdir(image_dir) if img.endswith(".jpg")])
    frame_list = []
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        frame = imageio.imread(image_path)
        frame_list.append(frame)
    
    # Save the video
    imageio.mimwrite(output_video_path, frame_list, fps=fps)
    print(f"Video created at {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="Path to the weights of the network", required=True)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--num_timesteps", type=int, help="Total number of timesteps to run", default=1000)
    parser.add_argument("--learner_interval", type=int, help="Number of timesteps for which the learner policy runs", default=20)
    parser.add_argument("--expert_interval", type=int, help="Number of timesteps to switch between learner and expert", default=100)
    parser.add_argument("--visualize_dir", help="Directory to save visualization frames", default='./visualize_frames')
    parser.add_argument("--cte_threshold", type=float, help="Cross-track error threshold to switch to expert policy", default=3.5)
    args = parser.parse_args()

    visualize_dagger_policy(args)

# import gym
# import argparse
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
# import os
# from full_state_car_racing_env import FullStateCarRacingEnv
# from driving_policy import DiscreteDrivingPolicy
# from utils import DEVICE

# def visualize_dagger_policy(args):
#     # Create directories for saving visualizations if they don't exist
#     if not os.path.exists(args.visualize_dir):
#         os.makedirs(args.visualize_dir)

#     # Load the current policy from the specified weights
#     learner = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
#     learner.load_weights_from(args.weights)

#     # Initialize the car racing environment
#     env = FullStateCarRacingEnv()
#     state = env.reset()

#     learner_action = np.array([0.0, 0.0, 0.0])
#     expert_action = None
#     timestep = 0
#     done = False
#     current_time = 0.0

#     # PID history reset
#     env.orientation_pid.erase_history()
#     env.distance_pid.erase_history()
#     env.gas_pid.erase_history()

#     while not done and timestep < args.num_timesteps:
#         env.render()

#         # Calculate cross-track error
#         _, cross_track_error, _ = env.get_cross_track_error(env.car, env.track)

#         # Determine whether to use the expert or learner policy based on cross-track error
#         if abs(cross_track_error) > args.cte_threshold:
#             # Use the expert action if the car is deviating too much
#             expert_action, eh, cte = env.get_expert_action(env.car, env.track)
#             learner_action = expert_action
#             print(f"Timestep {timestep}: Switching to expert policy due to deviation (CTE: {cross_track_error:.2f}).")

#             # Update PID controllers
#             current_time += 1.0 / env.metadata.get("render_fps", 50)
#             if len(env.orientation_pid.timestamps_of_errors) >= 1:
#                 last_timestamp = env.orientation_pid.timestamps_of_errors[-1]
#                 if current_time > last_timestamp:
#                     env.orientation_pid.update(eh, current_time)
#                     env.distance_pid.update(cte, current_time)
#                     env.gas_pid.update(eh, current_time)
#             else:
#                 env.orientation_pid.update(eh, current_time)
#                 env.distance_pid.update(cte, current_time)
#                 env.gas_pid.update(eh, current_time)
#         else:
#             # Use the learner's policy
#             learner_action[0] = learner.predict(state, DEVICE)
#             learner_action[1] = expert_action[1] if expert_action is not None else 0.5
#             learner_action[2] = expert_action[2] if expert_action is not None else 0.0
#             print(f"Timestep {timestep}: Using learner policy.")

#         # Take a step in the environment
#         next_state, _, reward, done, info = env.step(learner_action)

#         # Save the frame as an image
#         image_filename = os.path.join(args.visualize_dir, f'frame_{timestep}.jpg')
#         imageio.imwrite(image_filename, state)

#         state = next_state
#         timestep += 1

#     env.close()
#     print(f"Visualization frames saved in {args.visualize_dir}")

#     # Create a video from the saved frames
#     create_video(args.visualize_dir, 'dagger_visualization.mp4')
#     print('Video visualization saved as dagger_visualization.mp4')


# def create_video(image_dir, output_video_path, fps=30):
#     # Get list of images
#     image_files = sorted([img for img in os.listdir(image_dir) if img.endswith(".jpg")])
#     frame_list = []
    
#     for filename in image_files:
#         image_path = os.path.join(image_dir, filename)
#         frame = imageio.imread(image_path)
#         frame_list.append(frame)
    
#     # Save the video
#     imageio.mimwrite(output_video_path, frame_list, fps=fps)
#     print(f"Video created at {output_video_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", help="Path to the weights of the network", required=True)
#     parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
#     parser.add_argument("--num_timesteps", type=int, help="Total number of timesteps to run", default=1000)
#     parser.add_argument("--cte_threshold", type=float, help="Cross-track error threshold for switching to expert policy", default=5.0)
#     parser.add_argument("--visualize_dir", help="Directory to save visualization frames", default='./visualize_frames')
#     args = parser.parse_args()

#     visualize_dagger_policy(args)