# import train_policy
# import racer
# import argparse
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import gym
# from full_state_car_racing_env import FullStateCarRacingEnv
# import scipy
# import scipy.misc
# import imageio
# import numpy as np
# from driving_policy import DiscreteDrivingPolicy
# from utils import DEVICE, str2bool

# def run_policy_with_expert_correction(learner, expert, iteration, data_dir, num_timesteps, switch_interval=50):
#     """
#     Runs the environment using the learner's policy, but switches to the expert for corrections intermittently.
#     """
#     # Ensure the data directory exists
#     os.makedirs(data_dir, exist_ok=True)

#     # Initialize the car racing environment
#     env = FullStateCarRacingEnv()
#     state = env.reset()

#     learner_action = np.array([0.0, 0.0, 0.0])
#     expert_action = None

#     timestep = 0
#     done = False
#     current_time = 0.0

#     # Reset PID history
#     env.orientation_pid.erase_history()
#     env.distance_pid.erase_history()
#     env.gas_pid.erase_history()

#     # while not done and timestep < num_timesteps:
#     #     if timestep % (2 * switch_interval) < switch_interval:
#     #         # Use learner's policy for the current set of timesteps
#     #         learner_action[0] = learner.predict(state, DEVICE)
#     #         learner_action[1] = expert_action[1] if expert_action is not None else 0.5
#     #         learner_action[2] = expert_action[2] if expert_action is not None else 0.0
#     #     else:
#     #         # Use expert action for the current set of timesteps
#     #         expert_action, eh, cte = env.get_expert_action(env.car, env.track)

#     #         # Update the PID controllers
#     #         current_time += 1.0 / 50  # Update time by 1/FPS
#     #         if len(env.orientation_pid.timestamps_of_errors) >= 1:
#     #             last_timestamp = env.orientation_pid.timestamps_of_errors[-1]
#     #             if current_time > last_timestamp:
#     #                 env.orientation_pid.update(eh, current_time)
#     #                 env.distance_pid.update(cte, current_time)
#     #                 env.gas_pid.update(eh, current_time)
#     #         else:
#     #             env.orientation_pid.update(eh, current_time)
#     #             env.distance_pid.update(cte, current_time)
#     #             env.gas_pid.update(eh, current_time)

#     #         learner_action = expert_action  # Correct the action using the expert

#     #     # Take a step in the environment
#     #     next_state, _, reward, done, info = env.step(learner_action)

#     #     # Save the state image and the action used
#     #     steering_command = learner_action[0]
#     #     image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{steering_command}.jpg'
#     #     imageio.imwrite(image_filename, state)

#     while not done and timestep < args.num_timesteps:
#         env.render()

#         # Determine whether to use learner or expert based on the current interval
#         if (timestep // args.expert_interval) % 2 == 0 and (timestep % args.expert_interval < args.learner_interval):
#             # Use the learner's action for the learner interval
#             learner_action[0] = learner.predict(state, DEVICE)
#             learner_action[1] = expert_action[1] if expert_action is not None else 0.5
#             learner_action[2] = expert_action[2] if expert_action is not None else 0.0
#             print(f"Timestep {timestep}: Using learner policy.")
#         else:
#             # Use the expert action for the expert interval
#             expert_action, eh, cte = env.get_expert_action(env.car, env.track)
#             learner_action = expert_action
#             print(f"Timestep {timestep}: Switching to expert policy.")

#             # Increment the time by 1/FPS before calling update
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

#         # Take a step in the environment
#         next_state, _, reward, done, info = env.step(learner_action)

#         # Save the frame as an image
#         steering_command = learner_action[0]
#         image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{steering_command}.jpg'
#         imageio.imwrite(image_filename, state)

#         state = next_state
#         timestep += 1

#     env.close()
#     print(f"Visualization frames saved in {args.train_dir}")

# def dagger_algorithm(args):
#     # Create the dagger_dataset directory if it doesn't exist
#     if not os.path.exists('./dagger_dataset'):
#         os.makedirs('./dagger_dataset/train', exist_ok=True)
#         os.makedirs('./dagger_dataset/val', exist_ok=True)

#     # Initialize DAgger process
#     cross_track_errors = []  # Track cumulative cross-track error for each iteration

#     print('TRAINING LEARNER ON INITIAL DATASET')
#     args.weights_out_file = './weights/learner_0.weights'  # Set initial weight file
#     learner = train_policy.main(args)

#     for i in range(1, args.dagger_iterations + 1):
#         print(f'ITERATION {i}/{args.dagger_iterations}')

#         # Save the current learner's weights
#         weights_filename = f'./weights/learner_{i}.weights'
#         torch.save(learner.state_dict(), weights_filename)

#         # Collect new data using the learner's policy and expert corrections for 5 times
#         for j in range(5):
#             print(f'GETTING EXPERT DEMONSTRATIONS WITH CORRECTIONS - Run {j + 1}/5 ------------------------------ ')
#             run_policy_with_expert_correction(
#                 learner, expert=None, 
#                 iteration=i * 5 + j,  # Ensure unique filenames across multiple runs
#                 data_dir=args.train_dir, 
#                 num_timesteps=1000, 
#                 switch_interval=50
#             )

#         # Calculate the cumulative cross-track error for this iteration
#         cumulative_cross_track_error = racer.evaluate_cross_track_error(learner)
#         cross_track_errors.append(cumulative_cross_track_error)
#         print(f'Cumulative cross-track error for iteration {i}: {cumulative_cross_track_error}')

#         # Update weights_out_file for saving in the next iteration
#         args.weights_out_file = weights_filename

#         # Retrain the learner on the aggregated dataset
#         print('RETRAINING LEARNER ON AGGREGATED DATASET')
#         learner = train_policy.main(args)

#     # Plot the cumulative cross-track error over DAgger iterations
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, args.dagger_iterations + 1), cross_track_errors, marker='o')
#     plt.xlabel('DAgger Iterations')
#     plt.ylabel('Cumulative Cross-Track Error')
#     plt.title('DAgger Iterations vs. Cumulative Cross-Track Error')
#     plt.grid(True)
#     plt.savefig('dagger_iterations.png')
#     plt.show()
#     print('DAgger iterations plot saved as dagger_iterations.png')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
#     parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
#     parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
#     parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
#     parser.add_argument("--train_dir", help="directory of training data", default='./dagger_dataset/train')
#     parser.add_argument("--validation_dir", help="directory of validation data", default='./dagger_dataset/val')
#     parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
#     parser.add_argument("--dagger_iterations", help="number of DAgger iterations", type=int, default=10)
#     parser.add_argument("--learner_interval", type=int, help="Number of timesteps for which the learner policy runs", default=20)
#     parser.add_argument("--expert_interval", type=int, help="Number of timesteps to switch between learner and expert", default=100)
#     args = parser.parse_args()

#     dagger_algorithm(args)


import train_policy
import racer
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from full_state_car_racing_env import FullStateCarRacingEnv
import scipy
import imageio
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

def run_policy_with_expert_correction(learner, expert, iteration, data_dir, num_timesteps, switch_interval=50):
    os.makedirs(data_dir, exist_ok=True)
    env = FullStateCarRacingEnv()
    state = env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    timestep = 0
    done = False
    current_time = 0.0

    env.orientation_pid.erase_history()
    env.distance_pid.erase_history()
    env.gas_pid.erase_history()

    while not done and timestep < num_timesteps:
        env.render()
        if (timestep // switch_interval) % 2 == 0:
            learner_action[0] = learner.predict(state, DEVICE)
            learner_action[1] = expert_action[1] if expert_action is not None else 0.5
            learner_action[2] = expert_action[2] if expert_action is not None else 0.0
        else:
            expert_action, eh, cte = env.get_expert_action(env.car, env.track)
            learner_action = expert_action
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

        next_state, _, reward, done, info = env.step(learner_action)
        steering_command = learner_action[0]
        image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{steering_command:.3f}.jpg'
        imageio.imwrite(image_filename, state)

        state = next_state
        timestep += 1

    env.close()
    print(f"Visualization frames saved in {data_dir}")

def dagger_algorithm(args):
    if not os.path.exists('./dagger_dataset'):
        os.makedirs('./dagger_dataset/train', exist_ok=True)
        os.makedirs('./dagger_dataset/val', exist_ok=True)

    cross_track_errors = []
    print('TRAINING LEARNER ON INITIAL DATASET')
    args.weights_out_file = './weights/learner_0.weights'
    learner = train_policy.main(args)

    for i in range(1, args.dagger_iterations + 1):
        print(f'ITERATION {i}/{args.dagger_iterations}')
        weights_filename = f'./weights/learner_{i}.weights'
        torch.save(learner.state_dict(), weights_filename)

        for j in range(5):
            print(f'GETTING EXPERT DEMONSTRATIONS WITH CORRECTIONS - Run {j + 1}/5')
            run_policy_with_expert_correction(
                learner, expert=None, 
                iteration=i * 5 + j, 
                data_dir=args.train_dir, 
                num_timesteps=args.num_timesteps, 
                switch_interval=50
            )

        cumulative_cross_track_error = racer.evaluate_cross_track_error(learner)
        cross_track_errors.append(cumulative_cross_track_error)
        print(f'Cumulative cross-track error for iteration {i}: {cumulative_cross_track_error}')

        args.weights_out_file = weights_filename
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        learner = train_policy.main(args)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.dagger_iterations + 1), cross_track_errors, marker='o')
    plt.xlabel('DAgger Iterations')
    plt.ylabel('Cumulative Cross-Track Error')
    plt.title('DAgger Iterations vs. Cumulative Cross-Track Error')
    plt.grid(True)
    plt.savefig('dagger_iterations.png')
    plt.show()
    print('DAgger iterations plot saved as dagger_iterations.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dagger_dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dagger_dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", type=int, default=10, help="Number of DAgger iterations")
    parser.add_argument("--learner_interval", type=int, default=25, help="Number of timesteps for the learner policy")
    parser.add_argument("--expert_interval", type=int, default=100, help="Number of timesteps for the expert policy")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Total number of timesteps for data collection")
    args = parser.parse_args()

    dagger_algorithm(args)