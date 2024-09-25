import os
import imageio
import numpy as np
import argparse
from full_state_car_racing_env import FullStateCarRacingEnv  # Import the environment

# Define the frame rate per second (FPS) explicitly
FPS = 50  # Set to the same FPS as used in the FullStateCarRacingEnv environment

# Create the necessary directories if they don't exist
os.makedirs('./expert_dataset_2/train', exist_ok=True)
os.makedirs('./expert_dataset_2/val', exist_ok=True)

def collect_expert_demos(num_iterations, num_timesteps, data_dir='./expert_dataset_2/train'):
    # Initialize the car racing environment with the expert feedback controller
    env = FullStateCarRacingEnv()  # Use the expert environment

    for iteration in range(num_iterations):
        print(f"Collecting expert data for iteration {iteration}")
        
        # Reset the environment to start a new episode
        state = env.reset()  # Adjusted to expect only one return value
        
        timestep = 0
        done = False
        current_time = 0.0  # Initialize current time
        
        # Reset the PID history for each iteration
        env.orientation_pid.erase_history()
        env.distance_pid.erase_history()
        env.gas_pid.erase_history()

        while not done and timestep < num_timesteps:
            # Get the expert action from the feedback controller
            expert_action, eh, cte = env.get_expert_action(env.car, env.track)
            
            # Increment the time by 1/FPS before calling update
            current_time += 1.0 / FPS  # Use the defined FPS value here
            
            # Only update PID controllers if we have enough time difference
            if len(env.orientation_pid.timestamps_of_errors) >= 1:
                last_timestamp = env.orientation_pid.timestamps_of_errors[-1]
                if current_time > last_timestamp:  # Ensure time has advanced
                    env.orientation_pid.update(eh, current_time)
                    env.distance_pid.update(cte, current_time)
                    env.gas_pid.update(eh, current_time)
            else:
                # For the very first time, we allow the update
                env.orientation_pid.update(eh, current_time)
                env.distance_pid.update(cte, current_time)
                env.gas_pid.update(eh, current_time)

            # Take a step using the expert action
            next_state, _, reward, done, info = env.step(expert_action)

            # Save the current state (image) with the corresponding expert steering command
            steering_command = expert_action[0]  # Extract the steering command
            image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{steering_command}.jpg'
            imageio.imwrite(image_filename, state)

            # Update the state and timestep
            state = next_state
            timestep += 1

        print(f"Completed data collection for iteration {iteration}")

    env.close()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Collect expert demonstration data using a feedback controller.")
    parser.add_argument("--num_iterations", type=int, default=20, help="Number of iterations to collect expert data")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps per iteration")
    parser.add_argument("--data_dir", type=str, default='./expert_dataset_2/train', help="Directory to save collected data")

    args = parser.parse_args()

    # Collect expert data based on the provided arguments
    collect_expert_demos(num_iterations=args.num_iterations, num_timesteps=args.num_timesteps, data_dir=args.data_dir)