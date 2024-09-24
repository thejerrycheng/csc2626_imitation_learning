import train_policy
import racer
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def dagger_algorithm(args):
    # Create the dagger_dataset directory if it doesn't exist
    if not os.path.exists('./dagger_dataset'):
        os.makedirs('./dagger_dataset/train', exist_ok=True)
        os.makedirs('./dagger_dataset/val', exist_ok=True)

    # Initialize DAgger process
    cross_track_errors = []  # Track cumulative cross-track error for each iteration

    print('TRAINING LEARNER ON INITIAL DATASET')
    args.weights_out_file = './weights/learner_0.weights'  # Set initial weight file
    learner = train_policy.main(args)

    for i in range(1, args.dagger_iterations + 1):
        print(f'ITERATION {i}/{args.dagger_iterations}')

        # Save the current learner's weights
        weights_filename = f'./weights/learner_{i}.weights'
        torch.save(learner.state_dict(), weights_filename)

        # Collect new data using the learner's policy and expert corrections
        print('GETTING EXPERT DEMONSTRATIONS WITH INTERMITTENT EXPERT CORRECTIONS')
        collect_expert_demos_with_corrections(learner, iteration=i, data_dir=args.train_dir, args=args)

        # Calculate the cumulative cross-track error for this iteration
        cumulative_cross_track_error = racer.evaluate_cross_track_error(learner)
        cross_track_errors.append(cumulative_cross_track_error)
        print(f'Cumulative cross-track error for iteration {i}: {cumulative_cross_track_error}')

        # Update weights_out_file for saving in the next iteration
        args.weights_out_file = weights_filename

        # Retrain the learner on the aggregated dataset
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        learner = train_policy.main(args)

    # Plot the cumulative cross-track error over DAgger iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.dagger_iterations + 1), cross_track_errors, marker='o')
    plt.xlabel('DAgger Iterations')
    plt.ylabel('Cumulative Cross-Track Error')
    plt.title('DAgger Iterations vs. Cumulative Cross-Track Error')
    plt.grid(True)
    plt.savefig('dagger_iterations.png')
    plt.show()
    print('DAgger iterations plot saved as dagger_iterations.png')

def collect_expert_demos_with_corrections(learner, iteration, data_dir='./dagger_dataset/train', args=None):
    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Initialize the car racing environment
    env = FullStateCarRacingEnv()
    state = env.reset()

    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None

    timestep = 0
    done = False
    
    # Parameters for alternating control
    policy_timesteps = 50  # Number of steps the learner's policy controls
    expert_timesteps = 10  # Number of steps the expert controls

    while not done:
        # Switch to learner's policy for 'policy_timesteps' steps
        for _ in range(policy_timesteps):
            if done:
                break
            state, expert_action, reward, done, info = env.step(learner_action)
            learner_action[0] = learner.predict(state, device=DEVICE)  # Use learner's policy
            learner_action[1] = expert_action[1]  # Maintain the expert throttle
            learner_action[2] = expert_action[2]  # Maintain the expert brake

            # Save the state image and expert steering command
            image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{expert_action[0]}.jpg'
            imageio.imwrite(image_filename, state)
            timestep += 1
        
        # Switch to expert control for 'expert_timesteps' steps
        for _ in range(expert_timesteps):
            if done:
                break
            state, expert_action, reward, done, info = env.step(expert_action)  # Take step using expert action
            learner_action[0] = expert_action[0]  # Let the expert provide steering corrections
            learner_action[1] = expert_action[1]  # Maintain expert throttle
            learner_action[2] = expert_action[2]  # Maintain expert brake

            # Save the expert correction image
            image_filename = f'{data_dir}/expert_{iteration}_{timestep}_{expert_action[0]}.jpg'
            imageio.imwrite(image_filename, state)
            timestep += 1
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dagger_dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dagger_dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="number of DAgger iterations", type=int, default=10)
    parser.add_argument("--policy_timesteps", type=int, help="number of timesteps the policy controls in each iteration", default=50)
    parser.add_argument("--expert_timesteps", type=int, help="number of timesteps the expert corrects in each iteration", default=10)
    args = parser.parse_args()

    dagger_algorithm(args)