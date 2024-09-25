import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

def train_discrete(model, iterator, opt, args, class_weights=None):
    model.train()
    loss_hist = []
    
    # Use weighted loss if class_weights are provided
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)
    
    for i_batch, batch in enumerate(iterator):
        # Retrieve images and labels from batch
        x = batch['image']
        y = batch['cmd']

        # Move data to the GPU if available
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Zero the gradients
        opt.zero_grad()

        # Forward pass
        logits = model(x)
        
        # Compute the loss
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()

        # Update parameters
        opt.step()
        
        # Track the loss
        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)
        
        PRINT_INTERVAL = int(len(iterator) / 3)        
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print ('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0),
                time.time() - args.start_time,
            ))
def accuracy(y_pred, y_true):
    "y_true is (batch_size) and y_pred is (batch_size, K)"
    _, y_max_pred = y_pred.max(1)
    correct = ((y_true == y_max_pred).float()).mean() 
    acc = correct * 100
    return acc

def test_discrete(model, iterator, opt, args):
    model.eval()  # Set the model to evaluation mode
    
    acc_hist = []
    
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for i_batch, batch in enumerate(iterator):
            x = batch['image']
            y = batch['cmd']

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            logits = model(x)
            y_pred = F.softmax(logits, dim=1)

            acc = accuracy(y_pred, y)
            acc = acc.detach().cpu().numpy()
            acc_hist.append(acc)
        
    avg_acc = np.asarray(acc_hist).mean()
    
    print('\tVal: \tAcc: {}  Time: {:10.3f}'.format(
        avg_acc,
        time.time() - args.start_time,
    ))
    
    return avg_acc

def get_class_distribution(iterator, args):
    class_dist = np.zeros((args.n_steering_classes,), dtype=np.float32)
    for i_batch, batch in enumerate(iterator):
        y = batch['cmd'].detach().numpy().astype(np.int32)
        class_dist += np.bincount(y, minlength=args.n_steering_classes)
        
    return (class_dist / sum(class_dist))

def main(args):
    data_transform = transforms.Compose([ 
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=80),
        transforms.ToTensor()
    ])
    
    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)
    
    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)
    
    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    
    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()
    
    args.class_dist = get_class_distribution(training_iterator, args)
    print("Class distribution result is: ", args.class_dist)

    if args.weighted_loss:
        # Exclude class 19 from the weight calculation
        class_weights = np.where(np.arange(len(args.class_dist)) != 19, 1.0 / (args.class_dist + 1e-6), 0.0)
        class_weights /= class_weights.sum()  # Normalize to sum to 1
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Detailed printout
        print("\nDetailed Class Weights and Steering Ranges (excluding Class 19):")
        for i, weight in enumerate(class_weights):
            if i == 19:
                continue
            steering_min = (i / (args.n_steering_classes - 1.0)) * 2.0 - 1.0
            steering_max = ((i + 1) / (args.n_steering_classes - 1.0)) * 2.0 - 1.0
            print(f"Class {i}: Steering Range [{steering_min:.3f}, {steering_max:.3f}] => Weight: {weight.item() * 100:.2f}%")
    else:
        class_weights = None

    best_val_accuracy = 0 
    for epoch in range(args.n_epochs):
        print('EPOCH', epoch)

        # Train the driving policy with the weighted loss
        train_discrete(driving_policy, training_iterator, opt, args, class_weights)
        
        # Evaluate the driving policy on the validation set
        val_accuracy = test_discrete(driving_policy, validation_iterator, opt, args)
        
        # If the accuracy on the validation set is a new high, then save the network weights
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(driving_policy.state_dict(), args.weights_out_file)
            print(f"New best accuracy: {best_val_accuracy:.2f}. Model weights saved to {args.weights_out_file}")
        
    return driving_policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./expert_dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./expert_dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights",
                        required=True)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurrence",
                        default=True)
    
    args = parser.parse_args()

    main(args)