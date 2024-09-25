import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class DiscreteDrivingPolicy(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # Define the convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
        )

        # Update the fully connected layer with in_features = 4096
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4096, out_features=64),  # Adjusted to match the flattened output
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )
        
        self.apply(weights_init)

    def forward(self, x):
        f = self.features(x)
        logits = self.classifier(f)
        return logits
    
    # Rename the evaluation function to predict to avoid confusion
    def predict(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)
        
        y_pred = logits.view(-1, self.n_classes) 
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()
        
        steering_cmd = (steering_class / (self.n_classes - 1.0)) * 2.0 - 1.0
        # steering_cmd = (steering_class - ((self.n_classes - 1) / 2)) / ((self.n_classes - 1) / 2)
        print("Steering class: ", steering_class, "Steering command: ", steering_cmd)
        
        return steering_cmd
    
    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict({k: v for k, v in weights.items()}, strict=True)
    


