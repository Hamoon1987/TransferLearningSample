
from torchvision import models
import torch.nn as nn
from config import args



def fetch_model():
    # Import the model
    model = models.resnet18 (pretrained=True)
    # Keep the features extraction section of the model constant while the FC section could be updated
    for param in model.parameters():
        param.requires_grad = False
    # Changing the last layer of classification to the number of outputs
    n_inputs = model.fc.in_features
    last_layer = nn.Linear(n_inputs, args.output_classes)
    model.fc = last_layer
    return model

if __name__ == '__main__':
    model = models.resnet18 (pretrained=True)
    print(f"Initial ResNet  model: \n {model}", '\n')
    model = fetch_model()
    print(f"Adjusted last layer of the fc: \n {model.fc}" )