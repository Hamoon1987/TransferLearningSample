
from torchvision import models
import torch.nn as nn
from config import args



def fetch_model():
    # Import the model
    model = models.vgg16 (pretrained=True)
    # Keep the features extraction section of the model constant while the classification section could be updated
    for param in model.features.parameters():
        param.requires_grad = False
    # Changing the last layer of classification to the number of outputs
    n_inputs = model.classifier[-1].in_features
    last_layer = nn.Linear(n_inputs, args.output_classes)
    model.classifier[-1] = last_layer
    return model

if __name__ == '__main__':
    model = models.vgg16 (pretrained=True)
    print(f"Initial vgg16  model: \n {model}", '\n')
    model = fetch_model()
    print(f"Adjusted last layer of the classifier: \n {model.classifier[-1]}" )