#imports libraries

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch import nn, optim
from PIL import Image
# Image transform to perform data augmentation
transform = transforms.Compose([
    transforms.RandomAffine(degrees=(0, 20), translate=(0, 0.2), scale=(0.7, 1)), # Randomly rotate (+- 20 degrees), translate and scale the image
    transforms.Grayscale(num_output_channels=3), # 3 channels
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),  # Add noise
    transforms.ToTensor(), # Convert image to a tensor
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # Normalize the data, so that the gradients are more consistent, thus making it easier to train
])
transformtest = transforms.Compose([
    transforms.Grayscale(num_output_channels=____),
    transforms.ToTensor(),
    transforms.Normalize(mean=____, std=____)
])

batch_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # use gpu

# Load the MNIST datasets for training
train_dataset = datasets.MNIST(
    root='train',
    train=True,
    transform=transform,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataset = datasets.MNIST(
    root='test',
    train=False,
    transform=_______,
    download=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

model = models.resnet18(num_classes=10).to(device) # Load the ResNet18 model with 10 classes
model.load_state_dict(torch.load("/content/ioai_model_FINISHED.pth")) # Load the model, filling in the model's weights
# Define a ResNet18 model and Adam optimizer

loss_fn = nn.CrossEntropyLoss() # Use cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr = 0.005) # Use Adam optimizer with a learning rate of 0.005
from tqdm import tqdm # show estimate progress bar

num_epochs = 25 # set number of epochs

for epoch in tqdm(range(0, num_epochs)): # loop over the dataset multiple times
  model.train() # set the model to training mode

  for i, data in tqdm(enumerate(train_loader)): # loop over the data iterator

    images, labels = data # get the inputs
    images = images.to(device) # move the inputs to the device
    labels = labels.to(device) # move the labels to the device

    pred = model(images) # forward pass
    loss = loss_fn(pred, labels) # calculate the loss
    loss.backward() # backward pass
    optimizer.step() # update the weights
    optimizer.zero_grad() # zero the gradients

    if i % 100 == 99: # print the loss every 100 batches
      print(f"Epoch {epoch+1}, Step {i+1}, Loss = {loss.item()}") # print the loss value

torch.save(model.state_dict(), "/content/ioai_model_FINISHED.pth") # Save the model
import matplotlib.pyplot as plt

def imshow(img, title=None): # function to plot the image
    img, lbl = next(iter(train_loader))
    plt.imshow(img[0][0], cmap='gray')
    plt.title(f"{lbl[0]}")
    plt.show()


model.to(device)
model.eval()

# Evaluation

model.eval() # Set the model to evaluation mode (IMPORTANT)

with torch.no_grad(): 
    i = 0 # Counter for the amount of batches
    total_accuracy = 0 # Total accuracy used to calculate average batch accuracy at the end
    for images, labels in test_loader: # Loop through for every batch in test_loader

        # Send the image/label tensors to the GPU
        images = images.to(device)
        labels = labels.to(device)

        images = images.view(images.size(0), -1) # Flatten the images
        test_output = model(images) # Get the model predictions (NOT the actual prediction, but a vector of the probabilites of each number)
        pred_y = torch.max(test_output, 1)[1] # Get the class which has the highest probability (what number the model thinks the image corresponds to)
        batch_accuracy = (pred_y == labels).sum().item() / float(labels.size(0)) # Determines the accuracy of the model for this batch
        total_accuracy += batch_accuracy # Adds to the concurent accuracy
        i += 1

    # Calculates average accuracy across all batches
    accuracy = total_accuracy / i

print(f"Test Accuracy of the model on the 10000 test images: {accuracy:.3f}")

