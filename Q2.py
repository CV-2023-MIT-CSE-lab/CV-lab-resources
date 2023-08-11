import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseCNN(nn.Module):
    def __init__(self, num_classes, num_channels, image_height, image_width):
        super(SiameseCNN, self).__init__()

        # CNN Layers for Image Input A
        self.conv_image_a = nn.Conv2d(num_channels, 16, kernel_size=3)
        self.pool_image_a = nn.MaxPool2d(kernel_size=2)
        self.fc_image_a = nn.Linear(16 * (image_height // 2) * (image_width // 2), 64)

        # CNN Layers for Image Input B
        self.conv_image_b = nn.Conv2d(num_channels, 16, kernel_size=3)
        self.pool_image_b = nn.MaxPool2d(kernel_size=2)
        self.fc_image_b = nn.Linear(16 * (image_height // 2) * (image_width // 2), 64)

        # Merge the outputs of the two input branches
        self.fc_merged = nn.Linear(64 * 2, 128)

        # Final output layer
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, image_input_a, image_input_b):
        # Process Image Input A
        x_image_a = F.relu(self.conv_image_a(image_input_a))
        x_image_a = self.pool_image_a(x_image_a)
        x_image_a = x_image_a.view(-1, 16 * (image_input_a.size(2) // 2) * (image_input_a.size(3) // 2))
        x_image_a = F.relu(self.fc_image_a(x_image_a))

        # Process Image Input B
        x_image_b = F.relu(self.conv_image_b(image_input_b))
        x_image_b = self.pool_image_b(x_image_b)
        x_image_b = x_image_b.view(-1, 16 * (image_input_b.size(2) // 2) * (image_input_b.size(3) // 2))
        x_image_b = F.relu(self.fc_image_b(x_image_b))

        # Concatenate the processed inputs
        x_merged = torch.cat((x_image_a, x_image_b), dim=1)

        # Continue with fully connected layers
        x_merged = F.relu(self.fc_merged(x_merged))

        # Final output layer
        output = self.output_layer(x_merged)

        return output




#%%
# NEED TO INITIATE THE VALUES HERE 
# Example usage:
# Assuming image_input_a and image_input_b are the tensors for image inputs A and B, respectively.
# You would also need to set appropriate values for num_classes, num_channels, image_height, and image_width.

# Instantiate the model
model = SiameseCNN(num_classes, num_channels, image_height, image_width)

# Forward pass
output = model(image_input_a, image_input_b)
#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Assuming you have your data and dataloaders for training
# image_input_a_loader and image_input_b_loader are the dataloaders for the two image inputs

# Instantiate the model
model = SiameseCNN(num_classes, num_channels, image_height, image_width)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for batch_idx, (image_input_a, image_input_b, target) in enumerate(zip(image_input_a_loader, image_input_b_loader, target_loader)):
        # Move data to the device (CPU or GPU)
        image_input_a, image_input_b, target = image_input_a.to(device), image_input_b.to(device), target.to(device)

        # Clear the gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass
        output = model(image_input_a, image_input_b)

        # Calculate the loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        # Print training progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(image_input_a_loader)}], Loss: {loss.item()}")

# After training, the model should have learned to distinguish between the two image inputs based on their features.
# You can then use the trained model to make predictions on new image pairs.

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

# Assuming you have your data and dataloaders for evaluation
# image_input_a_loader_eval and image_input_b_loader_eval are the dataloaders for the two image inputs during evaluation
# eval_target_loader contains the ground truth labels/targets for evaluation

# Evaluation function
def evaluate_siamese_model(model, dataloader_a, dataloader_b, target_loader, device):
    model.eval()  # Set the model to evaluation mode

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for image_input_a, image_input_b, target in zip(dataloader_a, dataloader_b, target_loader):
            # Move data to the device (CPU or GPU)
            image_input_a, image_input_b, target = image_input_a.to(device), image_input_b.to(device), target.to(device)

            # Forward pass
            output = model(image_input_a, image_input_b)

            # Calculate similarity scores or distances
            # The actual interpretation may depend on your specific problem and activation function used in the model
            # For example, if the model uses a softmax activation, you might need to take argmax to get the predicted class.
            # For simplicity, we assume the output is a similarity score in the range [0, 1] where 0 means dissimilar and 1 means similar.
            similarity_scores = output.squeeze().cpu().numpy()

            # Store the ground truth labels and predicted scores
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(similarity_scores)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_targets, [1 if score >= 0.5 else 0 for score in all_outputs])
    mae = mean_absolute_error(all_targets, all_outputs)
    mse = mean_squared_error(all_targets, all_outputs)
    auc_roc = roc_auc_score(all_targets, all_outputs)

    return accuracy, mae, mse, auc_roc

# Example usage:
# Assuming model is the trained SiameseCNN model, image_input_a_loader_eval, image_input_b_loader_eval,
# and eval_target_loader are the dataloaders for evaluation, and device is the device the model is on (CPU or GPU).

accuracy, mae, mse, auc_roc = evaluate_siamese_model(model, image_input_a_loader_eval, image_input_b_loader_eval, eval_target_loader, device)

print(f"Accuracy: {accuracy:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
#%%

import torch
import matplotlib.pyplot as plt

# Assuming you have already trained the model and obtained the predicted similarity scores and true labels
# all_outputs and all_targets are the lists containing the predicted similarity scores and true labels, respectively

# Convert lists to NumPy arrays
all_outputs = torch.tensor(all_outputs)
all_targets = torch.tensor(all_targets)

# Convert similarity scores to distances if necessary (distance = 1 - similarity_score)
all_distances = 1 - all_outputs

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(all_targets, all_distances, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Predicted Distances')
plt.title('Scatter Plot of True Labels vs. Predicted Distances')
plt.grid(True)
plt.show()

data_path = "E:/Q2/Input - 2 - malaria cells/cell_images/"

#%%
#%%
#%%

#MALARIA
# RUN THIS ONLY ONCE - THE SPLIT WILL BE CREATED
import splitfolders
input_folder = "E:/Q2/Input - 2 - malaria cells/cell_images/"
output = "E:/Q2/Input - 2 - malaria cells/cell_images/split" 

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.7, .3)) 

#%%
#TUMOR
# RUN THIS ONLY ONCE - THE SPLIT WILL BE CREATED
import splitfolders
input_folder = "E:/Q2/Input - 1 - brain tumor/data/Tumor"
output = "E:/Q2/Input - 1 - brain tumor/split2" 

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.7, .3)) 

#%%
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


#train and test data directory
data_dir = "E:/Q2/Input - 2 - malaria cells/cell_images/split/train"
test_data_dir = "E:/Q2/Input - 2 - malaria cells/cell_images/split/val"

#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))

#%%
print("Follwing classes are there : \n",dataset.classes)
#%%

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

batch_size = 128
val_size = 5000
train_size = len(dataset) - val_size 

train_data,val_data = random_split(dataset,[train_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

#output
#Length of Train Data : 12034
#Length of Validation Data : 2000

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)        
#%%
batch_size = 128
#%%
import torch.nn as nn
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#%%

class NaturalSceneClassification(ImageClassificationBase):
    
    def __init__(self):
        
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6)
        )
    
    def forward(self, xb):
        return self.network(xb)

#%%
model = NaturalSceneClassification()
model
#%%
def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)
#%%
device = get_default_device()
device
#%%
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)
#%%
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history
#%%
model = to_device(NaturalSceneClassification(),device)
#%%
evaluate(model,val_dl)
#%%
num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.001
#%%
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
#%%
