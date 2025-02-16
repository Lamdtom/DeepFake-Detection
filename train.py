import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import os
from EfficientNetB4 import EfficientNetB4
from dataset import DeepfakeDataset
import argparse

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train Custom EfficientNet-B4 for Deepfake Detection')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_dataset', type=str, default='./dataset/train', help='Path to the training dataset')
    parser.add_argument('--val_dataset', type=str, default='./dataset/val', help='Path to the validation dataset')
    parser.add_argument('--test_dataset', type=str, default='./dataset/test', help='Path to the test dataset')
    return parser.parse_args()

accumulation_steps = 2  # Accumulate gradients over 4 batches

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad() 
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

             # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        
        # Calculate training accuracy for the epoch
        train_accuracy = 100 * correct_preds / total_preds
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

        #Validate model
        if(epoch%5 == 0):
            validate_model(model, val_loader, criterion, device)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    model = EfficientNetB4() 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare DataLoader for training, validation, and testing
    train_dataset = DeepfakeDataset(args.train_dataset, transform=transform)
    val_dataset = DeepfakeDataset(args.val_dataset, transform=transform)
    test_dataset = DeepfakeDataset(args.test_dataset, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs, device=device)

    # Test the model
    test_model(model, test_loader, device)
