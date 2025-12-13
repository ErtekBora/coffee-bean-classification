"""
Training script for Coffee Bean Classification
Supports both SimpleCNN and ResNet18 fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import argparse
import os

from model import SimpleCNN, create_resnet18_finetuned, freeze_resnet_except_layer4


def get_data_loaders(train_dir, test_dir, batch_size=12):
    """Create train and test data loaders"""
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path, model_name):
    """Training loop with validation"""
    
    best_val_acc = 0.0
    
    print("\n" + "="*70)
    print(f"STARTING TRAINING - {num_epochs} EPOCHS")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
        
        val_acc = 100 * correct / total
        arabica_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        robusta_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        
        # Learning rate scheduling
        lr_info = ""
        if epoch > 1:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                lr_info = f" → LR: {new_lr:.6f}"
        else:
            scheduler.step(val_acc)
        
        print(f"Epoch [{epoch:2d}/{num_epochs}] | Loss: {train_loss:.4f} | Val: {val_acc:.2f}% | "
              f"Ara: {arabica_acc:.2f}% | Rob: {robusta_acc:.2f}%{lr_info}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("  ✅ Best model saved!")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"⏱️  Total Time: {total_time/60:.2f} minutes")
    print(f"💾 Model saved: {save_path}")
    print("="*70)
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train Coffee Bean Classifier')
    parser.add_argument('--model', type=str, default='resnet18', choices=['simple_cnn', 'resnet18'],
                        help='Model architecture (default: resnet18)')
    parser.add_argument('--train_dir', type=str, default='dataset/train',
                        help='Path to training data')
    parser.add_argument('--test_dir', type=str, default='dataset/test',
                        help='Path to test data')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size (default: 12)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--patience', type=int, default=4,
                        help='Scheduler patience (default: 4)')
    parser.add_argument('--class_weights', type=str, default='1.2,1.0',
                        help='Class weights as comma-separated values (default: 1.2,1.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model == 'simple_cnn':
        model = SimpleCNN().to(device)
        print("Model: Custom 3-Layer SimpleCNN")
    else:
        model = create_resnet18_finetuned(num_classes=2, pretrained=True).to(device)
        model = freeze_resnet_except_layer4(model)
        print("Model: ResNet18 (Fine-tuned)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(args.train_dir, args.test_dir, args.batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Class weights
    class_weights = [float(w) for w in args.class_weights.split(',')]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer
    if args.model == 'resnet18':
        # Dual learning rates for fine-tuning
        optimizer = optim.Adam([
            {'params': model.layer4.parameters(), 'lr': args.lr / 10},  # Lower LR for pretrained
            {'params': model.fc.parameters(), 'lr': args.lr}            # Higher LR for new layer
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.patience
    )
    
    # Output path
    if args.output is None:
        args.output = f'coffee_model_{args.model}.pth'
    
    # Train
    best_acc = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        args.epochs, device, args.output, args.model
    )
    
    print(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
