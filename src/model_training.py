"""
Model Training for Tuberculosis Detection (PyTorch) - OPTIMIZED FOR CPU
"""
import os, sys, json, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_SIZE = 128  # Smaller for CPU speed
BATCH_SIZE = 32
EPOCHS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)

def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            train_ds.classes)

def compute_weights(loader):
    labels = []
    for _, y in loader: labels.extend(y.numpy())
    labels = np.array(labels)
    w = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.FloatTensor(w).to(DEVICE)


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def build_transfer_model(model_name):
    if model_name == 'ResNet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in list(m.parameters())[:-20]: p.requires_grad = False
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(),
                             nn.BatchNorm1d(256), nn.Dropout(0.5), nn.Linear(256, 2))
        return m
    elif model_name == 'VGG16':
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for p in list(m.features.parameters())[:-6]: p.requires_grad = False
        m.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Handle variable input size
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.5), nn.Linear(256, 2))
        # Replace avgpool to handle 128px input
        m.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        return m
    elif model_name == 'EfficientNetB0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in list(m.parameters())[:-20]: p.requires_grad = False
        inf = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(inf, 256),
                                      nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3), nn.Linear(256, 2))
        return m


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train_model(model, name, train_loader, val_loader, class_weights):
    log(f"\n{'='*60}\nTRAINING: {name}\n{'='*60}")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Trainable params: {trainable:,}")

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_acc, patience_count = 0, 0

    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history['loss'].append(tr_loss)
        history['accuracy'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_accuracy'].append(vl_acc)

        log(f"  Epoch {epoch+1}/{EPOCHS} [{time.time()-t0:.0f}s] "
            f"loss:{tr_loss:.4f} acc:{tr_acc:.4f} val_loss:{vl_loss:.4f} val_acc:{vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{name}_best.pth'))
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 4:
                log(f"  Early stopping at epoch {epoch+1}"); break

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{name}.pth'))
    with open(os.path.join(RESULTS_DIR, f'{name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    log(f"  Best val_acc: {best_acc:.4f}")
    return history


def main():
    log("=" * 60)
    log("TUBERCULOSIS DETECTION - MODEL TRAINING (PyTorch CPU-Optimized)")
    log("=" * 60)
    log(f"Device: {DEVICE} | Image size: {IMG_SIZE}px | Epochs: {EPOCHS}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("\nLoading data...")
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    class_weights = compute_weights(train_loader)
    log(f"Classes: {classes} | Weights: {class_weights.cpu().numpy()}")
    log(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'w') as f:
        json.dump({c: i for i, c in enumerate(classes)}, f, indent=2)

    configs = [
        ('Custom_CNN', lambda: CustomCNN()),
        ('ResNet50', lambda: build_transfer_model('ResNet50')),
        ('VGG16', lambda: build_transfer_model('VGG16')),
        ('EfficientNetB0', lambda: build_transfer_model('EfficientNetB0')),
    ]

    results = {}
    for name, builder in configs:
        try:
            model = builder()
            hist = train_model(model, name, train_loader, val_loader, class_weights)
            results[name] = {
                'best_val_accuracy': float(max(hist['val_accuracy'])),
                'best_val_loss': float(min(hist['val_loss'])),
                'epochs_trained': len(hist['loss']),
            }
            del model
        except Exception as e:
            log(f"\nERROR training {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}

    with open(os.path.join(RESULTS_DIR, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 60)
    log("MODEL TRAINING COMPLETE!")
    log("=" * 60)
    for name, res in results.items():
        if 'error' in res: log(f"  {name}: FAILED - {res['error']}")
        else: log(f"  {name}: val_acc={res['best_val_accuracy']:.4f}, epochs={res['epochs_trained']}")


if __name__ == '__main__':
    main()
