import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np


# ----------------------------------------------------------------------------
# ------------------------------ BASELINE SETUP ------------------------------
# ----------------------------------------------------------------------------

print("Part 1 - Baseline Setup")
# a) Load pre-trained ResNet-152
model = torchvision.models.resnet152(pretrained=True)

# b) Replace final classification layer for CIFAR-10
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# c) Freeze backbone layers, train only classification head
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# d) Training loop with performance recording
epochs = 5
train_losses = []
train_accs = []
val_losses = []
val_accs = []

print("Training Baseline Model")

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    print(f"Epoch {epoch+1}/{epochs} - Training...")
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(trainloader)}: Loss: {loss.item():.4f}')
    
    train_loss /= len(trainloader)
    train_acc = 100. * correct / total
    
    # Validation
    print(f"Epoch {epoch+1}/{epochs} - Validation...")
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(testloader)
    val_acc = 100. * correct / total
    

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
    print(f'         Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')


# Save baseline results
baseline_train_losses = train_losses.copy()
baseline_train_accs = train_accs.copy()
baseline_val_losses = val_losses.copy()
baseline_val_accs = val_accs.copy()

print("Baseline training completed!")
print("=" * 50)


# ----------------------------------------------------------------------------
# ------------------ RESIDUAL CONNECTION IN PRACTICE -------------------------
# ----------------------------------------------------------------------------

print("Part 2 - Residual Connection in Practice")

# a) Disable skip connections in selected residual blocks
class ModifiedBottleneck(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.conv3 = original_block.conv3
        self.bn3 = original_block.bn3
        self.relu = original_block.relu
        self.downsample = original_block.downsample
        
    def forward(self, x):
        # Forward without skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # No skip connection here
        out = self.relu(out)
        
        return out

# Create modified model
model_modified = torchvision.models.resnet152(pretrained=True)
model_modified.fc = nn.Linear(model_modified.fc.in_features, 10)

# Disable skip connections in last few blocks of layer4
for i in range(3):  # Modify last 3 blocks
    original_block = model_modified.layer4[-(i+1)]
    model_modified.layer4[-(i+1)] = ModifiedBottleneck(original_block)

# Freeze backbone, train only classification head
for param in model_modified.parameters():
    param.requires_grad = False

for param in model_modified.fc.parameters():
    param.requires_grad = True

# Setup modified model training
model_modified.to(device)
optimizer_modified = optim.Adam(model_modified.fc.parameters(), lr=0.001)

# Training loop for modified model
train_losses_mod = []
train_accs_mod = []
val_losses_mod = []
val_accs_mod = []

print("Training modified model (without skip connections)")

for epoch in range(epochs):
    # Training
    model_modified.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer_modified.zero_grad()
        outputs = model_modified(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_modified.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss /= len(trainloader)
    train_acc = 100. * correct / total
    
    # Validation
    model_modified.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_modified(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(testloader)
    val_acc = 100. * correct / total
    
    # Save results
    train_losses_mod.append(train_loss)
    train_accs_mod.append(train_acc)
    val_losses_mod.append(val_loss)
    val_accs_mod.append(val_acc)
    
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
    print(f'         Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')

# b) Compare results
print("\n" + "=" * 50)
print("COMPARISON RESULTS:")
print("=" * 50)

print(f"Baseline Final Val Accuracy: {baseline_val_accs[-1]:.2f}%")
print(f"Modified Final Val Accuracy: {val_accs_mod[-1]:.2f}%")

# Plot comparison
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(baseline_train_losses, label='Baseline', color='blue')
plt.plot(train_losses_mod, label='Modified', color='red')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(baseline_val_losses, label='Baseline', color='blue')
plt.plot(val_losses_mod, label='Modified', color='red')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(baseline_train_accs, label='Baseline', color='blue')
plt.plot(train_accs_mod, label='Modified', color='red')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(baseline_val_accs, label='Baseline', color='blue')
plt.plot(val_accs_mod, label='Modified', color='red')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('Figures\Task-1\comparison_results.png')
plt.show()

# ----------------------------------------------------------------------------
# ------------------ FEATURES HIERARCHIES & REPRESENTATIONS -----------------
# ----------------------------------------------------------------------------

print("Part 3 - Features Hierarchies & Representations")
# a) Collect features from early, middle, and late layers
def get_features(model, dataloader, device):
    model.eval()
    
    # Feature extraction hooks
    early_features = []
    middle_features = []
    late_features = []
    
    def hook_early(module, input, output):
        early_features.append(output.detach().cpu())
    
    def hook_middle(module, input, output):
        middle_features.append(output.detach().cpu())
        
    def hook_late(module, input, output):
        late_features.append(output.detach().cpu())
    
    # Register hooks
    handle_early = model.layer1[0].register_forward_hook(hook_early)
    handle_middle = model.layer2[0].register_forward_hook(hook_middle) 
    handle_late = model.layer4[0].register_forward_hook(hook_late)
    
    labels = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= 50:  # Limit samples for visualization
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            labels.extend(targets.numpy())
    
    # Remove hooks
    handle_early.remove()
    handle_middle.remove()
    handle_late.remove()
    
    # Concatenate and flatten features
    early_feat = torch.cat(early_features, dim=0)
    middle_feat = torch.cat(middle_features, dim=0)
    late_feat = torch.cat(late_features, dim=0)
    
    # Global average pooling for dimensionality reduction
    early_feat = torch.mean(early_feat, dim=(2, 3))
    middle_feat = torch.mean(middle_feat, dim=(2, 3))
    late_feat = torch.mean(late_feat, dim=(2, 3))
    
    return early_feat.numpy(), middle_feat.numpy(), late_feat.numpy(), labels

# Extract features from baseline model
print("Extracting features for visualization...")
early_features, middle_features, late_features, feature_labels = get_features(model, testloader, device)

# b) Visualize using t-SNE


print("Computing t-SNE embeddings...")

# Apply t-SNE to each feature set
tsne = TSNE(n_components=2, random_state=42, perplexity=30)

early_tsne = tsne.fit_transform(early_features)
middle_tsne = tsne.fit_transform(middle_features)
late_tsne = tsne.fit_transform(late_features)

# Plot feature visualizations
plt.figure(figsize=(15, 5))

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
colors = plt.cm.tab10(np.linspace(0, 1, 10))

plt.subplot(1, 3, 1)
for i in range(10):
    mask = np.array(feature_labels) == i
    plt.scatter(early_tsne[mask, 0], early_tsne[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.6, s=20)
plt.title('Early Layer Features (Layer 1)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 3, 2)
for i in range(10):
    mask = np.array(feature_labels) == i
    plt.scatter(middle_tsne[mask, 0], middle_tsne[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.6, s=20)
plt.title('Middle Layer Features (Layer 2)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.subplot(1, 3, 3)
for i in range(10):
    mask = np.array(feature_labels) == i
    plt.scatter(late_tsne[mask, 0], late_tsne[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.6, s=20)
plt.title('Late Layer Features (Layer 4)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('Figures\Task-1\feature_hierarchies.png', dpi=300, bbox_inches='tight')
plt.show()

print("Feature hierarchy analysis completed!")
print("=" * 50)


# ----------------------------------------------------------------------------
# ------------------ TRANSFER LEARNING AND GENERALIZATION ------------------
# ----------------------------------------------------------------------------

print("Part 4 - Transfer Learning & Generalization")


# Load CIFAR-100 dataset
trainset_c100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader_c100 = DataLoader(trainset_c100, batch_size=32, shuffle=True)

testset_c100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader_c100 = DataLoader(testset_c100, batch_size=32, shuffle=False)

# b) Compare ImageNet-pretrained vs random initialization
print("Experiment 1: ImageNet-pretrained vs Random initialization")

# Model 1: ImageNet-pretrained
model_pretrained = torchvision.models.resnet152(pretrained=True)
model_pretrained.fc = nn.Linear(model_pretrained.fc.in_features, 100)  # CIFAR-100 has 100 classes

# Model 2: Random initialization
model_random = torchvision.models.resnet152(pretrained=False)
model_random.fc = nn.Linear(model_random.fc.in_features, 100)

# Move models to device
model_pretrained.to(device)
model_random.to(device)

# Setup optimizers - fine-tune full backbone with lower learning rate
optimizer_pretrained = optim.Adam(model_pretrained.parameters(), lr=0.0001)
optimizer_random = optim.Adam(model_random.parameters(), lr=0.001)

# Training function
def train_model(model, optimizer, trainloader, testloader, epochs, model_name):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"Training {model_name}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss /= len(trainloader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(testloader)
        val_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'{model_name} Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return train_losses, train_accs, val_losses, val_accs

# Train both models (reduced epochs for comparison)
epochs_transfer = 3

pretrained_results = train_model(model_pretrained, optimizer_pretrained, trainloader_c100, testloader_c100, epochs_transfer, "Pretrained")
random_results = train_model(model_random, optimizer_random, trainloader_c100, testloader_c100, epochs_transfer, "Random Init")

print("\n" + "=" * 50)
print("Experiment 2: Final block only vs Full backbone fine-tuning")

# c) Compare final block vs full backbone fine-tuning
# Model 3: Fine-tune only final block
model_final_only = torchvision.models.resnet152(pretrained=True)
model_final_only.fc = nn.Linear(model_final_only.fc.in_features, 100)

# Freeze all parameters except final layer
for param in model_final_only.parameters():
    param.requires_grad = False
for param in model_final_only.fc.parameters():
    param.requires_grad = True

model_final_only.to(device)

# Model 4: Fine-tune only final block (layer4 + fc)
model_final_block = torchvision.models.resnet152(pretrained=True)
model_final_block.fc = nn.Linear(model_final_block.fc.in_features, 100)

# Freeze early layers, unfreeze layer4 and fc
for param in model_final_block.parameters():
    param.requires_grad = False
for param in model_final_block.layer4.parameters():
    param.requires_grad = True
for param in model_final_block.fc.parameters():
    param.requires_grad = True

model_final_block.to(device)

# Setup optimizers
optimizer_final_only = optim.Adam(model_final_only.fc.parameters(), lr=0.001)
optimizer_final_block = optim.Adam([
    {'params': model_final_block.layer4.parameters(), 'lr': 0.0001},
    {'params': model_final_block.fc.parameters(), 'lr': 0.001}
])

# Train comparison models
final_only_results = train_model(model_final_only, optimizer_final_only, trainloader_c100, testloader_c100, epochs_transfer, "Final Layer Only")
final_block_results = train_model(model_final_block, optimizer_final_block, trainloader_c100, testloader_c100, epochs_transfer, "Final Block")

# Compare all results
print("\n" + "=" * 60)
print("TRANSFER LEARNING COMPARISON RESULTS:")
print("=" * 60)

print(f"ImageNet Pretrained Final Accuracy: {pretrained_results[3][-1]:.2f}%")
print(f"Random Initialization Final Accuracy: {random_results[3][-1]:.2f}%")
print(f"Final Layer Only Final Accuracy: {final_only_results[3][-1]:.2f}%")
print(f"Final Block Fine-tune Final Accuracy: {final_block_results[3][-1]:.2f}%")

# Plot transfer learning comparison
plt.figure(figsize=(15, 10))

# Validation accuracy comparison
plt.subplot(2, 2, 1)
plt.plot(pretrained_results[3], label='Pretrained Full', color='blue')
plt.plot(random_results[3], label='Random Init Full', color='red')
plt.title('Pretrained vs Random Initialization')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(final_only_results[3], label='Final Layer Only', color='green')
plt.plot(final_block_results[3], label='Final Block', color='orange')
plt.plot(pretrained_results[3], label='Full Backbone', color='blue')
plt.title('Fine-tuning Strategy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.legend()

# Training loss comparison
plt.subplot(2, 2, 3)
plt.plot(pretrained_results[0], label='Pretrained Full', color='blue')
plt.plot(random_results[0], label='Random Init Full', color='red')
plt.title('Training Loss: Pretrained vs Random')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(final_only_results[0], label='Final Layer Only', color='green')
plt.plot(final_block_results[0], label='Final Block', color='orange')
plt.plot(pretrained_results[0], label='Full Backbone', color='blue')
plt.title('Training Loss: Fine-tuning Strategies')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('Figures\Task-1\transfer_learning_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Transfer learning analysis completed!")
print("=" * 60)


# ----------------------------------------------------------------------------
# ------------------------- OPTIONAL EXPERIMENTS ----------------------------
# ----------------------------------------------------------------------------

# a) Compare t-SNE vs UMAP in representing feature separability


print("Part 5 - Optional Experiments")

# Use existing features from part c
sample_size = min(300, len(feature_labels))
early_sample = early_features[:sample_size]
late_sample = late_features[:sample_size]
labels_sample = feature_labels[:sample_size]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
early_tsne = tsne.fit_transform(early_sample)
late_tsne = tsne.fit_transform(late_sample)

# Apply UMAP  
umap_model = umap.UMAP(n_components=2, random_state=42)
early_umap = umap_model.fit_transform(early_sample)
late_umap = umap_model.fit_transform(late_sample)

# Plot comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
for i in range(10):
    mask = np.array(labels_sample) == i
    plt.scatter(early_tsne[mask, 0], early_tsne[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.7)
plt.title('Early Features - t-SNE')
plt.legend()

plt.subplot(2, 2, 2)
for i in range(10):
    mask = np.array(labels_sample) == i
    plt.scatter(early_umap[mask, 0], early_umap[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.7)
plt.title('Early Features - UMAP')

plt.subplot(2, 2, 3)
for i in range(10):
    mask = np.array(labels_sample) == i
    plt.scatter(late_tsne[mask, 0], late_tsne[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.7)
plt.title('Late Features - t-SNE')

plt.subplot(2, 2, 4)
for i in range(10):
    mask = np.array(labels_sample) == i
    plt.scatter(late_umap[mask, 0], late_umap[mask, 1], 
               c=[colors[i]], label=class_names[i], alpha=0.7)
plt.title('Late Features - UMAP')

plt.tight_layout()
plt.savefig('Figures\Task-1\tsne_vs_umap.png')
plt.show()

# b) Analyze confusion between classes
print("Optional Experiment 2: Class confusion analysis")

# Get model predictions
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, targets in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(targets.numpy())

# Create confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(10), class_names, rotation=45)
plt.yticks(range(10), class_names)
plt.xlabel('Predicted')
plt.ylabel('True')

# Add numbers to plot
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center')

plt.tight_layout()
plt.savefig('Figures\Task-1\confusion_matrix.png')
plt.show()

# Find most confused pairs
confused_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i][j] > 30:
            confused_pairs.append((class_names[i], class_names[j], cm[i][j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)
print("Most confused class pairs:")
for class1, class2, count in confused_pairs[:5]:
    print(f"{class1} confused as {class2}: {count} times")

# c) Compare ResNet-152 vs ResNet-18
print("Optional Experiment 3: ResNet-152 vs ResNet-18 comparison")

# Load ResNet-18
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

# Freeze and setup
for param in resnet18.parameters():
    param.requires_grad = False
for param in resnet18.fc.parameters():
    param.requires_grad = True

resnet18.to(device)
optimizer18 = optim.Adam(resnet18.fc.parameters(), lr=0.001)

# Quick training
print("Training ResNet-18")
resnet18_acc = []
for epoch in range(3):
    resnet18.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer18.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer18.step()
    
    # Test accuracy
    resnet18.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = resnet18(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    resnet18_acc.append(acc)
    print(f'ResNet-18 Epoch {epoch+1}: {acc:.2f}%')

# Compare final accuracies
plt.figure(figsize=(8, 5))
models = ['ResNet-152', 'ResNet-18']
accuracies = [baseline_val_accs[-1], resnet18_acc[-1]]
plt.bar(models, accuracies, color=['blue', 'red'])
plt.title('Model Comparison')
plt.ylabel('Validation Accuracy (%)')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
plt.savefig('Figures\Task-1\model_comparison.png')
plt.show()

print(f"ResNet-152 final accuracy: {baseline_val_accs[-1]:.2f}%")
print(f"ResNet-18 final accuracy: {resnet18_acc[-1]:.2f}%")


print("Optional experiments completed!")
