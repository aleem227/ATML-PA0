import torch
import torchvision
import torchvision.transforms as transforms
import clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.linalg import orthogonal_procrustes
from torch.utils.data import DataLoader

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------------------------------------------------------
# ----------------------- 1. Zero-Shot Classification on STL-10 -------------
# ----------------------------------------------------------------------------

print("1. Zero-Shot Classification on STL-10")

# Download STL-10 dataset
stl10_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=preprocess)
dataloader = DataLoader(stl10_dataset, batch_size=32, shuffle=False)

# STL-10 class names
class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# (c) Different prompting strategies
prompting_strategies = {
    'plain': class_names,
    'prompted': [f"a photo of a {name}" for name in class_names],
    'descriptive': [f"a high-resolution photograph of a {name}" for name in class_names]
}

# Evaluate each strategy
results = {}
for strategy_name, prompts in prompting_strategies.items():
    print(f"\nEvaluating {strategy_name} strategy...")
    
    # Tokenize prompts
    text_tokens = clip.tokenize(prompts).to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Encode text prompts
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        for images, labels in dataloader:
            images = images.to(device)
            
            # Encode images
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=-1)
            
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    results[strategy_name] = accuracy
    print(f"{strategy_name} accuracy: {accuracy:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
strategies = list(results.keys())
accuracies = list(results.values())

plt.bar(strategies, accuracies, color=['blue', 'green', 'orange'])
plt.title('Zero-Shot Classification Accuracy on STL-10')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
plt.show()

# ----------------------------------------------------------------------------
# ----------------------- 2. Exploring the Modality Gap --------------------
# ----------------------------------------------------------------------------

print("\n2. Exploring the Modality Gap")

# (a) Extract embeddings from a subset
subset_size = 100
subset_data = torch.utils.data.Subset(stl10_dataset, range(subset_size))
subset_loader = DataLoader(subset_data, batch_size=32, shuffle=False)

image_embeddings = []
text_embeddings = []
labels_list = []

# Use the prompted strategy for text embeddings
text_tokens = clip.tokenize(prompting_strategies['prompted']).to(device)

with torch.no_grad():
    # Get text embeddings
    text_features_all = model.encode_text(text_tokens)
    text_features_all /= text_features_all.norm(dim=-1, keepdim=True)
    
    # Get image embeddings
    for images, labels in subset_loader:
        images = images.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        image_embeddings.append(image_features.cpu())
        labels_list.extend(labels.tolist())
        
        # Add corresponding text embeddings
        for label in labels:
            text_embeddings.append(text_features_all[label].cpu())

image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.stack(text_embeddings)

print(f"Image embeddings shape: {image_embeddings.shape}")
print(f"Text embeddings shape: {text_embeddings.shape}")

# (b) Dimensionality reduction with t-SNE
print("Computing t-SNE embeddings...")
combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(combined_embeddings.numpy())

image_2d = embeddings_2d[:len(image_embeddings)]
text_2d = embeddings_2d[len(image_embeddings):]

# (c) Visualize modality distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(image_2d[:, 0], image_2d[:, 1], c='blue', alpha=0.6, label='Image embeddings', s=20)
plt.scatter(text_2d[:, 0], text_2d[:, 1], c='red', alpha=0.6, label='Text embeddings', s=20)
plt.title('Modality Gap Visualization (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()

plt.tight_layout()
plt.show()

# Compute modality gap statistics
image_mean = image_embeddings.mean(dim=0)
text_mean = text_embeddings.mean(dim=0)
modality_gap = torch.norm(image_mean - text_mean).item()
print(f"Modality gap (L2 distance between means): {modality_gap:.4f}")

# ----------------------------------------------------------------------------
# ----------------------- 3. Bridging the Modality Gap ---------------------
# ----------------------------------------------------------------------------

print("\n3. Bridging the Modality Gap")

# (b) Pair image and text embeddings (already done above)
X = image_embeddings.numpy()  # Image features
Y = text_embeddings.numpy()   # Text features

# (c) Learn optimal rotation matrix using Procrustes alignment
R, scale = orthogonal_procrustes(X, Y)
print(f"Optimal rotation matrix shape: {R.shape}")
print(f"Scale factor: {scale:.4f}")

# (d) Apply rotation transform
X_aligned = X @ R

# Convert back to tensors
image_embeddings_aligned = torch.from_numpy(X_aligned)

# (e) Visualize aligned embeddings
print("Computing t-SNE for aligned embeddings...")
combined_aligned = torch.cat([image_embeddings_aligned, text_embeddings], dim=0)
tsne_aligned = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_aligned_2d = tsne_aligned.fit_transform(combined_aligned.numpy())

image_aligned_2d = embeddings_aligned_2d[:len(image_embeddings)]
text_aligned_2d = embeddings_aligned_2d[len(image_embeddings):]

plt.figure(figsize=(15, 5))

# Before alignment
plt.subplot(1, 3, 1)
plt.scatter(image_2d[:, 0], image_2d[:, 1], c='blue', alpha=0.6, label='Image', s=20)
plt.scatter(text_2d[:, 0], text_2d[:, 1], c='red', alpha=0.6, label='Text', s=20)
plt.title('Before Alignment')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()

# After alignment
plt.subplot(1, 3, 2)
plt.scatter(image_aligned_2d[:, 0], image_aligned_2d[:, 1], c='blue', alpha=0.6, label='Image (aligned)', s=20)
plt.scatter(text_aligned_2d[:, 0], text_aligned_2d[:, 1], c='red', alpha=0.6, label='Text', s=20)
plt.title('After Procrustes Alignment')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()

# Comparison
plt.subplot(1, 3, 3)
# Compute new modality gap
image_aligned_mean = image_embeddings_aligned.mean(dim=0)
text_mean_tensor = text_embeddings.mean(dim=0)
aligned_gap = torch.norm(image_aligned_mean - text_mean_tensor).item()

gaps = [modality_gap, aligned_gap]
labels = ['Before\nAlignment', 'After\nAlignment']
plt.bar(labels, gaps, color=['red', 'green'])
plt.title('Modality Gap Comparison')
plt.ylabel('L2 Distance')
for i, gap in enumerate(gaps):
    plt.text(i, gap + 0.001, f'{gap:.4f}', ha='center')

plt.tight_layout()
plt.show()

print(f"Modality gap before alignment: {modality_gap:.4f}")
print(f"Modality gap after alignment: {aligned_gap:.4f}")
print(f"Gap reduction: {((modality_gap - aligned_gap) / modality_gap * 100):.1f}%")

# (f) Test classification accuracy with aligned embeddings
print("\nTesting classification with aligned embeddings...")

# Use a subset for quick evaluation
test_subset = torch.utils.data.Subset(stl10_dataset, range(200))
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# Original CLIP accuracy on subset
correct_original = 0
correct_aligned = 0
total_test = 0

text_tokens = clip.tokenize(prompting_strategies['prompted']).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    for images, labels in test_loader:
        images = images.to(device)
        
        # Original image features
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Original CLIP prediction
        similarity_original = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predictions_original = similarity_original.argmax(dim=-1)
        correct_original += (predictions_original == labels.to(device)).sum().item()
        
        # Aligned image features
        image_features_aligned = torch.from_numpy(image_features.cpu().numpy() @ R).to(device).float()
        image_features_aligned /= image_features_aligned.norm(dim=-1, keepdim=True)
        
        # Aligned prediction
        similarity_aligned = (100.0 * image_features_aligned @ text_features.T).softmax(dim=-1)
        predictions_aligned = similarity_aligned.argmax(dim=-1)
        correct_aligned += (predictions_aligned == labels.to(device)).sum().item()
        
        total_test += labels.size(0)

original_accuracy = correct_original / total_test
aligned_accuracy = correct_aligned / total_test

print(f"Original CLIP accuracy on subset: {original_accuracy:.3f}")
print(f"Aligned CLIP accuracy on subset: {aligned_accuracy:.3f}")

# Plot final comparison
plt.figure(figsize=(8, 6))
methods = ['Original CLIP', 'Procrustes Aligned']
accuracies_comparison = [original_accuracy, aligned_accuracy]

plt.bar(methods, accuracies_comparison, color=['blue', 'orange'])
plt.title('Classification Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies_comparison):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
plt.show()