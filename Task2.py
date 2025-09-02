import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# ------------------------- USING PRE-TRAINED VIT ---------------------
# ----------------------------------------------------------------------------

print("Part 1 -  Using Pre-trained ViT")

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load CIFAR-10 dataset
cifar = datasets.CIFAR10(root='./data', train=False, download=True)
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Test on 3 images
for i in [0, 50, 100]:
    image, true_label = cifar[i]
    
    # Resize to 224x224 for ViT
    image = image.resize((224, 224))
    
    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f'Image {i}: {class_names[true_label]}')
    plt.axis('off')
    plt.show()
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get top prediction
    top_prediction = torch.argmax(predictions, dim=-1)
    confidence = torch.max(predictions).item()
    
    print(f"Image {i}:")
    print(f"True class: {class_names[true_label]}")
    print(f"Predicted class ID: {top_prediction.item()}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Seems reasonable: {'Yes' if confidence > 0.1 else 'Uncertain'}")
    print("-" * 30)


# ----------------------------------------------------------------------------
# ------------------------- Visualising Patch Attention ---------------------
# ----------------------------------------------------------------------------


print("\nPart 2: Visualizing Patch Attention")



# Test attention visualization on one image
test_image, test_label = cifar[100]
test_image = test_image.resize((224, 224))

print(f"Visualizing attention for: {class_names[test_label]}")

# Show original image
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_image)
plt.title('Original Image')
plt.axis('off')

# Get model output with attention weights
inputs = processor(images=test_image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # List of attention weights for each layer

# Get attention from last layer
last_layer_attention = attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
print(f"Last layer attention shape: {last_layer_attention.shape}")

# Average across heads for simplicity
averaged_attention = last_layer_attention.mean(dim=1)  # Shape: (batch_size, seq_len, seq_len)

# Get CLS token attention to patches
# CLS token is at position 0, patches are positions 1 to N
cls_attention = averaged_attention[0, 0, 1:]  # Remove CLS->CLS attention
print(f"CLS attention to patches shape: {cls_attention.shape}")

# Reshape to 2D patch grid
# For 224x224 image with 16x16 patches: 14x14 = 196 patches
patch_size = 16
num_patches = 224 // patch_size  # 14
attention_map = cls_attention.reshape(num_patches, num_patches)

print(f"Attention map shape: {attention_map.shape}")

# Show attention map
plt.subplot(1, 3, 2)
plt.imshow(attention_map.numpy(), cmap='hot', interpolation='nearest')
plt.title('Attention Map (14x14)')
plt.axis('off')

# Upsample attention map to image size and overlay
attention_upsampled = zoom(attention_map.numpy(), (224/num_patches, 224/num_patches), order=1)

# Normalize attention values
attention_upsampled = (attention_upsampled - attention_upsampled.min()) / (attention_upsampled.max() - attention_upsampled.min())

# Create overlay
plt.subplot(1, 3, 3)
plt.imshow(test_image)
plt.imshow(attention_upsampled, alpha=0.6, cmap='hot')
plt.title('Image + Attention Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print attention statistics
print(f"Max attention value: {cls_attention.max():.4f}")
print(f"Min attention value: {cls_attention.min():.4f}")
print(f"Mean attention value: {cls_attention.mean():.4f}")

print("Attention visualization completed!")


# ----------------------------------------------------------------------------
# ------------------------- Visualising Patch Attention ---------------------
# ----------------------------------------------------------------------------


print("\nPart 4: Patch Masking at Inference")

def mask_patches(image_tensor, mask_ratio=0.3, mask_type='random'):
    # image_tensor shape: (3, 224, 224)
    patch_size = 16
    num_patches_per_dim = 224 // patch_size  # 14
    
    # Create a copy
    masked_image = image_tensor.clone()
    
    if mask_type == 'random':
        # Random masking
        total_patches = num_patches_per_dim * num_patches_per_dim
        num_masked = int(total_patches * mask_ratio)
        
        # Generate random patch positions
        patch_positions = torch.randperm(total_patches)[:num_masked]
        
        for pos in patch_positions:
            row = pos // num_patches_per_dim
            col = pos % num_patches_per_dim
            
            # Mask the patch (set to gray)
            start_row = row * patch_size
            end_row = start_row + patch_size
            start_col = col * patch_size
            end_col = start_col + patch_size
            
            masked_image[:, start_row:end_row, start_col:end_col] = 0.5
            
    elif mask_type == 'center':
        # Mask center patches
        center_start = num_patches_per_dim // 4
        center_end = 3 * num_patches_per_dim // 4
        
        for row in range(center_start, center_end):
            for col in range(center_start, center_end):
                start_row = row * patch_size
                end_row = start_row + patch_size
                start_col = col * patch_size
                end_col = start_col + patch_size
                
                masked_image[:, start_row:end_row, start_col:end_col] = 0.5
    
    return masked_image

# Test masking on multiple images
test_images = [0, 50, 100]
mask_ratios = [0.1, 0.3, 0.5]
mask_types = ['random', 'center']

print("Testing robustness to patch masking...")

results = []

for img_idx in test_images:
    image, true_label = cifar[img_idx]
    image_resized = image.resize((224, 224))
    
    # Convert to tensor for masking
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_resized)
    
    # Original prediction
    inputs_orig = processor(images=image_resized, return_tensors="pt")
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
        pred_orig = torch.argmax(outputs_orig.logits, dim=-1).item()
        conf_orig = torch.max(torch.softmax(outputs_orig.logits, dim=-1)).item()
    
    print(f"\nImage {img_idx} - True: {class_names[true_label]}")
    print(f"Original prediction: {pred_orig}, Confidence: {conf_orig:.3f}")
    
    # Test different masking strategies
    for mask_type in mask_types:
        print(f"\n{mask_type.upper()} MASKING:")
        
        for mask_ratio in mask_ratios:
            # Apply masking
            masked_tensor = mask_patches(image_tensor, mask_ratio, mask_type)
            
            # Convert back to PIL for processor
            masked_image = transforms.ToPILImage()(masked_tensor)
            
            # Get prediction
            inputs_masked = processor(images=masked_image, return_tensors="pt")
            with torch.no_grad():
                outputs_masked = model(**inputs_masked)
                pred_masked = torch.argmax(outputs_masked.logits, dim=-1).item()
                conf_masked = torch.max(torch.softmax(outputs_masked.logits, dim=-1)).item()
            
            # Check if prediction changed
            prediction_changed = pred_orig != pred_masked
            confidence_drop = conf_orig - conf_masked
            
            print(f"  Mask ratio {mask_ratio}: Pred={pred_masked}, Conf={conf_masked:.3f}, "
                  f"Changed={prediction_changed}, Drop={confidence_drop:.3f}")
            
            results.append({
                'image': img_idx,
                'mask_type': mask_type,
                'mask_ratio': mask_ratio,
                'original_pred': pred_orig,
                'masked_pred': pred_masked,
                'original_conf': conf_orig,
                'masked_conf': conf_masked,
                'changed': prediction_changed,
                'conf_drop': confidence_drop
            })

# Visualize masking example
print("\nVisualizing masking examples...")
test_image, _ = cifar[0]
test_image_resized = test_image.resize((224, 224))
test_tensor = transforms.ToTensor()(test_image_resized)

# Create masked versions
random_masked = mask_patches(test_tensor, 0.3, 'random')
center_masked = mask_patches(test_tensor, 0.3, 'center')

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image_resized)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(transforms.ToPILImage()(random_masked))
plt.title('Random Masking (30%)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(transforms.ToPILImage()(center_masked))
plt.title('Center Masking')
plt.axis('off')

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*50)
print("MASKING ROBUSTNESS SUMMARY")
print("="*50)

# Calculate average confidence drops
for mask_type in mask_types:
    for mask_ratio in mask_ratios:
        subset = [r for r in results if r['mask_type'] == mask_type and r['mask_ratio'] == mask_ratio]
        avg_conf_drop = sum(r['conf_drop'] for r in subset) / len(subset)
        changed_count = sum(r['changed'] for r in subset)
        
        print(f"{mask_type.capitalize()} masking ({mask_ratio}): "
              f"Avg conf drop={avg_conf_drop:.3f}, "
              f"Predictions changed={changed_count}/{len(subset)}")


print("Patch masking analysis completed!")

# ----------------------------------------------------------------------------
# ------------------------- CLS vs Mean of Patch Token ---------------------
# ----------------------------------------------------------------------------

print("Part 5 - CLS vs Mean of Patch Token")

# Function to extract features
def extract_features(model, processor, images, pooling_method='cls'):
    all_features = []
    
    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)
            
            if pooling_method == 'cls':
                # Use CLS token (first token)
                features = last_hidden[0, 0, :].numpy()
            elif pooling_method == 'mean':
                # Use mean of patch tokens (skip CLS token)
                patch_features = last_hidden[0, 1:, :]
                features = torch.mean(patch_features, dim=0).numpy()
        
        all_features.append(features)
    
    return np.array(all_features)

# Prepare data

train_images = []
train_labels = []
test_images = []
test_labels = []

# Training data (first 300 samples for speed)
for i in range(300):
    image, label = cifar[i]
    image_resized = image.resize((224, 224))
    train_images.append(image_resized)
    train_labels.append(label)

# Test data (next 100 samples)
for i in range(300, 400):
    image, label = cifar[i]
    image_resized = image.resize((224, 224))
    test_images.append(image_resized)
    test_labels.append(label)

print(f"Training: {len(train_images)}, Testing: {len(test_images)}")

# Extract features
print("Extracting CLS features")

train_features_cls = extract_features(model, processor, train_images, 'cls')
test_features_cls = extract_features(model, processor, test_images, 'cls')

print("Extracting mean pooled features")

train_features_mean = extract_features(model, processor, train_images, 'mean')
test_features_mean = extract_features(model, processor, test_images, 'mean')

# Train linear probes
print("Training linear probes")

probe_cls = LogisticRegression(max_iter=1000, random_state=42)
probe_cls.fit(train_features_cls, train_labels)

probe_mean = LogisticRegression(max_iter=1000, random_state=42)
probe_mean.fit(train_features_mean, train_labels)

# Evaluate
pred_cls = probe_cls.predict(test_features_cls)
pred_mean = probe_mean.predict(test_features_mean)

acc_cls = accuracy_score(test_labels, pred_cls)
acc_mean = accuracy_score(test_labels, pred_mean)

print(f"\nResults:")
print(f"CLS token accuracy: {acc_cls:.4f} ({acc_cls*100:.2f}%)")
print(f"Mean pooling accuracy: {acc_mean:.4f} ({acc_mean*100:.2f}%)")

# Visualize results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
methods = ['CLS Token', 'Mean Pooling']
accuracies = [acc_cls, acc_mean]
colors = ['blue', 'red']
bars = plt.bar(methods, accuracies, color=colors)
plt.title('Linear Probe Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.hist(train_features_cls.flatten(), bins=50, alpha=0.7, color='blue', label='CLS')
plt.hist(train_features_mean.flatten(), bins=50, alpha=0.7, color='red', label='Mean')
plt.title('Feature Distributions')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()