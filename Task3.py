import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# ---------------------------- Baseline Setup -------------------------------
# ----------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. Model Architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x.view(-1, 784))

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Weight initialization
def init_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0.0, 0.02)

init_weights(generator)
init_weights(discriminator)

# 3. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# 4. Training Loop
num_epochs = 20
G_losses = []
D_losses = []
D_x_values = []
D_G_z_values = []
fixed_noise = torch.randn(64, 100).to(device)

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Labels
        real_labels = torch.full((batch_size, 1), 0.9).to(device)
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real images
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        fake_output_g = discriminator(fake_images)
        g_labels = torch.full((batch_size, 1), 1.0).to(device)
        g_loss = criterion(fake_output_g, g_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Save losses and discriminator outputs
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        D_x_values.append(real_output.mean().item())
        D_G_z_values.append(fake_output_g.mean().item())
        
        # Print progress with D(x) and D(G(z))
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}] '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f} '
                  f'D(x): {real_output.mean().item():.3f}, D(G(z)): {fake_output_g.mean().item():.3f}')
    
    # Generate images every 5 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            fake_samples = generator(fixed_noise)
            
        plt.figure(figsize=(8, 8))
        for j in range(64):
            plt.subplot(8, 8, j+1)
            plt.imshow(fake_samples[j].cpu().squeeze() * 0.5 + 0.5, cmap='gray')
            plt.axis('off')
        plt.show()

# 5. Plot Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(G_losses, label='Generator')
plt.plot(D_losses, label='Discriminator')
plt.title('Training Losses')
plt.legend()

plt.subplot(1, 3, 2)
window = 50
G_smooth = [np.mean(G_losses[max(0, i-window):i+1]) for i in range(len(G_losses))]
D_smooth = [np.mean(D_losses[max(0, i-window):i+1]) for i in range(len(D_losses))]
plt.plot(G_smooth, label='Generator (smooth)')
plt.plot(D_smooth, label='Discriminator (smooth)')
plt.title('Smoothed Losses')
plt.legend()

plt.subplot(1, 3, 3)
D_x_smooth = [np.mean(D_x_values[max(0, i-window):i+1]) for i in range(len(D_x_values))]
D_G_z_smooth = [np.mean(D_G_z_values[max(0, i-window):i+1]) for i in range(len(D_G_z_values))]
plt.plot(D_x_smooth, label='D(x) - Real')
plt.plot(D_G_z_smooth, label='D(G(z)) - Fake')
plt.title('Discriminator Confidence')
plt.legend()

plt.show()

# 6. Final Results
with torch.no_grad():
    final_samples = generator(torch.randn(25, 100).to(device))

plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(final_samples[i].cpu().squeeze() * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
plt.title('Final Generated Digits (5x5 Grid)')
plt.show()

# Additional larger sample grid
with torch.no_grad():
    large_samples = generator(torch.randn(64, 100).to(device))

plt.figure(figsize=(10, 10))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(large_samples[i].cpu().squeeze() * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
plt.title('Generated Digits (8x8 Grid)')
plt.show()

# 7. Training Statistics
print(f"Training Statistics:")
print(f"Final Generator Loss: {G_losses[-1]:.4f}")
print(f"Final Discriminator Loss: {D_losses[-1]:.4f}")
print(f"Final D(x) on Real: {D_x_values[-1]:.3f}")
print(f"Final D(G(z)) on Fake: {D_G_z_values[-1]:.3f}")

# Loss dynamics
early_G = np.mean(G_losses[:100])
late_G = np.mean(G_losses[-100:])
early_D = np.mean(D_losses[:100])
late_D = np.mean(D_losses[-100:])

print(f"Loss Evolution:")
print(f"Generator: {early_G:.4f} -> {late_G:.4f}")
print(f"Discriminator: {early_D:.4f} -> {late_D:.4f}")


# ----------------------------------------------------------------------------
# ------------------ EXPERIMENTING WITH TRAINING ISSUES -------------------
# ----------------------------------------------------------------------------

print("Experimenting with Training Issues")

# 1. Gradient Vanishing Experiment
print("\n1. Gradient Vanishing Problem")

# Reset models for experiment
def reset_models():
    gen = Generator()
    disc = Discriminator()
    init_weights(gen)
    init_weights(disc)
    gen.to(device)
    disc.to(device)
    return gen, disc

# Experiment 1: Strong Discriminator (causes vanishing gradients)
print("Training with strong discriminator...")
gen_exp1, disc_exp1 = reset_models()

# Strong D setup - 10x higher learning rate for D
optimizer_G_exp1 = optim.Adam(gen_exp1.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_exp1 = optim.Adam(disc_exp1.parameters(), lr=0.002, betas=(0.5, 0.999))  # 10x higher

# Track losses
G_losses_exp1 = []
D_losses_exp1 = []

# Train for 10 epochs with strong D
for epoch in range(10):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        real_labels = torch.full((batch_size, 1), 1.0).to(device)  # No smoothing
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)
        
        # Train D multiple times per G update in early epochs
        d_steps = 3 if epoch < 5 else 1
        
        for _ in range(d_steps):
            optimizer_D_exp1.zero_grad()
            
            real_output = disc_exp1(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = gen_exp1(noise)
            fake_output = disc_exp1(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D_exp1.step()
        
        # Train G once
        optimizer_G_exp1.zero_grad()
        fake_output = disc_exp1(fake_images)
        g_labels = torch.full((batch_size, 1), 1.0).to(device)
        g_loss = criterion(fake_output, g_labels)
        g_loss.backward()
        optimizer_G_exp1.step()
        
        G_losses_exp1.append(g_loss.item())
        D_losses_exp1.append(d_loss.item())
        
        if i % 200 == 0:
            print(f'Epoch {epoch+1}, Batch {i}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

print("Strong D experiment completed - D overpowered G")

# Experiment 2: Mitigated Training (label smoothing + non-saturating loss)
print("\nTraining with mitigation techniques...")
gen_exp2, disc_exp2 = reset_models()

optimizer_G_exp2 = optim.Adam(gen_exp2.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_exp2 = optim.Adam(disc_exp2.parameters(), lr=0.0002, betas=(0.5, 0.999))  # Equal rates

G_losses_exp2 = []
D_losses_exp2 = []

for epoch in range(10):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Label smoothing applied
        real_labels = torch.full((batch_size, 1), 0.9).to(device)  # Smoothed
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)
        
        # Train D
        optimizer_D_exp2.zero_grad()
        
        real_output = disc_exp2(real_images)
        d_loss_real = criterion(real_output, real_labels)
        
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = gen_exp2(noise)
        fake_output = disc_exp2(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D_exp2.step()
        
        # Train G with non-saturating loss (already using this approach)
        optimizer_G_exp2.zero_grad()
        fake_output = disc_exp2(fake_images)
        g_labels = torch.full((batch_size, 1), 1.0).to(device)
        g_loss = criterion(fake_output, g_labels)  # Non-saturating: max log(D(G(z)))
        g_loss.backward()
        optimizer_G_exp2.step()
        
        G_losses_exp2.append(g_loss.item())
        D_losses_exp2.append(d_loss.item())
        
        if i % 200 == 0:
            print(f'Epoch {epoch+1}, Batch {i}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

print("Mitigation experiment completed")

# Compare results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(G_losses_exp1[:2000], label='G - Strong D', color='red')
plt.plot(D_losses_exp1[:2000], label='D - Strong D', color='blue')
plt.title('Strong Discriminator (Vanishing Gradients)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(G_losses_exp2[:2000], label='G - Mitigated', color='red')
plt.plot(D_losses_exp2[:2000], label='D - Mitigated', color='blue')
plt.title('Mitigated Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(G_losses_exp1[:2000], label='G - Strong D', color='red', linestyle='--')
plt.plot(G_losses_exp2[:2000], label='G - Mitigated', color='red')
plt.plot(D_losses_exp1[:2000], label='D - Strong D', color='blue', linestyle='--')
plt.plot(D_losses_exp2[:2000], label='D - Mitigated', color='blue')
plt.title('Comparison')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Generate samples from both experiments
with torch.no_grad():
    test_noise = torch.randn(16, 100).to(device)
    samples_exp1 = gen_exp1(test_noise)
    samples_exp2 = gen_exp2(test_noise)

plt.figure(figsize=(12, 6))

# Strong D samples
plt.subplot(2, 8, 1)
plt.text(0.5, 0.5, 'Strong D\nSamples', ha='center', va='center', fontsize=12)
plt.axis('off')

for i in range(7):
    plt.subplot(2, 8, i+2)
    plt.imshow(samples_exp1[i].cpu().squeeze() * 0.5 + 0.5, cmap='gray')
    plt.axis('off')

# Mitigated samples
plt.subplot(2, 8, 9)
plt.text(0.5, 0.5, 'Mitigated\nSamples', ha='center', va='center', fontsize=12)
plt.axis('off')

for i in range(7):
    plt.subplot(2, 8, i+10)
    plt.imshow(samples_exp2[i].cpu().squeeze() * 0.5 + 0.5, cmap='gray')
    plt.axis('off')

plt.suptitle('Generated Samples Comparison')
plt.show()

# Analysis results
print("\nObservations:")
print(f"Strong D - Final G Loss: {G_losses_exp1[-1]:.4f}")
print(f"Mitigated - Final G Loss: {G_losses_exp2[-1]:.4f}")
print(f"Strong D - G Loss Std: {np.std(G_losses_exp1[-500:]):.4f}")
print(f"Mitigated - G Loss Std: {np.std(G_losses_exp2[-500:]):.4f}")





# 2. Mode Collapse Experiment
print("\n2. Mode Collapse Problem")





gen_mode1, disc_mode1 = reset_models()


optimizer_G_mode1 = optim.Adam(gen_mode1.parameters(), lr=0.002, betas=(0.5, 0.999))  # 10x higher
optimizer_D_mode1 = optim.Adam(disc_mode1.parameters(), lr=0.0002, betas=(0.5, 0.999))

G_losses_mode1 = []
D_losses_mode1 = []

# Train for 15 epochs with strong G
for epoch in range(15):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        real_labels = torch.full((batch_size, 1), 1.0).to(device)
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)
        
        # Train D once
        optimizer_D_mode1.zero_grad()
        
        real_output = disc_mode1(real_images)
        d_loss_real = criterion(real_output, real_labels)
        
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = gen_mode1(noise)
        fake_output = disc_mode1(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D_mode1.step()
        
        # Train G multiple times (3 steps per D step)
        for _ in range(3):
            optimizer_G_mode1.zero_grad()
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = gen_mode1(noise)
            fake_output = disc_mode1(fake_images)
            g_labels = torch.full((batch_size, 1), 1.0).to(device)
            g_loss = criterion(fake_output, g_labels)
            g_loss.backward()
            optimizer_G_mode1.step()
        
        G_losses_mode1.append(g_loss.item())
        D_losses_mode1.append(d_loss.item())
        
        if i % 200 == 0:
            print(f'Epoch {epoch+1}, Batch {i}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    # Check for mode collapse every 5 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            test_samples = gen_mode1(torch.randn(16, 100).to(device))
        
        print(f"Checking diversity at epoch {epoch+1}...")
        # Simple diversity check - calculate variance
        samples_flat = test_samples.view(16, -1)
        diversity = torch.var(samples_flat, dim=0).mean().item()
        print(f"Sample diversity (variance): {diversity:.6f}")


# Experiment 2: Balanced Training (mode collapse mitigation)


print("\nTraining with balanced approach...")


gen_mode2, disc_mode2 = reset_models()


optimizer_G_mode2 = optim.Adam(gen_mode2.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_mode2 = optim.Adam(disc_mode2.parameters(), lr=0.0002, betas=(0.5, 0.999))

G_losses_mode2 = []
D_losses_mode2 = []

for epoch in range(15):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        real_labels = torch.full((batch_size, 1), 0.9).to(device)  # Label smoothing
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)
        
        # Train D twice per G step
        for _ in range(2):
            optimizer_D_mode2.zero_grad()
            
            real_output = disc_mode2(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = gen_mode2(noise)
            fake_output = disc_mode2(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D_mode2.step()
        
        # Train G once with small noise perturbation
        optimizer_G_mode2.zero_grad()
        noise = torch.randn(batch_size, 100).to(device)
        # Add small perturbation for diversity
        noise += torch.randn_like(noise) * 0.1
        fake_images = gen_mode2(noise)
        fake_output = disc_mode2(fake_images)
        g_labels = torch.full((batch_size, 1), 1.0).to(device)
        g_loss = criterion(fake_output, g_labels)
        g_loss.backward()
        optimizer_G_mode2.step()
        
        G_losses_mode2.append(g_loss.item())
        D_losses_mode2.append(d_loss.item())
        
        if i % 200 == 0:
            print(f'Epoch {epoch+1}, Batch {i}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    # Check diversity
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            test_samples = gen_mode2(torch.randn(16, 100).to(device))
        
        samples_flat = test_samples.view(16, -1)
        diversity = torch.var(samples_flat, dim=0).mean().item()
        print(f"Balanced training diversity at epoch {epoch+1}: {diversity:.6f}")


# Compare mode collapse results
plt.figure(figsize=(15, 10))

# Loss comparison
plt.subplot(2, 3, 1)
plt.plot(G_losses_mode1[:2000], label='G - Strong G', color='red')
plt.plot(D_losses_mode1[:2000], label='D - Strong G', color='blue')
plt.title('Strong Generator (Mode Collapse)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(G_losses_mode2[:2000], label='G - Balanced', color='red')
plt.plot(D_losses_mode2[:2000], label='D - Balanced', color='blue')
plt.title('Balanced Training')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(G_losses_mode1[:2000], label='G - Strong G', color='red', linestyle='--')
plt.plot(G_losses_mode2[:2000], label='G - Balanced', color='red')
plt.plot(D_losses_mode1[:2000], label='D - Strong G', color='blue', linestyle='--')
plt.plot(D_losses_mode2[:2000], label='D - Balanced', color='blue')
plt.title('Mode Collapse Comparison')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

# Sample comparison
with torch.no_grad():
    collapse_samples = gen_mode1(torch.randn(16, 100).to(device))
    diverse_samples = gen_mode2(torch.randn(16, 100).to(device))

# Strong G samples (likely collapsed)
plt.subplot(2, 3, 4)
grid_img = torch.zeros(4*28, 4*28)
for i in range(16):
    row = i // 4
    col = i % 4
    img = collapse_samples[i].cpu().squeeze() * 0.5 + 0.5
    grid_img[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img, cmap='gray')
plt.title('Strong G Samples (Mode Collapse)')
plt.axis('off')

# Balanced samples (should be diverse)
plt.subplot(2, 3, 5)
grid_img2 = torch.zeros(4*28, 4*28)
for i in range(16):
    row = i // 4
    col = i % 4
    img = diverse_samples[i].cpu().squeeze() * 0.5 + 0.5
    grid_img2[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img2, cmap='gray')
plt.title('Balanced Training Samples')
plt.axis('off')

# Diversity comparison
plt.subplot(2, 3, 6)
# Calculate diversity metrics
collapse_flat = collapse_samples.view(16, -1)
diverse_flat = diverse_samples.view(16, -1)

collapse_diversity = torch.var(collapse_flat, dim=0).mean().item()
diverse_diversity = torch.var(diverse_flat, dim=0).mean().item()

diversity_scores = [collapse_diversity, diverse_diversity]
methods = ['Strong G\n(Collapsed)', 'Balanced\n(Diverse)']

bars = plt.bar(methods, diversity_scores, color=['red', 'green'])
plt.title('Sample Diversity Comparison')
plt.ylabel('Variance (Diversity)')

for bar, score in zip(bars, diversity_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
             f'{score:.6f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


print("\n3. Discriminator Overfitting")

# Create limited dataset
limited_data = []
limited_targets = []
samples_per_class = 20  # Very limited

for class_idx in range(10):
    class_samples = [x for x, y in zip(limited_data, limited_targets) if y == class_idx]
    limited_data.extend(class_samples[:samples_per_class])
    limited_targets.extend([class_idx] * samples_per_class)

limited_data = torch.stack(limited_data)
limited_targets = torch.tensor(limited_targets)

print(f"Limited dataset size: {len(limited_data)}")

# Create validation set (held-out real data)
val_data = limited_data[2000:2500]  # 500 unseen real images

# Reset models for overfitting experiment
class OvercapacityDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x.view(-1, 784))

gen_overfit = Generator().to(device)
disc_overfit = OvercapacityDiscriminator().to(device)

optimizer_G_overfit = optim.Adam(gen_overfit.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_overfit = optim.Adam(disc_overfit.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Training on limited data
G_losses_overfit = []
D_losses_overfit = []
D_train_acc = []
D_val_acc = []

for epoch in range(100):
    epoch_G_loss = 0
    epoch_D_loss = 0
    train_correct = 0
    
    # Shuffle limited data
    perm = torch.randperm(len(limited_data))
    shuffled_data = limited_data[perm]
    
    for i in range(0, len(limited_data), batch_size):
        batch_data = shuffled_data[i:i+batch_size].to(device)
        current_batch = batch_data.size(0)
        
        # Train Discriminator extensively (causes overfitting)
        for _ in range(3):
            optimizer_D_overfit.zero_grad()
            
            # Real data
            real_labels = torch.ones(current_batch, 1).to(device)
            real_output = disc_overfit(batch_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data
            noise = torch.randn(current_batch, 100).to(device)
            fake_data = gen_overfit(noise)
            fake_labels = torch.zeros(current_batch, 1).to(device)
            fake_output = disc_overfit(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D_overfit.step()
            
            epoch_D_loss += d_loss.item()
            
            # Training accuracy
            train_correct += ((real_output > 0.5).float() == real_labels).sum().item()
            train_correct += ((fake_output < 0.5).float() == (1 - fake_labels)).sum().item()
        
        # Train Generator
        optimizer_G_overfit.zero_grad()
        noise = torch.randn(current_batch, 100).to(device)
        fake_data = gen_overfit(noise)
        output = disc_overfit(fake_data)
        g_loss = criterion(output, torch.ones(current_batch, 1).to(device))
        g_loss.backward()
        optimizer_G_overfit.step()
        
        epoch_G_loss += g_loss.item()
    
    # Validation accuracy on unseen real data
    with torch.no_grad():
        val_batch = val_data[:100].to(device)
        val_output = disc_overfit(val_batch)
        val_correct = ((val_output > 0.5).float() == torch.ones(100, 1).to(device)).sum().item()
        val_accuracy = val_correct / 100
    
    train_accuracy = train_correct / (len(limited_data) * 3 * 2)  # 3 D steps, 2 predictions each
    
    G_losses_overfit.append(epoch_G_loss)
    D_losses_overfit.append(epoch_D_loss)
    D_train_acc.append(train_accuracy)
    D_val_acc.append(val_accuracy)

# Experiment 2: Regularized Discriminator
print("Training with Dropout regularization")

class RegularizedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x.view(-1, 784))

gen_reg = Generator().to(device)
disc_reg = RegularizedDiscriminator().to(device)

optimizer_G_reg = optim.Adam(gen_reg.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_reg = optim.Adam(disc_reg.parameters(), lr=0.0002, betas=(0.5, 0.999))

G_losses_reg = []
D_losses_reg = []
D_train_acc_reg = []
D_val_acc_reg = []

for epoch in range(100):
    epoch_G_loss = 0
    epoch_D_loss = 0
    train_correct = 0
    
    perm = torch.randperm(len(limited_data))
    shuffled_data = limited_data[perm]
    
    for i in range(0, len(limited_data), batch_size):
        batch_data = shuffled_data[i:i+batch_size].to(device)
        current_batch = batch_data.size(0)
        
        # Train Discriminator (only once per batch to prevent overfitting)
        optimizer_D_reg.zero_grad()
        
        # Real data
        real_labels = torch.ones(current_batch, 1).to(device)
        real_output = disc_reg(batch_data)
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(current_batch, 100).to(device)
        fake_data = gen_reg(noise)
        fake_labels = torch.zeros(current_batch, 1).to(device)
        fake_output = disc_reg(fake_data.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D_reg.step()
        
        epoch_D_loss += d_loss.item()
        train_correct += ((real_output > 0.5).float() == real_labels).sum().item()
        train_correct += ((fake_output < 0.5).float() == (1 - fake_labels)).sum().item()
        
        # Train Generator
        optimizer_G_reg.zero_grad()
        noise = torch.randn(current_batch, 100).to(device)
        fake_data = gen_reg(noise)
        output = disc_reg(fake_data)
        g_loss = criterion(output, torch.ones(current_batch, 1).to(device))
        g_loss.backward()
        optimizer_G_reg.step()
        
        epoch_G_loss += g_loss.item()
    
    # Validation accuracy
    with torch.no_grad():
        val_batch = val_data[:100].to(device)
        disc_reg.eval()
        val_output = disc_reg(val_batch)
        disc_reg.train()
        val_correct = ((val_output > 0.5).float() == torch.ones(100, 1).to(device)).sum().item()
        val_accuracy = val_correct / 100
    
    train_accuracy = train_correct / (len(limited_data) * 2)  # 1 D step, 2 predictions
    
    G_losses_reg.append(epoch_G_loss)
    D_losses_reg.append(epoch_D_loss)
    D_train_acc_reg.append(train_accuracy)
    D_val_acc_reg.append(val_accuracy)

# Generate samples for comparison
with torch.no_grad():
    noise = torch.randn(16, 100).to(device)
    overfit_samples = gen_overfit(noise)
    reg_samples = gen_reg(noise)

# Visualization
plt.figure(figsize=(15, 10))

# Loss comparison
plt.subplot(2, 4, 1)
plt.plot(G_losses_overfit, label='Overfit G', color='red')
plt.plot(G_losses_reg, label='Regularized G', color='blue')
plt.title('Generator Loss')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(D_losses_overfit, label='Overfit D', color='red')
plt.plot(D_losses_reg, label='Regularized D', color='blue')
plt.title('Discriminator Loss')
plt.legend()

# Accuracy comparison
plt.subplot(2, 4, 3)
plt.plot(D_train_acc, label='Overfit Train', color='red')
plt.plot(D_train_acc_reg, label='Reg Train', color='blue')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(D_val_acc, label='Overfit Val', color='red')
plt.plot(D_val_acc_reg, label='Reg Val', color='blue')
plt.title('Validation Accuracy')
plt.legend()

# Sample comparison
plt.subplot(2, 4, 5)
grid_img1 = torch.zeros(4*28, 4*28)
for i in range(16):
    row = i // 4
    col = i % 4
    img = overfit_samples[i].cpu().squeeze() * 0.5 + 0.5
    grid_img1[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img1, cmap='gray')
plt.title('Overfit D Samples')
plt.axis('off')

plt.subplot(2, 4, 6)
grid_img2 = torch.zeros(4*28, 4*28)
for i in range(16):
    row = i // 4
    col = i % 4
    img = reg_samples[i].cpu().squeeze() * 0.5 + 0.5
    grid_img2[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img2, cmap='gray')
plt.title('Regularized D Samples')
plt.axis('off')

# Final accuracy comparison
plt.subplot(2, 4, 7)
final_train = [D_train_acc[-1], D_train_acc_reg[-1]]
final_val = [D_val_acc[-1], D_val_acc_reg[-1]]
methods = ['Overfit', 'Regularized']

x = range(len(methods))
plt.bar([i-0.2 for i in x], final_train, 0.4, label='Train', color='lightblue')
plt.bar([i+0.2 for i in x], final_val, 0.4, label='Validation', color='orange')
plt.xticks(x, methods)
plt.title('Final Accuracies')
plt.legend()

# Sample diversity
plt.subplot(2, 4, 8)
overfit_flat = overfit_samples.view(16, -1)
reg_flat = reg_samples.view(16, -1)

overfit_diversity = torch.var(overfit_flat, dim=0).mean().item()
reg_diversity = torch.var(reg_flat, dim=0).mean().item()

diversity_scores = [overfit_diversity, reg_diversity]
bars = plt.bar(methods, diversity_scores, color=['red', 'green'])
plt.title('Sample Diversity')
plt.ylabel('Variance')

for bar, score in zip(bars, diversity_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
             f'{score:.6f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
