import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from architecture import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------------------------------------------------------
# ------------------------- 1. Train the VAE ------------------------------------
# ----------------------------------------------------------------------------

print("1. Training VAE on FashionMNIST")

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model
model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# VAE Loss function
def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss, recon_loss, kl_loss

# Training
train_losses = []
val_losses = []
recon_losses = []
kl_losses = []

epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_data, mu, logvar = model(data)
        loss, recon, kl = vae_loss(recon_data, data, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            loss, _, _ = vae_loss(recon_data, data, mu, logvar)
            val_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    val_loss /= len(test_loader.dataset)
    train_recon /= len(train_loader.dataset)
    train_kl /= len(train_loader.dataset)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    recon_losses.append(train_recon)
    kl_losses.append(train_kl)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Total Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(recon_losses, label='Reconstruction', color='blue')
plt.title('Reconstruction Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(kl_losses, label='KL Divergence', color='red')
plt.title('KL Divergence')
plt.legend()

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------
# ----------- 2. Visualize Reconstructions and Generations ------------------
# ----------------------------------------------------------------------------

print("2. Visualizing Reconstructions and Generations")

model.eval()

# Get test samples for reconstruction
test_data = next(iter(test_loader))[0][:8].to(device)

with torch.no_grad():
    # Reconstructions
    recon_data, _, _ = model(test_data)
    
    # Generations from Gaussian prior
    z_gaussian = torch.randn(8, 20).to(device)
    generated_gaussian = model.decode(z_gaussian)
    
    # Generations from Laplacian prior
    z_laplacian = torch.distributions.Laplace(0, 1).sample((8, 20)).to(device)
    generated_laplacian = model.decode(z_laplacian)

# Visualization
plt.figure(figsize=(12, 6))

# Original images
for i in range(8):
    plt.subplot(3, 8, i + 1)
    plt.imshow(test_data[i].cpu().squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')

# Reconstructions
for i in range(8):
    plt.subplot(3, 8, i + 9)
    plt.imshow(recon_data[i].cpu().squeeze(), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

# Generated from Gaussian
for i in range(8):
    plt.subplot(3, 8, i + 17)
    plt.imshow(generated_gaussian[i].cpu().squeeze(), cmap='gray')
    plt.title('Gen (Gaussian)')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Compare different priors
plt.figure(figsize=(8, 4))

# Gaussian generations
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(generated_gaussian[i].cpu().squeeze(), cmap='gray')
    plt.title('Gaussian')
    plt.axis('off')

# Laplacian generations
for i in range(8):
    plt.subplot(2, 8, i + 9)
    plt.imshow(generated_laplacian[i].cpu().squeeze(), cmap='gray')
    plt.title('Laplacian')
    plt.axis('off')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------
# ------------------- 3. Posterior Collapse Investigation -------------------
# ----------------------------------------------------------------------------

print("3. Posterior Collapse Investigation")

# Analyze ELBO components
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.title('ELBO Components Over Training')
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
kl_ratio = [kl / (recon + kl) for recon, kl in zip(recon_losses, kl_losses)]
plt.plot(kl_ratio)
plt.title('KL / Total Loss Ratio')
plt.ylabel('Ratio')

plt.tight_layout()
plt.show()

# Check posterior collapse by examining latent codes
model.eval()
with torch.no_grad():
    test_batch = next(iter(test_loader))[0][:100].to(device)
    mu, logvar = model.encode(test_batch)
    
    # Check if all latent codes are similar (collapsed)
    mu_std = torch.std(mu, dim=0)
    logvar_mean = torch.mean(logvar, dim=0)
    
    print(f"Mean std of mu across samples: {torch.mean(mu_std):.6f}")
    print(f"Mean logvar: {torch.mean(logvar_mean):.6f}")

# Visualize latent space statistics
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(mu.cpu().flatten().numpy(), bins=50, alpha=0.7)
plt.title('Distribution of mu values')

plt.subplot(1, 3, 2)
plt.hist(logvar.cpu().flatten().numpy(), bins=50, alpha=0.7)
plt.title('Distribution of logvar values')

plt.subplot(1, 3, 3)
plt.plot(mu_std.cpu().numpy())
plt.title('Std of mu per dimension')
plt.ylabel('Standard deviation')
plt.xlabel('Latent dimension')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------
# --------------- 4. Mitigating Posterior Collapse -------------------------
# ----------------------------------------------------------------------------

print("4. Mitigating Posterior Collapse")

# Strategy: Beta-VAE with annealing
class BetaVAE:
    def __init__(self, model, initial_beta=0.0, final_beta=1.0, anneal_steps=1000):
        self.model = model
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.anneal_steps = anneal_steps
        self.step = 0
    
    def get_beta(self):
        if self.step >= self.anneal_steps:
            return self.final_beta
        else:
            return self.initial_beta + (self.final_beta - self.initial_beta) * (self.step / self.anneal_steps)
    
    def loss(self, x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        beta = self.get_beta()
        total_loss = recon_loss + beta * kl_loss
        
        self.step += 1
        return total_loss, recon_loss, kl_loss, beta

# Train new model with beta annealing
model_beta = VAE(latent_dim=20).to(device)
optimizer_beta = optim.Adam(model_beta.parameters(), lr=1e-3)
beta_vae = BetaVAE(model_beta, initial_beta=0.0, final_beta=1.0, anneal_steps=2000)

train_losses_beta = []
recon_losses_beta = []
kl_losses_beta = []
beta_values = []

print("Training with beta annealing")

for epoch in range(30):
    model_beta.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0
    epoch_beta = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer_beta.zero_grad()
        
        recon_data, mu, logvar = model_beta(data)
        loss, recon, kl, beta = beta_vae.loss(recon_data, data, mu, logvar)
        
        loss.backward()
        optimizer_beta.step()
        
        epoch_loss += loss.item()
        epoch_recon += recon.item()
        epoch_kl += kl.item()
        epoch_beta += beta
    
    epoch_loss /= len(train_loader.dataset)
    epoch_recon /= len(train_loader.dataset)
    epoch_kl /= len(train_loader.dataset)
    epoch_beta /= len(train_loader)
    
    train_losses_beta.append(epoch_loss)
    recon_losses_beta.append(epoch_recon)
    kl_losses_beta.append(epoch_kl)
    beta_values.append(epoch_beta)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Beta: {epoch_beta:.3f}')

# Compare results
plt.figure(figsize=(15, 10))

# Loss comparison
plt.subplot(2, 3, 1)
plt.plot(recon_losses, label='Original Recon')
plt.plot(recon_losses_beta, label='Beta Recon')
plt.title('Reconstruction Loss')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(kl_losses, label='Original KL')
plt.plot(kl_losses_beta, label='Beta KL')
plt.title('KL Divergence')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(beta_values)
plt.title('Beta Annealing Schedule')
plt.ylabel('Beta value')

# Generate samples with both models
model.eval()
model_beta.eval()

with torch.no_grad():
    # Test reconstructions
    test_sample = test_data[:4]
    recon_original, _, _ = model(test_sample)
    recon_beta, _, _ = model_beta(test_sample)
    
    # Generate from prior
    z_sample = torch.randn(4, 20).to(device)
    gen_original = model.decode(z_sample)
    gen_beta = model_beta.decode(z_sample)

# Sample comparison
plt.subplot(2, 3, 4)
grid_original = torch.cat([test_sample[:4].cpu(), recon_original.cpu()], dim=0)
grid_img = torch.zeros(2*28, 4*28)
for i in range(8):
    row = i // 4
    col = i % 4
    img = grid_original[i].squeeze()
    grid_img[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img, cmap='gray')
plt.title('Original VAE\n(Top: Real, Bottom: Recon)')
plt.axis('off')

plt.subplot(2, 3, 5)
grid_beta = torch.cat([test_sample[:4].cpu(), recon_beta.cpu()], dim=0)
grid_img_beta = torch.zeros(2*28, 4*28)
for i in range(8):
    row = i // 4
    col = i % 4
    img = grid_beta[i].squeeze()
    grid_img_beta[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img_beta, cmap='gray')
plt.title('Beta VAE\n(Top: Real, Bottom: Recon)')
plt.axis('off')

plt.subplot(2, 3, 6)
grid_gen = torch.cat([gen_original.cpu(), gen_beta.cpu()], dim=0)
grid_img_gen = torch.zeros(2*28, 4*28)
for i in range(8):
    row = i // 4
    col = i % 4
    img = grid_gen[i].squeeze()
    grid_img_gen[row*28:(row+1)*28, col*28:(col+1)*28] = img

plt.imshow(grid_img_gen, cmap='gray')
plt.title('Generated Samples\n(Top: Original, Bottom: Beta)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Final latent space analysis
with torch.no_grad():
    test_batch = next(iter(test_loader))[0][:100].to(device)
    
    # Original model
    mu_orig, logvar_orig = model.encode(test_batch)
    mu_std_orig = torch.std(mu_orig, dim=0)
    
    # Beta model
    mu_beta, logvar_beta = model_beta.encode(test_batch)
    mu_std_beta = torch.std(mu_beta, dim=0)
    
    print("Final Comparison:")
    print(f"Original VAE - Mean std of mu: {torch.mean(mu_std_orig):.6f}")
    print(f"Beta VAE - Mean std of mu: {torch.mean(mu_std_beta):.6f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(mu_std_orig.cpu().numpy(), label='Original VAE')
plt.plot(mu_std_beta.cpu().numpy(), label='Beta VAE')
plt.title('Std of mu per dimension')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(mu_orig.cpu().flatten().numpy(), bins=50, alpha=0.5, label='Original')
plt.hist(mu_beta.cpu().flatten().numpy(), bins=50, alpha=0.5, label='Beta')
plt.title('Distribution of mu values')
plt.legend()

plt.tight_layout()
plt.show()