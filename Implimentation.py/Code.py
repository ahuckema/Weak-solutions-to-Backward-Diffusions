import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. Hyperparameter & Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 10     # Leicht erhöht, um die volle Kapazität des neuen Modells auszuschöpfen
T = 1000        # Standard-Diffusionshorizont

# ==========================================
# 2. Diffusion Schedule
# ==========================================
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(torch.sqrt(alphas_cumprod), t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1. - alphas_cumprod), t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# ==========================================
# 3. Modernes DDPM UNet (GroupNorm, ResBlocks, SiLU)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """Vollwertiger Residual Block mit GroupNorm und Zeit-Injektion"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.act = nn.SiLU()
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act(self.norm1(x)))
        # Zeit-Einbettung addieren
        time_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.conv2(self.act(self.norm2(h)))
        # Residual Connection
        return h + self.residual_conv(x)

class ModernUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 128
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        self.init_conv = nn.Conv2d(1, 32, 3, padding=1)
        
        # Encoder
        self.down1 = ResBlock(32, 64, time_dim)
        self.down2 = ResBlock(64, 128, time_dim)
        
        # Bottleneck
        self.mid1 = ResBlock(128, 128, time_dim)
        self.mid2 = ResBlock(128, 128, time_dim)
        
        # Decoder (Kanalanzahl = Upsample + Skip Connection)
        self.up1 = ResBlock(128 + 128, 64, time_dim)
        self.up2 = ResBlock(64 + 64, 32, time_dim)
        
        self.out_conv = nn.Conv2d(32, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x0 = self.init_conv(x)
        
        x1 = self.down1(x0, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        
        x_mid = self.mid1(self.pool(x2), t_emb)
        x_mid = self.mid2(x_mid, t_emb)
        
        x_up = self.upsample(x_mid)
        x_up = torch.cat([x_up, x2], dim=1) # Skip connection 2
        x_up = self.up1(x_up, t_emb)
        
        x_up = self.upsample(x_up)
        x_up = torch.cat([x_up, x1], dim=1) # Skip connection 1
        x_up = self.up2(x_up, t_emb)
        
        return self.out_conv(x_up)

# ==========================================
# 4. Training
# ==========================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = ModernUNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
loss_fn = nn.MSELoss()

saved_models = {}

print("Starte Training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, (images, _) in enumerate(dataloader):
        images = images.to(device)
        
        t = torch.randint(0, T, (images.shape[0],), device=device).long()
        noise = torch.randn_like(images)
        x_noisy = q_sample(images, t, noise)
        
        predicted_noise = model(x_noisy, t)
        loss = loss_fn(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoche {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    
    # Speichern: 1. Epoche, Mitte und Finale Epoche
    if epoch == 0 or epoch == (epochs//2) or epoch == epochs - 1:
        saved_models[f"Epoche {epoch+1}"] = {k: v.cpu() for k, v in model.state_dict().items()}

# ==========================================
# 5. Reverse Process Generation & Plot
# ==========================================
print("\nGeneriere Bilder für den finalen Plot...")

fig, axes = plt.subplots(nrows=len(saved_models), ncols=5, figsize=(15, 3 * len(saved_models)))
fig.suptitle("Rückwärtsprozess: Von Rauschen zur Ziffer", fontsize=16)

fixed_noise = torch.randn((1, 1, 28, 28), device=device)
col_titles = [f"t={T} (Rauschen)", f"t={int(T*0.75)}", f"t={int(T*0.5)}", f"t={int(T*0.25)}", "t=0 (Finale Ziffer)"]

with torch.no_grad(): 
    for row_idx, (epoch_name, state_dict) in enumerate(saved_models.items()):
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()
        
        img = fixed_noise.clone()
        intermediate_images = [img.detach().cpu().squeeze().numpy()]
        save_steps = [int(T*0.75), int(T*0.5), int(T*0.25), 0]
        
        for i in reversed(range(0, T)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            pred_noise = model(img, t)
            
            alpha = extract(alphas, t, img.shape)
            alpha_cumprod = extract(alphas_cumprod, t, img.shape)
            beta = extract(betas, t, img.shape)
            
            noise = torch.randn_like(img) if i > 0 else torch.zeros_like(img)
            img = 1 / torch.sqrt(alpha) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * pred_noise) + torch.sqrt(beta) * noise
            
            if i in save_steps:
                intermediate_images.append(img.detach().cpu().squeeze().numpy())
                
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            ax.imshow(intermediate_images[col_idx], cmap='gray')
            ax.axis('off')
            
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            if col_idx == 0:
                ax.text(-5, 14, epoch_name, rotation=90, va='center', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()