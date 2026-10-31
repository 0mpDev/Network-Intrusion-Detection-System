import torch, torch.nn as nn

def _mlp(sizes, last_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if last_act is not None:
        layers.append(last_act)
    return nn.Sequential(*layers)

class AE(nn.Module):
    """Plain autoencoder baseline."""
    def __init__(self, d, latent=16):
        super().__init__()
        self.enc = _mlp([d, 64, 32, latent])
        self.dec = _mlp([latent, 32, 64, d], nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))

class VAE(nn.Module):
    """Variational autoencoder: encoder -> (mu, logvar) -> sample z -> decoder."""
    def __init__(self, d, latent=16):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)
        self.dec = _mlp([latent, 32, 64, d], nn.Sigmoid())
    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)      # reparameterization trick
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

class GANomalyG(nn.Module):
    """GANomaly generator: encoder1 -> decoder -> encoder2."""
    def __init__(self, d, latent=16):
        super().__init__()
        self.enc1 = _mlp([d, 64, 32, latent])
        self.dec = _mlp([latent, 32, 64, d], nn.Sigmoid())
        self.enc2 = _mlp([d, 64, 32, latent])
    def forward(self, x):
        z = self.enc1(x)
        x_hat = self.dec(z)
        z_hat = self.enc2(x_hat)
        return x_hat, z, z_hat

class GANomalyD(nn.Module):
    """Discriminator; also returns intermediate features for feature-matching loss."""
    def __init__(self, d):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(0.2),
                                      nn.Linear(64, 32), nn.LeakyReLU(0.2))
        self.classifier = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        f = self.features(x)
        return self.classifier(f).squeeze(1), f

# ---- anomaly scores (higher = more anomalous) ----
@torch.no_grad()
def score_ae(model, X):
    model.eval()
    return ((model(X) - X) ** 2).mean(1)

@torch.no_grad()
def score_vae(model, X):
    model.eval()
    mu, _ = model.encode(X)
    return ((model.dec(mu) - X) ** 2).mean(1)

@torch.no_grad()
def score_ganomaly(G, X):
    G.eval()
    _, z, z_hat = G(X)
    return ((z - z_hat) ** 2).mean(1)
