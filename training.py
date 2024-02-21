import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import torch.nn.functional as F
import torch; torch.manual_seed(0)
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt #; plt.rcParams['figure.dpi'] = 200
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

batch_size = 128
z_dim = 2

#dataset_wrapper_class
class cMNIST(dsets.MNIST):
    def __init__(self, root, download, transform, train=True):
        super(cMNIST, self).__init__(root, train=train, download=download, transform=transform)
        #print('im in init')

    def __getitem__(self, index: int):
        #print('im in get')
        image, label = self.data[index], int(self.targets[index])
        color = torch.randint(0, 256, size=(3,))/ 255
        #color = torch.Tensor([0,0,255])
        colored_image = torch.zeros(size=(3, 28, 28), dtype=torch.uint8)              ###(3, 28, 28)
        for layer, rgb in enumerate(color):
            colored_image[layer] = image * rgb

        colored_image = transforms.functional.to_pil_image(colored_image)

        if self.transform is not None:
            colored_image = self.transform(colored_image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return colored_image, label



train_dataset = cMNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = cMNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False)


def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

# for x, _ in train_loader:
#     show_batch(x)
#     plt.savefig("final.png")
#     break

########################################################################################################################
# Continuous VAE
class contEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(contEncoder, self).__init__()
        self.linear1 = nn.Linear(784*3, 512)
        self.to_mean_logvar = nn.Linear(512, 2 * latent_dims)

    def reparametrization_trick(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x = x.view(-1, 784*3)       #torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu, log_var = torch.split(self.to_mean_logvar(x), 2, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)


class contDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(contDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784*3)      #*3

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 3, 28, 28))           #z




def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-20):
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    bs, N, K = logits.size()
    y_soft = gumbel_softmax_sample(logits.view(bs * N, K), tau=tau, eps=eps)

    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=K)

        # 1. makes the output value exactly one-hot
        # 2.makes the gradient equal to y_soft gradient
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y.reshape(bs, N * K)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784*3), reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD



class contVAE(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = contEncoder(latent_dims)
        self.decoder = contDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def cont_train(vae, data, epochs=20):
    opt = torch.optim.Adam(vae.parameters(), lr = 0.001)
    for epoch in range(epochs):
        print('epoch:', epoch)
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = vae(x)
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + vae.encoder.kl
            loss.backward()
            opt.step()
    return vae


cont_vae = contVAE(z_dim).to(device) # GPU

cont_vae = cont_train(cont_vae, train_loader, epochs=20)

torch.save(cont_vae.state_dict(), 'cont_VAE.pkl')



########################################################################################################################
# Discrete VAE
class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(DiscreteVAE, self).__init__()

        self.fc1 = nn.Linear(784*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784*3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim
        self.K = categorical_dim

    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        q = self.encoder(x.view(-1, 784*3))
        q_y = q.view(q.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K)

N = 3
K = 20  # one-of-K vector

temp = 1.0
hard = False
temp_min = 0.5
ANNEAL_RATE = 0.00003



disc_vae = DiscreteVAE(N, K).to(device)

optimizer = torch.optim.Adam(disc_vae.parameters(), lr=1e-3)



def disc_train(num_epochs=20, temp=1.0, hard=False):
    disc_vae.train()
    train_loss = 0
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_hat, qy = disc_vae(x, temp, hard)
            #print(x.shape)
            #print(x_hat.shape)
            loss = loss_function(x_hat, x, qy)
            #x_hat = x_hat.view(100,3, 28, 28)
            #print(x_hat.shape)
            loss.backward()
            train_loss += loss.item() * len(x)
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)



disc_train(20, temp, hard)

torch.save(disc_vae.state_dict(), 'disc_VAE.pkl')



########################################################################################################################
# Both VAE
class bothVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim, w_dim):
        super(bothVAE, self).__init__()

        self.fc1 = nn.Linear(784*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784*3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.N = latent_dim
        self.K = categorical_dim

        self.cont_encoder = contEncoder(w_dim)
        self.cont_decoder = contDecoder(w_dim)

    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decoder(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        w = self.cont_encoder(x)
        q = self.encoder(x.view(-1, 784*3))
        q_y = q.view(q.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(q.size(0) * self.N, self.K), self.cont_decoder(w)


N = 3
K = 20  # one-of-K vector

temp = 1.0
hard = False
temp_min = 0.5
ANNEAL_RATE = 0.00003

both = bothVAE(N, K, z_dim).to(device)

boptimizer = torch.optim.Adam(both.parameters(), lr=1e-3)


def both_train(num_epochs=20, temp=1.0, hard=False):
    both.train()
    train_loss = 0
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            boptimizer.zero_grad()
            x_hat_disc, qy, x_hat_cont  = both(x, temp, hard)
            #print(x.shape)
            #print(x_hat.shape)
            loss = loss_function(x_hat_disc, x, qy) + F.binary_cross_entropy(x_hat_cont, x, reduction='sum') + both.cont_encoder.kl
            #x_hat = x_hat.view(100,3, 28, 28)
            #print(x_hat.shape)
            loss.backward()
            train_loss += loss.item() * len(x)
            boptimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)



both_train(20, temp, hard)

torch.save(both.state_dict(), 'both_VAE.pkl')
