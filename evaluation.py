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
import IPython


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
#     plt.savefig("final_ev.png")
#     break

########################################################################################################################
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




class contVAE(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = contEncoder(latent_dims)
        self.decoder = contDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



########################################################################################################################
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



########################################################################################################################
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


########################################################################################################################
#evaluation


cont_vae = contVAE(z_dim).to(device) # GPU
cont_vae.load_state_dict(torch.load('cont_VAE.pkl',map_location=lambda storage, loc: storage))
cont_vae.eval()


N = 3
K = 20  # one-of-K vector
temp = 1.0
hard = False
temp_min = 0.5
ANNEAL_RATE = 0.00003


disc_vae = DiscreteVAE(N, K).to(device)
disc_vae.load_state_dict(torch.load('disc_VAE.pkl',map_location=lambda storage, loc: storage))
disc_vae.eval()



N = 3
K = 20  # one-of-K vector
temp = 1.0
hard = False
temp_min = 0.5
ANNEAL_RATE = 0.00003


both_vae = bothVAE(N, K, z_dim).to(device)
both_vae.load_state_dict(torch.load('both_VAE.pkl',map_location=lambda storage, loc: storage))
both_vae.eval()


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show
    plt.savefig("latent_cont_vae.png")
    plt.clf()


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = []
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            x_hat = autoencoder.decoder(z)
            img.append(x_hat)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=12).permute(1, 2, 0).detach().cpu()
    img = img.numpy()
    plt.imshow(img, extent=[*r0, *r1])
    plt.show
    plt.savefig("reconstructed_cont_vae.png")
    plt.clf()



#continuous_vae visuals

plot_latent(cont_vae, train_loader)

plot_reconstructed(cont_vae, r0=(-4, 5), r1=(-7, 5))

for batch, _ in test_loader:
    show_batch(batch)
    plt.savefig("cont_batch.png")
    plt.clf()
    #print(batch.shape)
    batch_hat=torch.empty(batch.shape)
    for i in range(len(batch)):
        img = cont_vae(batch[i])
        img = img.view(3, 28, 28)
        batch_hat[i] = img
    show_batch(batch_hat)
    plt.savefig("cont_batch_hat.png")
    plt.clf()
    break


#discrete_vae visuals

ind = torch.zeros(N,1).long()
#print(ind)
images_list = []
for k in range(K):
    to_generate = torch.zeros(K*K,N,K)
    #print('to_generate:', to_generate)
    index = 0
    for i in range(K):
        for j in range(K):

            ind[1]=k
            ind[0]=i
            ind[2]=j
            z = F.one_hot(ind, num_classes=K).squeeze(1)
            to_generate[index]=z
            #print(to_generate)
            index += 1

    generate = to_generate.view(-1,K*N)
    #print('generate:', generate)
    reconst_images= disc_vae.decoder(generate)
    reconst_images = reconst_images.view(reconst_images.size(0), 3, 28, 28).detach()
    grid_img = torchvision.utils.make_grid(reconst_images,nrow=K).permute(1, 2, 0).numpy() * 255
    grid_img = grid_img.astype(np.uint8)
    images_list.append(Image.fromarray(grid_img))

images_list[0].save(
    'latent_disc_vae.gif',
    save_all=True,
    duration=700,
    append_images=images_list[1:],
    loop=10)
IPython.display.IFrame("latent_disc_vae.gif", width=900, height=450)

for batch, _ in test_loader:
    show_batch(batch)
    plt.savefig("disc_batch.png")
    plt.clf()
    #print(batch.shape)
    batch_hat=torch.empty(batch.shape)
    for i in range(len(batch)):
        img, label = disc_vae(batch[i], temp=1.0, hard=False)
        img = img.view(3, 28, 28)
        batch_hat[i] = img
    show_batch(batch_hat)
    plt.savefig("disc_batch_hat.png")
    plt.clf()
    break



for batch, _ in test_loader:
    show_batch(batch)
    plt.savefig("both_batch.png")
    plt.clf()
    #print(batch.shape)
    batch_hat=torch.empty(batch.shape)
    for i in range(len(batch)):
        img_disc, label, img_cont  = both_vae(batch[i], temp=1.0, hard=False)
        img_disc = img_disc.view(3, 28, 28)
        img = torch.add(img_disc, img_cont)
        img = torch.div(img,2)
        img = img.view(3, 28, 28)
        batch_hat[i] = img
    show_batch(batch_hat)
    plt.savefig("both_batch_hat.png")
    plt.clf()
    break



    #visualize latent of both

def plot_latent_both(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.cont_encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show
    plt.savefig("latent_cont_both_vae.png")
    plt.clf()



plot_latent_both(both_vae, train_loader)



ind = torch.zeros(N,1).long()
#print(ind)
images_list = []
for k in range(K):
    to_generate = torch.zeros(K*K,N,K)
    #print('to_generate:', to_generate)
    index = 0
    for i in range(K):
        for j in range(K):

            ind[1]=k
            ind[0]=i
            ind[2]=j
            z = F.one_hot(ind, num_classes=K).squeeze(1)
            to_generate[index]=z
            #print(to_generate)
            index += 1

    generate = to_generate.view(-1,K*N)
    #print('generate:', generate)
    reconst_images= both_vae.decoder(generate)
    reconst_images = reconst_images.view(reconst_images.size(0), 3, 28, 28).detach()
    grid_img = torchvision.utils.make_grid(reconst_images,nrow=K).permute(1, 2, 0).numpy() * 255
    grid_img = grid_img.astype(np.uint8)
    images_list.append(Image.fromarray(grid_img))

images_list[0].save(
    'latent_disc_both_vae.gif',
    save_all=True,
    duration=700,
    append_images=images_list[1:],
    loop=10)
IPython.display.IFrame("latent_disc_both_vae.gif", width=900, height=450)











