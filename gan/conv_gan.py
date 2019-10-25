class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape=shape
    def forward(self,input):
        return input.view(self.shape)
      
      
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)
        
class GAN_generator(nn.Module):
    def __init__(self, code_size, output_size):
        super(GAN_generator, self).__init__()
        #n_features = 100
        #n_out = 784
        self.generator_net =  nn.Sequential(
            nn.Linear(code_size, 10 * 8 * 8),
            nn.LeakyReLU(0.2),
            Reshape([-1, 10, 8, 8]),
            nn.ConvTranspose2d(10, 32, kernel_size=(5,5)),
            nn.LeakyReLU(0.2),
            Flatten(),
            nn.Linear(4608, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        return self.generator_net(x)
    
    
class GAN_discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(GAN_discriminator, self).__init__()
        #n_features = 784
        #n_out = 1
        self.discriminator_net = nn.Sequential(
            nn.Linear(input_size, 10 * 8 * 8),
            nn.LeakyReLU(0.2),
            Reshape([-1, 10, 8, 8]),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(10, 32, 5),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 5, stride=2, padding=1),
            Flatten(),
            nn.Linear(121, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_size)
        )    

    def forward(self, x):
        return self.discriminator_net(x)

    
# Noise
def sample_noise_batch(batch_size, code_size):
    n = Variable(torch.randn(batch_size, code_size))
    return n 
    
def generator_loss(generator, discriminator, noise):
    generated_data = generator(noise)
    disc_on_generated_data = discriminator(generated_data)
    logp_gen_is_real = F.logsigmoid(disc_on_generated_data)
    loss = -torch.mean(logp_gen_is_real, 0)
    return loss

def discriminator_loss(discriminator, real_data, generated_data):
    disc_on_real_data = discriminator(real_data)
    disc_on_fake_data = discriminator(generated_data)
    logp_real_is_real = F.logsigmoid(disc_on_real_data)
    logp_gen_is_fake = F.logsigmoid(- disc_on_fake_data)
    loss = -torch.mean(logp_real_is_real, 0) - torch.mean(logp_gen_is_fake, 0)
    return loss
    
    
def GAN_train(discr_input, discr_output, gen_input, gen_output, batch_size, data_loader, n_epochs):
    d_losses = []
    g_losses = []
    
    discriminator = GAN_discriminator(discr_input, discr_output)
    discriminator.to(device)
    generator = GAN_generator(gen_input, gen_output)
    generator.to(device)
    
    #optimizers
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=[0.5, 0.999])
    gen_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=[0.5, 0.999])
    
    for epoch in range(n_epochs):
        if  len(g_losses) == 0 or (g_losses[-1] < d_losses[-1] * 16): #Hack? Yeah, it's hack.
            d_loss = []
            for i in range(10):
                # Train discriminator
                data_loader_iter = iter(data_loader)
                x_batch, y_batch = next(data_loader_iter)
                fake_data = generator(sample_noise_batch(batch_size, gen_input).to(device)) #gen_input==code_size
                disc_loss = discriminator_loss(discriminator, x_batch.to(device), fake_data)
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()
                d_loss.append(disc_loss.data.cpu().numpy()[0])

                
        if len(d_losses) > 0:
            d_losses.append(np.mean(d_loss) * 0.05 + d_losses[-1] * (1 - 0.05))
        else:
            d_losses.append(np.mean(d_loss))

        # Train generator
        noise = sample_noise_batch(batch_size, gen_input).to(device)
        gen_loss = generator_loss(generator, discriminator, noise)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        
        if len(g_losses) > 0:
            g_losses.append(gen_loss.data.cpu().numpy()[0] * 0.05 + g_losses[-1] * (1 - 0.05))
        else:
            g_losses.append(gen_loss.data.cpu().numpy()[0])
    
    return generator
