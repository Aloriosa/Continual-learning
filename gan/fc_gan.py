class GAN_generator(nn.Module):
    def __init__(self, code_size, output_size):
        super(GAN_generator, self).__init__()
        #n_features = 100
        #n_out = 784
        self.generator_net =  nn.Sequential(
            nn.Linear(code_size, 256),
            nn.LeakyReLU(0.2),            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
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
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_size)
        )    

    def forward(self, x):
        return self.discriminator_net(x)#.to(device)
