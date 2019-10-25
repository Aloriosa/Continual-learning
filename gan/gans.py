class Fc_generator(nn.Module):
    def __init__(self, code_size, output_size):
        super(Fc_generator, self).__init__()
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
    
    
class Fc_discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fc_discriminator, self).__init__()
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
        return self.discriminator_net(x)
        
        
class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape=shape
    def forward(self,input):
        return input.view(self.shape)
      
      
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)
        
class Conv_generator(nn.Module):
    def __init__(self, code_size, output_size):
        super(Conv_generator, self).__init__()
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
    
    
class Conv_discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv_discriminator, self).__init__()
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
