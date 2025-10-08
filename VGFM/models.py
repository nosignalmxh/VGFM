import torch, torch.nn as nn

class velocityNet(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()


        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        #print(num)
        t = t.expand(num, 1)  
        #print(t)
        state  = torch.cat((t,x),dim=1)
        #print(state)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x

class growthNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
        #return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)


class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super(FNet, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)  # v = dx/dt
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)  # g

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()

        return v, g

class ODEFunc2(nn.Module):
    def __init__(self, f_net):
        super(ODEFunc2, self).__init__()
        self.f_net = f_net

    def forward(self, t, state):
        z, _= state #data0,lnw0
        v, g = self.f_net(t, z)
        
        dz_dt = v
        dlnw_dt = g
        #w = torch.exp(lnw)
        
        return dz_dt.float(), dlnw_dt.float()


class ODEFunc(nn.Module):
    def __init__(self, v_net):
        super(ODEFunc, self).__init__()
        self.v_net = v_net

    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        return dz_dt.float()

