import torch
import torch.nn as nn

# class FANLayer(nn.Module):
#     def __init__(self, input_dim, d_p,d_p_bar, activation=nn.GELU()):
#         super().__init__()
#         self.Wp = nn.Parameter(torch.randn(input_dim, d_p))
#         self.Wp_bar = nn.Parameter(torch.randn(input_dim, d_p_bar))
#         self.Bp_bar = nn.Parameter(torch.zeros(d_p_bar))
#         self.activation = activation
        
#     def forward(self, x):
#         cos_term = torch.cos(torch.matmul(x, self.Wp))
#         sin_term = torch.sin(torch.matmul(x, self.Wp))
#         non_periodic_term = self.activation(torch.matmul(x, self.Wp_bar) + self.Bp_bar)
        
#         return torch.cat([cos_term, sin_term, non_periodic_term], dim=-1)

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias) # There is almost no difference between bias and non-bias in our experiments.
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output
    
class FAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(FAN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)   
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output


    
# class FAN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, ratio = 4, num_layers = 3, output_dim = 1, activation=nn.GELU()):  
#         super().__init__()
#         self.layers = nn.ModuleList()
        
#         d_p = hidden_dim // ratio
#         d_p_bar = hidden_dim
        
#         for _ in range(num_layers-1):
#             self.layers.append(FANLayer(input_dim, d_p, d_p_bar, activation))
#             input_dim = 2 * d_p + d_p_bar
            
#         self.WL = nn.Parameter(torch.randn(input_dim, output_dim))
#         self.BL = nn.Parameter(torch.zeros(output_dim))
        
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         # Implementation of the final layer
#         return torch.matmul(x, self.WL) + self.BL
    
