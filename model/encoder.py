import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)


    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x): 
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, node_num = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, node_num]).to(x)], dim=1)
        else:
            x = x
        return x
    

class Dilation_Gated_TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, dilation, node_num):
        super(Dilation_Gated_TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.node_num = node_num
        self.align = Align(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, \
                            kernel_size=(Kt, 1), enable_padding=False, dilation=dilation)
        self.dilation = dilation

    def forward(self, x):
        x_in = self.align(x)[:, :, (self.Kt - 1)*self.dilation:, :]
        x_causal_conv = self.causal_conv(x)

        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))
        return x
    
class STBlock(nn.Module):
    def __init__(self, SBlocks, TBlocks, node_num_ob, node_num_un, dropout, Kt, sem_dim, has_shallow_encode):
        super(STBlock, self).__init__()
        self.s_mlp = nn.ModuleList()
        self.t_mlp = nn.ModuleList()
        
        # for i in range(len(SBlocks)):
        #     self.s_mlp.append(Attention_MLP(node_num_ob+node_num_un, 
        #                                     SBlocks[i][0], 
        #                                     SBlocks[i][-1], 
        #                                     SBlocks[i][1], 
        #                                     dropout, 
        #                                     sem_dim, 
        #                                     has_shallow_encode[i])
        #                                     )
        for j in range(len(TBlocks)):
            dilation = 1
            self.t_mlp.append(Dilation_Gated_TemporalConvLayer(Kt, 
                                                               TBlocks[j][0], 
                                                               TBlocks[j][-1], 
                                                               dilation, 
                                                               node_num_ob+node_num_un)
                                                               )
        self.t_mlp.append(Dilation_Gated_TemporalConvLayer(2, 
                                                           TBlocks[j][0], 
                                                           TBlocks[j][-1], 
                                                           dilation, 
                                                           node_num_ob+node_num_un)
                                                           )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.SBlocks_len = len(SBlocks)
        self.TBlocks_len = len(TBlocks)
        self.node_num_ob = node_num_ob

    def forward(self, x):
        # for i in range(self.SBlocks_len):
        #     sem = self.s_mlp[i](sem)
        x_list = []
        for j in range(self.TBlocks_len + 1):
            x = self.t_mlp[j](x)
            x_list.append(x)
        x = torch.cat(x_list, dim=2)
        return x


class Temporal_Convolution_Block(nn.Module):
    def __init__(self, device, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, spatial_emb_matrix):
        super(Temporal_Convolution_Block, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.spatial_emb_matrix = spatial_emb_matrix

        self.w = nn.Parameter(spatial_emb_matrix, requires_grad=True).to(device)
        # self.w = self.w.unsqueeze(0)  # Shape becomes (1, 10, 20)
        # Replicate the tensor 64 times along the new first dimension
        self.linear_in = nn.Linear(hidden_channels+10, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        # Add a new dimension at the start
        # print(self.w.shape)
        # print(z.shape)
        w = self.w.repeat(z.shape[0], 1, 1)  # Shape becomes (64, 10, 20)
        z = torch.cat((z, w), dim=2)
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = z.tanh()
        return z


class ST_Adaptive_Fusion(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(ST_Adaptive_Fusion, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings1 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, 4, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))
            # self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=True)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z, rw_adj_in, rw_adj_out):
        z = self.linear_in(z)
        z = z.relu()
       
        if self.g_type == 'agc':
            z = self.agc(z, rw_adj_in, rw_adj_out)
        else:
            raise ValueError('Check g_type argument')
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z, rw_adj_in, rw_adj_out):
        node_num = self.node_embeddings1.shape[0]
        supports1 = F.softmax(F.relu(torch.mm(self.node_embeddings1, self.node_embeddings1.transpose(0, 1))), dim=1)
        supports2 = F.softmax(F.relu(torch.mm(self.node_embeddings2, self.node_embeddings2.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports1.device), supports1, F.softmax(rw_adj_in.mean(dim=0)/z.shape[0]), F.softmax(rw_adj_out.mean(dim=0)/z.shape[0])]
        supports = torch.stack(support_set, dim=0)
        
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings1, self.weights_pool) 
        bias = torch.matmul(self.node_embeddings1, self.bias_pool)                     
        x_g = torch.einsum("knm,bmc->bknc", supports, z)     
        x_g = x_g.permute(0, 2, 1, 3)  
        z = torch.einsum('bnki,nkio->bno', x_g, weights)+bias     
        return z

