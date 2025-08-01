import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from model.encoder import *

class GSSG(nn.Module):
    def __init__(self, args, temporal_f, fusion_f, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(GSSG, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.temporal_f = temporal_f
        self.fusion_f = fusion_f
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        # predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, times, coeffs, rw_adj_in, rw_adj_out):
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        h0 = self.initial_h(spline.evaluate(times[0]))
        z0 = self.initial_z(spline.evaluate(times[0]))
        z = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, h0=h0, z0=z0, func_f = self.temporal_f, func_g = self.fusion_f, t=times, rw_adj_in=rw_adj_in,rw_adj_out=rw_adj_out, method=self.solver, atol=self.atol, rtol=self.rtol)
        z = z[-1:,...].transpose(0,1)
        output = self.end_conv(z)                        
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node).permute(0, 1, 3, 2)                       
        return output