# Copyright (c) 2026 Wenet Community.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# Modified for DuplexKWS
"""Window classifier with shared GRU encoding and gated latent subtraction."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseConv1d(nn.Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.left_context = kernel_size - 1
        self.convolution = nn.Conv1d(channels,
                                     channels,
                                     kernel_size,
                                     groups=channels)

    def forward(self, x):
        return self.convolution(
            F.pad(x.transpose(1, 2), (self.left_context, 0))).transpose(1, 2)


class EchoPath(nn.Module):

    def __init__(self, hidden_size, hidden_dim, kernel_size, gate_groups):
        super().__init__()
        if min(hidden_size, hidden_dim, kernel_size, gate_groups) < 1:
            raise ValueError('Echo dimensions must be positive')
        if hidden_size % gate_groups:
            raise ValueError('gate_groups must divide hidden_size')
        self.channels_per_gate = hidden_size // gate_groups
        self.input_projection = nn.Linear(4 * hidden_size, hidden_dim)
        self.activation = nn.SiLU()
        self.temporal = CausalDepthwiseConv1d(hidden_dim, kernel_size)
        self.output_projection = nn.Linear(hidden_dim, hidden_size)
        self.echo_gate_projection = nn.Linear(hidden_dim, gate_groups)
        self.residual_gate_projection = nn.Linear(2 * hidden_size, gate_groups)

    def forward(self, mixture, reference):
        pair = torch.cat(
            (reference, mixture, reference * mixture, mixture - reference),
            dim=2)
        hidden = self.temporal(self.activation(self.input_projection(pair)))
        estimated_echo = self.output_projection(hidden)
        echo_gate = torch.sigmoid(self.echo_gate_projection(hidden))
        echo_gate = echo_gate.repeat_interleave(self.channels_per_gate, dim=2)
        residual = mixture - echo_gate * estimated_echo
        residual_gate = torch.sigmoid(
            self.residual_gate_projection(torch.cat((residual, mixture),
                                                    dim=2)))
        residual_gate = residual_gate.repeat_interleave(self.channels_per_gate,
                                                        dim=2)
        return mixture - residual_gate * echo_gate * estimated_echo


class DuplexKWS(nn.Module):
    """Return [batch, 35] logits; each call resets all recurrent states.

    Inputs are [batch, frames, 64] Kaldi FBank tensors. ``lengths`` excludes
    right padding. A missing reference bypasses the entire echo branch;
    an explicitly supplied silent reference still activates that branch.
    """

    def __init__(self,
                 input_dim=64,
                 output_dim=35,
                 hidden_size=96,
                 encoder_layers=2,
                 decoder_layers=3,
                 echo_hidden_dim=70,
                 echo_kernel_size=5,
                 gate_groups=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mixture_norm = nn.BatchNorm1d(input_dim)
        self.reference_norm = nn.BatchNorm1d(input_dim)
        self.encoder = nn.GRU(input_dim,
                              hidden_size,
                              num_layers=encoder_layers,
                              batch_first=True)
        self.decoder = nn.GRU(hidden_size,
                              hidden_size,
                              num_layers=decoder_layers,
                              batch_first=True)
        self.output = nn.Linear(hidden_size, output_dim)
        self.echo_path = EchoPath(hidden_size, echo_hidden_dim,
                                  echo_kernel_size, gate_groups)

    def forward(self, mixture, reference=None, has_ref=None, lengths=None):
        if mixture.ndim != 3 or mixture.shape[2] != self.input_dim:
            raise ValueError('mixture must have shape [batch, frames, 64]')
        batch, frames, _ = mixture.shape
        if batch == 0 or frames == 0:
            raise ValueError('mixture must contain at least one frame/item')
        if lengths is None:
            lengths = torch.full((batch, ),
                                 frames,
                                 dtype=torch.long,
                                 device=mixture.device)
        if lengths.shape != (batch, ) or lengths.is_floating_point():
            raise ValueError('lengths must be an integer vector of batch size')
        lengths = lengths.to(mixture.device)
        if bool(((lengths < 1) | (lengths > frames)).any()):
            raise ValueError('lengths must be between 1 and input frames')
        if reference is not None and reference.shape != mixture.shape:
            raise ValueError('reference must have the same shape as mixture')
        if has_ref is None:
            has_ref = torch.full((batch, ),
                                 reference is not None,
                                 dtype=torch.bool,
                                 device=mixture.device)
        if has_ref.shape != (batch, ) or has_ref.dtype != torch.bool:
            raise ValueError('has_ref must be a boolean vector of batch size')
        has_ref = has_ref.to(mixture.device)
        if reference is None and bool(has_ref.any()):
            raise ValueError('active reference flags require reference input')

        normalized = self.mixture_norm(mixture.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.encoder(normalized)
        if bool(has_ref.any()):
            normalized_ref = self.reference_norm(reference[has_ref].transpose(
                1, 2)).transpose(1, 2)
            encoded_ref, _ = self.encoder(normalized_ref)
            near = self.echo_path(encoded[has_ref], encoded_ref)
            encoded = encoded.clone()
            encoded[has_ref] = near
        decoded, _ = self.decoder(encoded)
        logits = self.output(decoded)
        positions = torch.arange(frames, device=mixture.device).unsqueeze(0)
        excluded = positions >= lengths.unsqueeze(1)
        return logits.masked_fill(excluded.unsqueeze(2), float('-inf')).amax(1)
