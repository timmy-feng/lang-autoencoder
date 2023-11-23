import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DiscreteGumbelSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, temperature):
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        scaled_logits = (logits + gumbel_noise) / temperature
        softmax = F.softmax(scaled_logits, dim = -1)
        ctx.save_for_backward(softmax, temperature)
        samples = torch.multinomial(softmax, num_samples = 1)
        return F.one_hot(samples, num_classes = logits.shape[-1]).float()

    @staticmethod
    def backward(ctx, gradient):
        softmax, temperature = ctx.saved_tensors
        output = (softmax * (1 - softmax) * gradient) / temperature
        return output, None, None

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature, eps = 1e-10):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def gumbel_sample(self, logits):
        uniform_noise = torch.rand_like(logits, device = logits.device)
        uniform_noise = torch.clamp(uniform_noise, min = self.eps, max = 1 - self.eps)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        return gumbel_noise

    def forward(self, logits):
        gumbel_noise = self.gumbel_sample(logits)
        scaled_logits = (logits + gumbel_noise) / self.temperature
        softmax = F.softmax(scaled_logits, dim = -1)
        return softmax

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096):
        super().__init__()

        positional_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype = torch.float).view(-1, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        output_len,
        input_embed,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        temperature,
        dropout,
    ):
        super().__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.output_len = output_len

        self.input_embed = input_embed

        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = self.encoder_layer,
            num_layers = num_layers,
        )
        self.lin = nn.Linear(d_model, output_vocab_size)

        self.position_encoder = PositionalEncoding(d_model)
        self.softmax = GumbelSoftmax(temperature)
        self.temperature = temperature

    def forward(self, src):
        src = self.input_embed(src) * math.sqrt(self.d_model)
        src = self.position_encoder(src)

        output = self.encoder(src)[:, :self.output_len, :]
        output = self.lin(output)
        output = self.softmax(output)

        return output

class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        input_embed,
        output_embed,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.input_embed = input_embed
        self.output_embed = output_embed

        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True,
        )
        self.lin = nn.Linear(d_model, output_vocab_size)

        self.position_encoder = PositionalEncoding(d_model)
    
    def forward(self, src, tgt):
        src = self.input_embed(src) * math.sqrt(self.d_model)
        tgt = self.output_embed(tgt) * math.sqrt(self.d_model)
        src = self.position_encoder(src)
        tgt = self.position_encoder(tgt)

        tgt_mask = self.get_tgt_mask(tgt.size(1))

        output = self.transformer(src, tgt, tgt_mask = tgt_mask)
        output = self.lin(output)

        return output

    def get_tgt_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        tokens = self.encoder(src)
        return self.decoder(tokens, src[:, :-1])