import torch
import torch.nn as nn
import torch.nn.functional as F

import math

UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

class WeightedCrossEntropyLoss(nn.Module):
    def forward(self, input, target, alpha = 1):
        batch_size = input.size(0)
        seq_len = input.size(1)

        input = input.flatten(0, 1)
        target = target.flatten()

        loss = F.cross_entropy(input, target, reduction='none')
        loss = loss.reshape(batch_size, seq_len)

        weights = (alpha ** torch.arange(0, seq_len)).view(1, -1)

        return (loss * weights).sum() / (weights.sum() * batch_size)

class StraightThroughSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, softmax):
        ctx.save_for_backward(softmax)
        seq_len = softmax.size(1)
        samples = torch.multinomial(softmax.view(-1, softmax.shape[-1]), num_samples = 1)
        samples = samples.view(-1, seq_len)
        result = F.one_hot(samples, num_classes = softmax.shape[-1]).float()
        return result
    
    @staticmethod
    def backward(ctx, gradient):
        softmax, = ctx.saved_tensors
        return softmax * gradient

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature, discrete, eps = 1e-10):
        super().__init__()

        self.temperature = temperature
        self.discrete = discrete
        self.eps = eps

        self.sample = StraightThroughSample.apply

    def gumbel_sample(self, logits):
        uniform_noise = torch.rand_like(logits, device = logits.device)
        uniform_noise = torch.clamp(uniform_noise, min = self.eps, max = 1 - self.eps)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        return gumbel_noise

    def forward(self, logits):
        assert torch.isnan(logits).sum() == 0

        gumbel_noise = self.gumbel_sample(logits) if self.training \
            else torch.zeros_like(logits, device = logits.device)

        scaled_logits = (logits + gumbel_noise) / self.temperature
        softmax = F.softmax(scaled_logits, dim = -1)

        samples = softmax if not self.discrete and self.training \
            else self.sample(softmax)

        return samples

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

class Encoder(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        input_embed,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()

        self.input_vocab_size = input_vocab_size

        self.d_model = d_model

        self.input_embed = input_embed
        self.position_encoder = PositionalEncoding(d_model)

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

    def forward(self, src):
        src = self.input_embed(src) * math.sqrt(self.d_model)
        src = self.position_encoder(src)
        output = self.encoder(src)
        return output

class TokenizedEncoder(Encoder):
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
        dropout,
    ):
        super().__init__(
            input_vocab_size,
            input_embed,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )

        self.lin = nn.Linear(d_model, output_vocab_size)
        self.output_len = output_len

    def forward(self, src):
        if src.size(1) < self.output_len:
            padding = torch.full((src.size(0), self.output_len - src.size(1)), PAD_ID, device = src.device)
            src = torch.cat((src, padding), dim = 1)

        output = super().forward(src)
        output = output[:, :self.output_len, :]
        output = self.lin(output)
        return output

class Decoder(nn.Module):
    def __init__(
        self,
        output_vocab_size,
        output_embed,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()

        self.output_vocab_size = output_vocab_size
        self.d_model = d_model

        self.output_embed = output_embed
        self.position_encoder = PositionalEncoding(d_model)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer = self.decoder_layer,
            num_layers = num_layers,
        )

        self.lin = nn.Linear(d_model, output_vocab_size)
    
    def forward(self, tgt, memory):
        tgt_mask = self.get_tgt_mask(tgt.size(1))

        tgt = self.output_embed(tgt) * math.sqrt(self.d_model)
        tgt = self.position_encoder(tgt)

        output = self.decoder(tgt, memory, tgt_mask = tgt_mask)
        output = self.lin(output)
        return output

    def get_tgt_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

class Autoencoder(nn.Module):
    def __init__(self,
        input_vocab_size,
        output_vocab_size,
        output_len,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        temperature,
        dropout,
        discrete,
        skip = 1,
    ):
        super().__init__()

        self.input_embed = nn.Linear(input_vocab_size, d_model, bias = False)
        self.output_embed = nn.Linear(output_vocab_size, d_model, bias = False)

        self.src_to_con = TokenizedEncoder(
            input_vocab_size,
            output_vocab_size,
            output_len,
            self.input_embed,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
        )

        self.con_to_vec = Encoder(
            output_vocab_size,
            self.output_embed,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
        )

        self.src_to_vec = Encoder(
            input_vocab_size,
            self.input_embed,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
        )

        self.vec_to_src = Decoder(
            input_vocab_size,
            self.input_embed,
            d_model,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )

        self.softmax = GumbelSoftmax(temperature, discrete)
        self.temperature = temperature
        self.skip = 1

    def forward(self, src):
        con_logits = self.src_to_con(src)
        con = self.softmax(con_logits)
        con_vec = self.con_to_vec(con)

        src_vec = self.src_to_vec(src)
        src_vec_dropout = torch.full_like(src_vec[:, :, :1], self.skip).bernoulli()
        src_vec = src_vec * src_vec_dropout
        
        vec = torch.cat((con_vec, src_vec), dim = 1)

        return self.vec_to_src(src[:, :-1], vec)

    def translate(self, src):
        assert not self.training
        con_logits = self.src_to_con(src)
        con = self.softmax(con_logits)
        return torch.argmax(con, dim = -1)

    def backtranslate(self, seq, length):
        raise NotImplementedError