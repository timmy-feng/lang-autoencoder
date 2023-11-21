import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, temperature):
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        scaled_logits = (logits + gumbel_noise) / temperature
        softmax = F.softmax(scaled_logits, dim = -1)
        ctx.save_for_backward(softmax, torch.tensor(temperature))
        samples = torch.multinomial(softmax, num_samples = 1)
        return F.one_hot(samples, num_classes = logits.shape[-1]).float()

    @staticmethod
    def backward(ctx, gradient):
        softmax, temperature = ctx.saved_tensors
        output = (softmax * (1 - softmax) * gradient) / temperature
        return output, None, None

class LangTransformer(nn.Module):
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
        sos_token = 0,
        output_seq_len = 256,
        temperature = 1,
    ):
        super().__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.sos_token = sos_token
        self.output_seq_len = output_seq_len
        self.temperature = temperature

        self.input_embed = input_embed
        self.output_embed = output_embed

        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            batch_first = True,
        )
        self.softmax = GumbelSoftmax.apply
        self.lin = nn.Linear(d_model, output_vocab_size)
    
    def forward(self, src, tgt = None):
        src = self.input_embed(src)
        if tgt is None:
            seq = torch.zeros((src.shape[0], 1, self.input_vocab_size), device = src.device)
            seq[:, 0, self.sos_token] = 1

            output = []
            for _ in range(self.output_seq_len):
                tgt = self.output_embed(seq)
                logits = self.lin(self.transformer(src, tgt)[:, -1])
                output.append(logits)
                seq = torch.cat((seq, self.softmax(logits, self.temperature)), dim = 1)
            return seq, torch.stack(output, dim = 1)
        else:
            tgt = self.output_embed(tgt)
            output = self.transformer(src, tgt)
            return None, self.lin(output)

class LangAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.softmax = GumbelSoftmax.apply
        self.decoder = decoder

    def forward(self, src):
        tokens = self.encoder(src)[0]
        return self.decoder(tokens, src)[1]