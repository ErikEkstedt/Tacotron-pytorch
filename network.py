#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import tqdm
import time

from module import Prenet, CBHG, AttentionDecoder, SeqLinear
from text.symbols import symbols
import hyperparams as hp
from trainer import Trainer


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, embedding_size):
        """

        :param embedding_size: dimension of embedding
        """
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size)
        self.prenet = Prenet(embedding_size, hp.hidden_size * 2, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)

    def forward(self, input_):

        input_ = torch.transpose(self.embed(input_),1,2)
        prenet = self.prenet.forward(input_)
        memory = self.cbhg.forward(prenet)

        return memory


class MelDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = Prenet(hp.num_mels, hp.hidden_size * 2, hp.hidden_size)
        self.attn_decoder = AttentionDecoder(hp.hidden_size * 2)

    def forward(self, decoder_input, memory):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(decoder_input.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step

            # [GO] Frame
            prev_output = dec_input[:, :, 0]

            for i in range(timesteps):
                out = self.attn_decoder.forward(prev_output,
                        memory,
                        attn_hidden=attn_hidden,
                        gru1_hidden=gru1_hidden,
                        gru2_hidden=gru2_hidden)

                prev_output, attn_hidden, gru1_hidden, gru2_hidden = out
                outputs.append(prev_output)

                if torch.rand(1).item() < hp.teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * hp.outputs_per_step]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)

        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:,:,0]
                out = self.attn_decoder.forward(
                        prev_output,
                        memory,
                        attn_hidden=attn_hidden,
                        gru1_hidden=gru1_hidden,
                        gru2_hidden=gru2_hidden)

                prev_output, attn_hidden, gru1_hidden, gru2_hidden = out 
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

        return outputs


class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """
    def __init__(self):
        super(PostProcessingNet, self).__init__()
        self.postcbhg = CBHG(hp.hidden_size,
                             K=8,
                             projection_size=hp.num_mels,
                             is_post=True)
        self.linear = SeqLinear(hp.hidden_size * 2,
                                hp.num_freq)

    def forward(self, input_):
        out = self.postcbhg.forward(input_)
        out = self.linear.forward(torch.transpose(out,1,2))

        return out


class Tacotron(nn.Module):
    """
    End-to-end Tacotron Network
    """
    def __init__(self):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(hp.embedding_size)
        self.decoder1 = MelDecoder()
        self.decoder2 = PostProcessingNet()

    def forward(self, characters, mel_input):
        memory = self.encoder.forward(characters)
        mel_output = self.decoder1.forward(mel_input, memory)
        linear_output = self.decoder2.forward(mel_output)

        return mel_output, linear_output


class TacotronTrainer(Trainer):
    def __init__(self, model, **kwargs):
        super(model, **kwargs).__init__()

        # Loss for frequency of human register
        self.n_priority_freq = n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    def loss_fn(self, mel_output, linear_output, mel_spectrogram, linear_spectrogram):
        # Calculate loss
        mel_loss = self.criterion(mel_output, mel_spectrogram)
        linear_loss = torch.abs(linear_output-linear_spectrogram)
        linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:, :self.n_priority_freq,:])
        loss = mel_loss + linear_loss
        loss = loss.cuda()

    def train_epoch(self):
        for i, data in tqdm(enumerate(self.train_loader)):
            optimizer.zero_grad()

            # Make decoder input by concatenating [GO] Frame
            try:
                mel_input = np.concatenate( (np.zeros( [args.batch_size, hp.num_mels, 1], dtype=np.float32), data[2][:, :, 1:]), axis=2)
            except:
                raise TypeError("not same dimension")

            characters = torch.from_numpy(data[0]).long()
            mel_input = torch.from_numpy(mel_input).float()
            mel_spectrogram = torch.from_numpy(data[2]).float()
            linear_spectrogram = torch.from_numpy(data[1]).float()

            # Forward
            mel_output, linear_output = self.model.forward(characters, mel_input)
            loss, mel_loss, linear_loss = self.loss_fn(mel_output, linear_output, mel_spectrogram, linear_spectrogram)


            start_time = time.time()

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm(self.model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time

            if self.total_batches % hp.log_step == 0:
                print("time per step: %.2f sec" % time_per_step)
                print("At timestep %d" % self.total_batches)
                print("linear loss: %.4f" % linear_loss.item())
                print("mel loss: %.4f" % mel_loss.data.item())
                print("total loss: %.4f" % loss.data.item())

            if self.total_batches % hp.save_step == 0:
                self.save_checkpoint()

            if self.total_batches in hp.decay_step:
                optimizer = self.adjust_learning_rate()

    def adjust_learning_rate(self):
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr
        pass
