import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def get_mel_spectrogram(stft, wav):
    mel, energy = stft.mel_spectrogram(wav)
    mel, energy = mel.squeeze(0), energy.squeeze(0)
    return mel, energy


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0, center=False):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        # self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', torch.hann_window(win_length))

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        # similar to librosa, reflect-pad the input
        y = torch.clip(torch.FloatTensor(y).unsqueeze(0), -1, 1)

        input_data = F.pad(
            y.unsqueeze(1),
            (int((self.filter_length - self.hop_length) / 2), int((self.filter_length - self.hop_length) / 2)),
            mode='reflect')
        input_data = input_data.squeeze(1)

        magnitudes = torch.stft(input_data, self.filter_length, hop_length=self.hop_length, win_length=self.win_length,
                                window=self.hann_window, return_complex=False,
                                center=self.center, pad_mode='reflect', normalized=False, onesided=True)

        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1) + (1e-9))

        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)

        energy = magnitudes.norm(dim=1)

        return mel_output, energy
