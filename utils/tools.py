import re
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")
import wandb


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data) == 9:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
        step=None, losses=None, fig=None, audio=None, sampling_rate=22050,
        tag="", type="train", kl_weight=None, means=None
):
    if losses is not None:
        wandb.log({
            type + ".Loss/total_loss": losses[0],
            type + ".Loss/mel_loss": losses[1],
            type + ".Loss/kl_loss": losses[2],
            type + ".Loss/duration_loss": losses[3],
        }, step=step)
        # logger.add_scalar("Loss/total_loss", losses[0], step)
        # logger.add_scalar("Loss/mel_loss", losses[1], step)
        # logger.add_scalar("Loss/kl_loss", losses[2], step)
        # logger.add_scalar("Loss/duration_loss", losses[3], step)
    if means is not None:
        if means[0] is not None:
            wandb.log({
                type + ".Loss/post_probs_mean": means[0]
            }, step=step)
        if means[1] is not None:
            wandb.log({
                type + ".Loss/prior_probs_mean": means[1]
            }, step=step)
    if kl_weight is not None:
        wandb.log({
            type + ".Loss/kl_weight": kl_weight,
        }, step=step)

    if fig is not None:
        # logger.add_figure(tag, fig)
        wandb.log({
            tag: fig,
        }, step=step)

    if audio is not None:
        # logger.add_audio(
        #     tag,
        #     audio / max(abs(audio)),
        #     sample_rate=sampling_rate,
        # )
        wandb.log({
            tag: wandb.Audio(audio / max(abs(audio)), sample_rate=sampling_rate),
        }, step=step)


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).long().item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def synth_one_sample(targets,
                     predictions,
                     dec_alignments,
                     reduced_mel_lens,
                     vocoder,
                     preprocess_config):
    basename = targets[0][0]
    src_len = targets[4][0].item()
    mel_len = targets[7][0].item()
    reduced_mel_len = reduced_mel_lens[0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[0, :mel_len].detach().transpose(0, 1)

    attn_keys, attn_values = list(), list()
    for key, value in sorted(dec_alignments.items()):
        attn_keys.append(key)
        attn_values.append(value[0, :src_len, :reduced_mel_len].detach().transpose(-2, -1).cpu().numpy())
    attn_figs = plot_multi_attn(attn_keys, attn_values)

    fig = plot_mel(
        [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Decoder Alignment"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, attn_figs, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, pred_lens, reduced_pred_lens, text_lens, dec_alignments, vocoder,
                  preprocess_config, path):
    basenames = targets[0]
    for i in range(len(targets[0])):
        basename = basenames[i]
        src_len = text_lens[i].item()
        mel_len = pred_lens[i].item()
        reduced_mel_len = reduced_pred_lens[i].item()
        mel_prediction = predictions[i, :mel_len].detach().transpose(0, 1)

        attn_keys, attn_values = list(), list()
        for key, value in sorted(dec_alignments.items()):
            attn_keys.append(key)
            attn_values.append(value[0, :src_len, :reduced_mel_len].detach().transpose(-2, -1).cpu().numpy())
        attn_figs = plot_multi_attn(
            attn_keys, attn_values,
            save_dir=[
                os.path.join(path, "{}_attn_{}.png".format(basename, attn_idx))
                for attn_idx in range(len(attn_keys))
            ],
        )

        fig = plot_mel(
            [
                mel_prediction.cpu().numpy(),
            ],
            ["Synthetized Spectrogram"],
            save_dir=os.path.join(path, "{}.png".format(basename)),
        )

    from .model import vocoder_infer

    mel_predictions = predictions.transpose(1, 2)
    lengths = pred_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, titles, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()

    return fig


def plot_multi_attn(attn_keys, attn_values, save_dir=None):
    figs = list()
    for i, attn in enumerate(attn_values):
        fig = plt.figure()
        num_head = attn.shape[0]
        for j, head_ali in enumerate(attn):
            ax = fig.add_subplot(2, num_head // 2, j + 1)
            ax.set_xlabel('Audio timestep (reduced)') if j >= num_head - 2 else None
            ax.set_ylabel('Text timestep') if j % 2 == 0 else None
            im = ax.imshow(head_ali, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax)
        # plt.tight_layout()
        fig.suptitle(attn_keys[i], fontsize=10)
        figs.append(fig)
        if save_dir is not None:
            plt.savefig(save_dir[i])
        plt.close()

    return figs


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
