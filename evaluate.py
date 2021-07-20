import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from dataset import Dataset


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, reduction_factor, length_weight, kl_weight,
             logger=None, vocoder=None, losses_len=4, device="cuda:0"):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=True, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Evaluation
    loss_sums = [0 for _ in range(losses_len)]

    # posterior statistics
    post_means_epoch, prior_means_epoch = 0, 0

    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                (predictions, mel_l2, kl_divergence, length_l2, dec_alignments, reduced_mel_lens,
                 post_probs, prior_probs) = model(
                    *(batch[2:]),
                    reduce_loss=True,
                    reduction_factor=reduction_factor
                )

                post_means_epoch += post_probs.sum().item()
                prior_means_epoch += prior_probs.sum().item()

                # Cal Loss
                total_loss = mel_l2 + kl_weight * kl_divergence + length_weight * length_l2
                losses = list([total_loss, mel_l2, kl_divergence, length_l2])

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    post_means = [post_means_epoch / len(dataset), prior_means_epoch / len(dataset)]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, KLD Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        # denormalize mels, expect [80, mel_T], hence T
        batch = list(batch)
        batch[6] = torch.stack([dataset.denormalize(batch[6][i].cpu().T, dataset.scaler).T.to(device)
                                for i in range(len(batch[6]))])
        predictions = torch.stack([dataset.denormalize(predictions[i].cpu().T, dataset.scaler).T.to(device)
                                   for i in range(len(predictions))])
        fig, attn_figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            predictions,
            dec_alignments,
            reduced_mel_lens,
            vocoder,
            preprocess_config,
        )

        log(step, losses=loss_means, type="val")
        log(
            step,
            fig=fig,
            tag=f"Validation/{tag}",
        )
        log(step, means=post_means, type='val')

        for attn_idx, attn_fig in enumerate(attn_figs):
            log(
                step,
                fig=attn_fig,
                tag=f"Validation/dec_attn_{attn_idx}/{tag}",
            )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag=f"Validation/{tag}_reconstructed",
        )
        log(
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag=f"Validation/{tag}_synthesized",
        )

    return message
