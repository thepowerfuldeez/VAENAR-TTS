import argparse
import os

import torch
import yaml
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from pathlib import Path
import numpy as np

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from dataset import Dataset
from synthesize import preprocess_english

from evaluate import evaluate
import wandb


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(rank, args, configs):
    if rank == 0:
        print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    if train_config['num_gpus'] > 1:
        init_process_group(backend=train_config['dist_config']['dist_backend'],
                           init_method=train_config['dist_config']['dist_url'],
                           world_size=train_config['dist_config']['world_size'] * train_config['num_gpus'], rank=rank)

    device = torch.device('cuda:{:d}'.format(rank))

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)

    train_sampler = DistributedSampler(dataset) if train_config['num_gpus'] > 1 else None
    loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size * group_size,
        # shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    if train_config['num_gpus'] > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)
        model.init = model.module.init
        model.state_dict = model.module.state_dict
    num_param = get_param_num(model)
    if rank == 0:
        print("Number of VAENAR Parameters:", num_param)

    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")

    if rank == 0:
        # Load vocoder
        vocoder = get_vocoder(model_config, device)

        # Init logger
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)
        wandb.init(project="tts", name="vaenar")

    # Training
    step = args.restore_step + 1 if not args.reset_count else 1
    epoch = step // (len(loader) * group_size)
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    intervals = train_config["alignment"]["reduce_interval"]
    rfs = train_config["alignment"]["reduction_factors"]
    length_weight = train_config["length"]["length_weight"]
    kl_weight_init = train_config["kl"]["kl_weight_init"]
    kl_weight_end = train_config["kl"]["kl_weight_end"]
    kl_weight_inc_epochs = train_config["kl"]["kl_weight_increase_epoch"]
    kl_weight_step = (kl_weight_end - kl_weight_init) / kl_weight_inc_epochs

    if rank == 0:
        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = step - 1
        outer_bar.update()

        picked_lines = Path(train_config["picked_texts_filename"]).read_text().strip("\n").split("\n\n")

        # reduction factor computation
    def _get_reduction_factor(ep):
        i = 0
        while i < len(intervals) and intervals[i] <= ep:
            i += 1
        i = i - 1 if i > 0 else 0
        return rfs[i]

    while True:
        reduction_factor = _get_reduction_factor(epoch)
        kl_weight = kl_weight_init + kl_weight_step * epoch if epoch <= kl_weight_inc_epochs else kl_weight_end

        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)


        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                if step == 1:
                    with torch.no_grad():
                        model.init(text_inputs=batch[2:][1], mel_lengths=batch[2:][5], text_lengths=batch[2:][2])

                # Forward
                (predictions, mel_l2, kl_divergence, length_l2, dec_alignments, reduced_mel_lens, *_) = model(
                    *(batch[2:]),
                    reduce_loss=True,
                    reduction_factor=reduction_factor
                )

                # Cal Loss
                total_loss = mel_l2 + kl_weight * torch.max(kl_divergence, torch.tensor(0., device=device)) \
                             + length_weight * length_l2

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                losses = [l.item() for l in list([total_loss, mel_l2, kl_divergence, length_l2])]

                if rank == 0:
                    if step % log_step == 0:
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, KLD Loss: {:.4f}, Duration Loss: {:.4f}".format(
                            *losses
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(step, losses=losses, type='train', kl_weight=kl_weight)

                    if step % synth_step == 0:
                        fig, attn_figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            predictions,
                            dec_alignments,
                            reduced_mel_lens,
                            vocoder,
                            preprocess_config,
                        )
                        log(
                            step,
                            fig=fig,
                            tag=f"Training/{tag}",
                        )
                        for attn_idx, attn_fig in enumerate(attn_figs):
                            log(
                                step,
                                fig=attn_fig,
                                tag=f"Training/dec_attn_{attn_idx}/{tag}",
                            )

                        random_text = random.choice(picked_lines)[:100]
                        ids = raw_texts = [random_text]
                        speakers = np.array([0])
                        texts = np.array([preprocess_english(random_text, preprocess_config)])
                        text_lens = np.array([len(texts[0])])
                        random_batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))

                        model.eval()
                        random_batch = to_device(random_batch, device)
                        with torch.no_grad():
                            texts, text_lenghts = random_batch[3], random_batch[4]
                            mel, mel_lengths, reduced_mel_lengths, alignments, prior_probs = model.inference(
                                inputs=texts, text_lengths=text_lenghts, reduction_factor=reduction_factor)
                            mel = dataset.denormalize(mel[0].T.cpu(), dataset.scaler).T.to(mel.device).unsqueeze(0)

                            fig, attn_figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                                [*random_batch, mel, mel_lengths],
                                mel,
                                alignments,
                                reduced_mel_lengths,
                                vocoder,
                                preprocess_config,
                            )

                            sampling_rate = preprocess_config["preprocessing"]["audio"][
                                "sampling_rate"
                            ]
                            log(step, means=[None, prior_probs.view(-1).item()], type="inference")
                            for attn_idx, attn_fig in enumerate(attn_figs):
                                log(
                                    step,
                                    fig=attn_fig,
                                    tag=f"Inference/dec_attn_{attn_idx}/{tag}",
                                )
                            log(
                                step=step,
                                audio=wav_prediction,
                                sampling_rate=sampling_rate,
                                tag=f"Inference/{tag}_synthesized",
                            )
                        model.train()

                    if step % val_step == 0:
                        model.eval()

                        message = evaluate(model, step, configs, reduction_factor, length_weight, kl_weight, val_logger,
                                           vocoder, len(losses), device)

                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    quit()
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--reset_count", action='store_true')
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    if train_config['num_gpus'] > 1:
        mp.spawn(train, nprocs=train_config['num_gpus'], args=(args, configs,))
    else:
        train(0, args, configs)
    main(args, configs)


if __name__ == "__main__":
    main()
