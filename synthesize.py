import argparse
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from tqdm.auto import tqdm

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, read_lexicon
from dataset import TextDataset
from text import grapheme_to_phoneme, text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_english(text, preprocess_config):
    g2p = G2p()
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = grapheme_to_phoneme(text, g2p, lexicon)
    phones = "{" + " ".join(phones) + "}"

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, dataset, configs, vocoder, batchs, temperature):
    preprocess_config, model_config, train_config = configs

    final_reduction_factor = model_config["common"]["final_reduction_factor"]

    for batch in tqdm(batchs):
        batch = to_device(batch, device)
        with torch.no_grad():
            texts, text_lengths = batch[3], batch[4]
            mel, mel_lengths, reduced_mel_lengths, alignments = model.inference(
                inputs=texts, text_lengths=text_lengths, reduction_factor=final_reduction_factor)
            mel = dataset.denormalize(mel.T.cpu(), dataset.scaler).T.to(mel.device)

            synth_samples(
                batch,
                mel,
                mel_lengths,
                reduced_mel_lengths,
                text_lengths,
                alignments,
                vocoder,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
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
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    synthesize(model, dataset, configs, vocoder, batchs, args.temperature)
