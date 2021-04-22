#!/usr/bin/env python3
"""Generate `trn` files for Librispeech

Given a Librispeech directory, parse transcript files,
transcribe the corresponding audio, and generate hypothesis files.
"""
import os
import time
import logging
import argparse
from pathlib import Path

import torch
import torchaudio
import fairseq
import simple_ctc


_LG = logging.getLogger(__name__)


def _parse_args():
    def _path(path):
        return Path(os.path.normpath(path))

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--root-dir',
        required=True,
        type=_path,
        help='The root directory on which data are persed.'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=_path,
        help='The output directory where trn files are generated.'
    )
    parser.add_argument(
        '--model-file',
        required=True,
        type=_path,
        help='Path to a finetuned weight file.'
    )
    parser.add_argument(
        '--dict-file',
        required=True,
        type=_path,
        help='Path to `dict.ltr.txt` file.'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=4,
        help='Maximum number of threads .'
    )

    args = parser.parse_args()
    for path in [args.root_dir, args.output_dir, args.model_file, args.dict_file]:
        if not os.path.exists(path):
            raise RuntimeError(f'File or directory does not exist: {path}')
    return args


def _parse_transcript(path):
    with open(path) as trans_fileobj:
        for line in trans_fileobj:
            line = line.strip()
            if not line:
                continue
            id, transcription = line.split(' ', maxsplit=1)
            yield id, transcription


def _parse_transcriptions(root_dir, output_dir):
    _LG.info('Parsing transcriptions')
    audios = []
    trn = output_dir / 'ref.trn'
    txt = output_dir / 'ref.trans.txt'
    with open(trn, 'w') as trn_fileobj, open(txt, 'w') as txt_fileobj:
        for trans_file in root_dir.glob('**/*.trans.txt'):
            trans_dir = trans_file.parent
            for id, transcription in _parse_transcript(trans_file):
                trn_fileobj.write(f'{transcription} ({id})\n')
                txt_fileobj.write(f'{id} {transcription}\n')
                audio_path = trans_dir / f'{id}.flac'
                audios.append((id, audio_path))
    return audios


def _load_vocab(dict_file):
    tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(dict_file, mode='r', encoding='utf-8') as fileobj:
        for line in fileobj:
            tokens.append(line.split()[0])
    return tokens


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _load_model(model_file, dict_file):
    _LG.info('Loading the model')
    labels = _load_vocab(dict_file)

    overrides = {'data': str(dict_file.parent)}

    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(model_file)], arg_overrides=overrides
    )
    model = models[0].eval()

    encoder = model.w2v_encoder

    decoder = simple_ctc.BeamSearchDecoder(
        labels,
        cutoff_top_n=40,
        cutoff_prob=0.8,
        beam_size=100,
        num_processes=1,
        blank_id=0,
        is_nll=True,
    )
    _LG.info('#parameters: %s', _count_params(encoder))
    return encoder, decoder


def _decode(audios, encoder, decoder, output_dir):
    trn = output_dir / 'hyp.trn'
    trans = output_dir / 'hyp.trans.txt'
    t_enc, t_dec, num_frames = 0.0, 0.0, 0
    with open(trn, 'w') as trn_fileobj, open(trans, 'w') as txt_fileobj:
        for i, (id, path) in enumerate(audios):
            waveform, _ = torchaudio.load(path)
            mask = torch.zeros_like(waveform)

            t0 = time.monotonic()
            ir = encoder(waveform, mask)['encoder_out'].transpose(1, 0)
            t1 = time.monotonic()
            result = decoder.decode(ir)
            t2 = time.monotonic()
            trn = ''.join(result.label_sequences[0][0]).replace('|', ' ')
            trn_fileobj.write(f'{trn} ({id})\n')
            txt_fileobj.write(f'{id} {trn}\n')
            _LG.info('%d/%d: %s: %s', i, len(audios), id, trn)

            num_frames += waveform.size(1)
            t_enc += t1 - t0
            t_dec += t2 - t1
    t_audio = num_frames / 16000
    _LG.info('Audio duration:       %s [sec]', t_audio)
    _LG.info('Encoding Time:        %s [sec]', t_enc)
    _LG.info('Decoding Time:        %s [sec]', t_dec)
    _LG.info('Total Inference Time: %s [sec]', t_enc + t_dec)


def _main():
    args = _parse_args()
    torch.set_num_threads(args.num_threads)
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO)
    audios = _parse_transcriptions(args.root_dir, args.output_dir)
    encoder, decoder = _load_model(args.model_file, args.dict_file)
    _decode(audios, encoder, decoder, args.output_dir)


if __name__ == '__main__':
    _main()
