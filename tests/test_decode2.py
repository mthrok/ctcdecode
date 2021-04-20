import os

import torch
from ctcdecode import CTCBeamDecoder

WAV2VEC2_ENGLISH_LABEL = [
    '<s>',
    '<pad>',
    '</s>',
    '<unk>',
    '|',
    'E',
    'T',
    'A',
    'O',
    'N',
    'I',
    'H',
    'S',
    'R',
    'D',
    'L',
    'U',
    'M',
    'W',
    'C',
    'F',
    'G',
    'Y',
    'P',
    'B',
    'V',
    'K',
    "'",
    'X',
    'J',
    'Q',
    'Z',
]


def test_decode_wav2vec2():
    encoder_output = torch.load(
        os.path.join(
            os.path.dirname(__file__),
            'librispeech-test-clean-121-121726-0000-with-wav2vec_small_960h.pt',
        )
    )
    decoder = CTCBeamDecoder(
        WAV2VEC2_ENGLISH_LABEL,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=True,
    )
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(encoder_output)
    transcript = "".join([WAV2VEC2_ENGLISH_LABEL[n] for n in beam_results[0][0][:out_lens[0][0]]])

    assert transcript == 'ALSO|A|POPULAR|CONTRIVANCE|WHEREBY|LOVE|MAKING|MAY|BE|SUSPENDED|BUT|NOT|STOPPED|DURING|THE|PICNIC|SEASON|'