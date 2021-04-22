# Evaluating fairseq's wav2vec2.0

Using the simple CTC decoder, we can reproduce the number in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477).

## Prerequisite
* [fairseq](https://github.com/pytorch/fairseq)
* [SCTK](https://github.com/usnistgov/SCTK)
* A finetuned model file and a dictionary file downloaded from [the example page](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec).
* LibriSpeech test-clean dataset from [OpenSLR](https://www.openslr.org/12).

## Steps

The following command will run the inference on the LibriSpeech test-clean dataset and generate `ref.trn` and `hyp.trn` file.

```bash
output_dir=<path to your output directory>
python evaluate_wav2vec2_librispeech.py \
    --root-dir <path to /LibriSpeech/test-clean/> \
    --output-dir "${output_dir}" \
    --model-file <path to model file> \
    --dict-file <path to `dict.ltr.txt` file> \
    --num-threads 8
```

Then you can use `sclite` to get the WER.

```
sclite -r "${output_dir}/ref.trn" -h "${output_dir}/hyp.trn" -i wsj -o pralign -o sum
```

The SYS file contains the WER.

```
                     SYSTEM SUMMARY PERCENTAGES by SPEAKER

      ,-----------------------------------------------------------------.
      |                            /hyp.trn                             |
      |-----------------------------------------------------------------|
      | SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      |--------+--------------+-----------------------------------------|
      ...
      
      |=================================================================|
      | Sum/Avg| 2620   52576 | 96.9    2.8    0.3    0.3    3.4   38.2 |
      |=================================================================|
      |  Mean  | 67.2  1348.1 | 96.9    2.8    0.3    0.3    3.4   39.6 |
      |  S.D.  | 20.7  291.9  |  1.2    1.1    0.2    0.2    1.3   13.9 |
      | Median | 63.0  1296.0 | 96.9    2.8    0.2    0.3    3.2   37.5 |
      `-----------------------------------------------------------------'
```

The PRA file contains the comparison between hypothesis and refrences.

```
id: (237-134500-0005)
Scores: (#C #S #D #I) 8 1 0 0
REF:  oh but i'm glad to get this place MOWED
HYP:  oh but i'm glad to get this place MOULD
Eval:                                   S
```

Using `wav2vec_small_960h.pt` (Wav2Vec 2.0 Base finetuned with 960 hours Librispeech), we get WER 3.1, which is close to 3.4 in the paper.
