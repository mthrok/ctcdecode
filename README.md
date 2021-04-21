# Simplified ctcdecode

This is a simplified version of https://github.com/parlance/ctcdecode.

For the detail of the `ctcdecode`, please checkout https://github.com/parlance/ctcdecode .

The main difference is;

* Remove KenLM support
* Remove dependencies
  * Boost
  * utf8
  * ThreadPool
* Clean-up library
* Use TorchScript for bind the C++
* Use Torch's `at::parallel_for` in place of `ThreadPool`.
* Remove unused functions
* Rename to CTCBeamSearchDecoder

## Installation

```
pip install git+https://github.com/mthrok/ctcdecode
```

**NOTE** This downloads and compiles `OpenFST`, so it takes a while.

### Requirements

* gcc
* cmake
* PyTorch 1.7<=

## Usage

**NOTE** The order and the names of arguments are different from `parlance/ctcdecode`'s version.

```python
from ctcdecode import CTCBeamSearchDecoder

decoder = CTCBeamSearchDecoder(
    labels,
    beam_size=100,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    blank_id=0,
    is_nll=False,
    num_processes=4,
)
beams, beam_lengths, scores, timesteps = decoder.decode(prob_seqs, seq_lens)
```

This decoder supports TorchScript. You should be able to deploy the dumped object in non-Python environment by loading the `libctcdecode.so` in your application.

```python
import torch

path = 'decoder.zip'
torch.jit.save(decoder, path)
decoder = torch.jit.load(path)
```
