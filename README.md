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
* Rename the module and decoder class (`simple_ctc.BeamSearchDecoder`)
* Moved the original decode method to `forward` and replace `decode` method with high level API that performs label conversion as well.

## TODO

* Add language model back.
  * Which LM should we use?
    * [KenLM](https://github.com/kpu/kenlm) (as in the original)
    * [SRILM](http://www.speech.sri.com/projects/srilm/) (non-commerial use only without huge fee)
    * [IRSTLM](https://hlt-mt.fbk.eu/technologies/irstlm)
    * [MITLM](https://github.com/mitlm/mitlm)
* Fix timestep bug.

## Dependencies

* OpenFST (statically built/linked when installing)
* PyTorch

See [requirements.txt](./requirements.txt) for the Python package requirements.

## Installation

**NOTE** The build process downloads and compiles `OpenFST`, so it takes a while.

```
pip install git+https://github.com/mthrok/ctcdecode
```

For development

```
git clone https://github.com/mthrok/ctcdecode
cd ctcdecode
python setup.py develop
```

## Usage

**NOTE** Currently, `timesteps` is not correctly computed. so an empty list is returned.

```python
from simple_ctc import BeamSearchDecoder

decoder = BeamSearchDecoder(
    labels,
    beam_size=100,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    blank_id=0,
    is_nll=False,
    num_processes=4,
)
result = decoder.decode(prob_seqs, seq_lens)

print(result.labels[batch][beam][:])  # Resulting label sequences. 3D list.
print(result.scores[batch][beam])  # Scores of the sequences. 2D list.
print(result.timesteps[batch][beam][:])  # Timesteps of each label peak probabilities. 3D list.
```

This decoder supports TorchScript. You should be able to deploy the dumped object in non-Python environment by loading the `libctcdecode.so` in your application.

```python
import torch

path = 'decoder.zip'
torch.jit.save(decoder, path)
decoder = torch.jit.load(path)

result = decoder.decoder(prob_seqs, seq_lens)
```
