#!/usr/bin/env bash

set -eux

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "${this_dir}"

num_threads="${1:-8}"
model_dir="${HOME}/cluster/dataset/models/wav2vec2"
librispeech_dir="${HOME}/cluster/dataset/librispeech/LibriSpeech/"
split="test-clean"
output_dir="${this_dir}/output/${split}"
mkdir -p "${output_dir}"

python "${this_dir}/evaluate_wav2vec2_librispeech.py" \
       --root-dir "${librispeech_dir}/${split}/" \
       --output-dir "${output_dir}" \
       --model-file "${model_dir}/wav2vec_small_960h.pt" \
       --dict-file "${model_dir}/dict.ltr.txt" \
       --num-threads "${num_threads}"

sclite -r "${output_dir}/ref.trn" -h "${output_dir}/hyp.trn" -i wsj -o pralign -o sum
