#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1

set -eux

base_dir="/fsx/users/moto"
model_dir="${base_dir}/dataset/models/wav2vec2"
librispeech_dir="${base_dir}/dataset/librispeech/LibriSpeech"
repo_dir="${base_dir}/ctcdecode/example/evalutate_wav2vec2"
base_output="${repo_dir}/output"

for num_threads in 32 16 8 4 2 1; do
    for split in test-clean test-other; do
        for model in wav2vec_small_960h wav2vec_big_960h; do
            output_dir="${base_output}/${split}/${model}/${num_threads}"
            mkdir -p "${output_dir}"
            srun --cpus-per-task=${num_threads} --time=2:00:00 \
                 python "${repo_dir}/evaluate_wav2vec2_librispeech.py" \
                     --root-dir "${librispeech_dir}/${split}/" \
                     --output-dir "${output_dir}" \
                     --model-file "${model_dir}/${model}.pt" \
                     --dict-file "${model_dir}/dict.ltr.txt" \
                     --num-threads "${num_threads}" \
                     > "${output_dir}/log.txt" 2>&1
        done
    done
done
