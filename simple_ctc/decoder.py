from typing import List, Tuple, Optional

import torch
from torch import Tensor


class BeamSearchDecoder(torch.nn.Module):
    """Beam search decoder

    The implementation was ported from

    * https://github.com/parlance/ctcdecode
    * https://github.com/PaddlePaddle/DeepSpeech

    Args:
        labels (list):
            The tokens/vocabulary used in model training. It must be ordered
            in the same way as the model's output.
        beam_size (int):
            The number of beams to retain / returned. Providing higher values
            could return beams with better scores, but it will make the search
            exponentially slower.
        cutoff_top_n (int):
            Cutoff number in pruning. Only the top ``cutoff_top_n`` labels
            with the highest probabilities will be used in the search.
        cutoff_prob (float):
            Cumulative probability threshold in pruning. When provided, labels
            with the highest probabilities, of which cumulative probability
            does not exceed this value will be retained.
        blank_id (int):
            The index of the CTC blank token used in model training.
            Typically this is 0.
        is_nll (bool):
            Indicates whether the probabilities will be given in the form of
            negative log likelihood.
        num_processes (int):
            The number of processes to parallelize the batch.
    """
    def __init__(
            self,
            labels: List[str],
            beam_size: int = 100,
            cutoff_top_n: int = 40,
            cutoff_prob: Optional[float] = None,
            blank_id: int = 0,
            is_nll: bool = False,
            num_processes: int = 4,
    ):
        super().__init__()
        self.labels = labels
        self.beam_size = beam_size
        self.cutoff_top_n = cutoff_top_n
        self.cutoff_prob = cutoff_prob
        self.blank_id = blank_id
        self.is_nll = is_nll
        self.num_processes = num_processes

    def forward(
            self,
            probs: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Performs beam search on the sequence of probabilities

        Args:
            probs (torch.Tensor):
                Sequences of probabilities (or negative log likelihood when
                ``log_probs`` is True.) over labels. The output from encoder.
                Shape: ``[batch, num_timesteps, num_labels]``.
            seq_lens (torch.Tensor, optional):
                The valid length of sequences in the batch.
                Shape: ``[batch]``.
                If not provided, ``num_timesteps`` is used for all items.

        Returns:
            Tuple of four torch.Tensors: Tuple of ``beams``, ``length``, ``scores`` and ``timesteps``
            beams:
                Integer Tensor representing the top ``n`` beams.
                Shape: ``[batch, num_beams, num_timesteps]``.
            beam_lengths:
                Integer Tensor representing the length of each beam.
                Shape: ``[batch, num_beams]``.
            scores:
                Float Tensor representing the likelihood of each beam.
                Shape: ``[batch, num_beams]``.
            timesteps:
                Integer Tensor representing the timesteps at which
                the corresponding output character has peak probability.
                Shape: ``[batch, num_beams, num_timesteps]``.
        """
        return torch.ops.simple_ctc.beam_search_decode(
            probs, seq_lens, self.labels, self.beam_size,
            self.cutoff_top_n, self.cutoff_prob,
            self.blank_id, self.is_nll,
            self.num_processes,
        )

    @torch.jit.export
    def decode(
            self,
            probs: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[List[str]]], List[List[float]], List[List[List[int]]]]:
        beams, lengths, scores, timesteps = self.forward(probs, seq_lens)

        batch_size = beams.size(0)
        scores: List[List[float]] = scores.tolist()
        # TODO: Add timesteps
        # Timesteps seems to have an issue in C++ side.
        # timesteps: List[List[List[int]]] = timesteps.tolist()
        batch_texts: List[List[List[str]]] = []
        batch_ts: List[List[List[int]]] = []
        for i in range(batch_size):
            sample_texts: List[List[str]] = []
            # sample_ts: List[List[int]] = []
            for j in range(self.beam_size):
                sample_texts.append([self.labels[k] for k in beams[i, j, :lengths[i, j]]])
                # ts: List[int] = timesteps[i][j][:lengths[i, j]]
                # sample_ts.append(ts)
            batch_texts.append(sample_texts)
            # batch_ts.append(sample_ts)

        return batch_texts, scores, batch_ts
