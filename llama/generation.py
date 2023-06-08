# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.model import Transformer
from llama.tokenizer import Tokenizer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        use_cpu: bool = False,
    ) -> List[str]:
        # prompts will be ["Hi", "Hello how are you?"]...
        # ["Hello", "I fine"]
        # assume max_gen_len is 2
        # number of generations -> for simplicity consider this to be 2
        bsz = len(prompts)
        params = self.model.params

        # ensure input len is less than set params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # tokenize the list of inputs
        # output will be something like [[12], [5, 6, 7, 9, 10]]
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        # max, min = 5,1
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # 7
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        if use_cpu:
            # initialize the (bsz, total_len) matrix default padding value
            # pad_id is usually -1
            tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long()
        else:
            tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        # fill the tokens matrix with the prompt tokens
        # [[12, -1, -1, -1, -1, -1]
        # [5, 6, 7, 9, 10, 56, -1]]
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        # true for all non-pad tokens
        # [[1, 0, 0, 0, 0, 0, 0]]
        # [1,1,1,1,1, 0, 0]]
        input_text_mask = tokens != self.tokenizer.pad_id

        # from where do we wanna generate
        start_pos = min_prompt_size  # 1
        prev_pos = 0

        for cur_pos in range(start_pos, total_len):
            print(f"Generate Process:{round((cur_pos/total_len)*100,2)}%")
            # take at [prev_pos:curr_pos]
            # sample inputs for 0th iteration are sliced tokens
            # [[12, 5]]
            # [5, 6]]

            # prev_pos is 1
            # curr_pos is 2

            # tokens[:, prev_pos:cur_pos]
            # [5, -1]
            # [6, 7]

            # prev_pos is 0
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # list of probs over vocab
            # sampling for next token
            # ignore this and assume greedy for now
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)

            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)  # get to single dim
            # !! only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
