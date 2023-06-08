# %%
from example import load, setup_model_parallel
from llama.tokenizer import Tokenizer

# %%
# constants
max_seq_len = 512
max_batch_size = 1
ckpt_dir = "../weights/7B"
tokenizer_path = "../weights/tokenizer.model"
use_cpu = True
# %%
# testing block
tokenizer = Tokenizer(model_path=tokenizer_path)


# %%

local_rank, world_size = setup_model_parallel(use_cpu=use_cpu)

# %%
generator, model = load(
    ckpt_dir,
    tokenizer_path,
    local_rank=local_rank,
    world_size=world_size,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    use_cpu=use_cpu,
)

# %%
results = generator.generate(
    ["hello how are you"], max_gen_len=256, temperature=0, top_p=1.0, use_cpu=use_cpu
)
print(results)
