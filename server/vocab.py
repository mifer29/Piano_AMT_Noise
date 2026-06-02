def build_vocab():

    special_tokens  = ["PAD", "BOS", "EOS"]
    time_tokens     = [f"time_{i}"      for i in range(1000)]
    velocity_tokens = [f"velocity_{i}"  for i in range(128)]
    note_on_tokens  = [f"note_on_{i}"   for i in range(128)]
    note_off_tokens = [f"note_off_{i}"  for i in range(128)]
    vocab = special_tokens + time_tokens + velocity_tokens + note_on_tokens + note_off_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, token_to_id, token_to_id["PAD"], token_to_id["BOS"], token_to_id["EOS"]

vocab, token_to_id, pad_id, bos_id, eos_id = build_vocab()
