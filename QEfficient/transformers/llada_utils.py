import torch

def _sample(start_time,
            qpc_session,
            prompt,
            attention_mask,
            steps=128, 
            gen_length=128, 
            block_length=128, 
            mask_id=126336):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long)
    print('Total length input + output is ',x.shape)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    print(f'Number of blocks are {num_blocks}')

    for num_block in range(num_blocks):
        confidence_mask = torch.ones_like(x, dtype = torch.int64)
        confidence_mask[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
        confidence_mask = confidence_mask.numpy()
        x = x.numpy()

        for i in range(steps):
            inputs = dict(input_ids=x, confidence_mask = confidence_mask)
            x = qpc_session.run(inputs)['logits']
    return x