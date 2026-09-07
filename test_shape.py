import torch
next_token_logits = torch.randn(1, 128256)
next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
print(next_token_id.shape)
print(next_token_id[0])
print(next_token_id[0].item())
