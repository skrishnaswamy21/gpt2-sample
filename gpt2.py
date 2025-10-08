import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = config.n_embd
        self.scale = (config.n_embd // config.n_head) ** -0.5

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, mask):
        w = torch.matmul(q, k.transpose(-2, -1))
        w = w * self.scale
        if mask is not None:
            w = w.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x, mask=None):
        qkv = self.c_attn(x)
        query, key, value = qkv.split(x.size(-1), dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attn_out = self._attn(query, key, value, mask)
        attn_out = self.merge_heads(attn_out)
        attn_out = self.c_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)
        return attn_out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        attn_out = self.attn(self.ln_1(x), mask=mask)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.size()
        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, mask=attention_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Example usage:
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    config = GPT2Config()
    model = GPT2Model(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompt = "The meaning of life is"
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    logits = model(input_ids)
    next_token = torch.argmax(logits[0, -1]).item()
    generated = tokenizer.decode(next_token)
    print(prompt + generated)
