path = '<your_path_to_generated_samples>'
out_path = path.split(".txt")[0] + "_embeddings.pt"
print("Out path is ", out_path)
L = []
with open(path, 'r') as f:
    s = f.read()
while s.find('END') != -1:
    x = s.find('END')
    L.append(s[s.find('START') + len('START') :x].strip())
    s = s[x:]
    s = s[s.find('START'):]

print(L)
print(len(L))

import transformers
embedder_tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/gtr-t5-base')
embedder_model = transformers.AutoModel.from_pretrained(
    'sentence-transformers/gtr-t5-base',
    low_cpu_mem_usage=True,
    output_hidden_states=False,
).encoder
tokenized = embedder_tokenizer(L, padding = True, return_tensors = 'pt')
import torch
embeddings_list = []


def mean_pool(hidden_states, attention_mask):
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim = 1) / attention_mask.sum(dim = 1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

with torch.no_grad():
    for i in range(0, len(tokenized['input_ids']), 500):
        if i % 500 == 0:
            print(i)
        embeddings = embedder_model(input_ids = tokenized['input_ids'][i:i+500], attention_mask = tokenized['attention_mask'][i:i+500])
        pooled_embeddings = mean_pool(embeddings.last_hidden_state, tokenized['attention_mask'][i:i+500])
        assert((len(pooled_embeddings) == 500) or (i + 500 >= len(tokenized['input_ids'])))
        for j in range(len(pooled_embeddings)):
            embeddings_list.append(pooled_embeddings[j])
final_embeddings = torch.stack(embeddings_list)

print("Final embeddings shape", final_embeddings.shape)
torch.save(final_embeddings, out_path)
