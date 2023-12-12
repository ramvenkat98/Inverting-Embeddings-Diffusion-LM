import torch
test_embeddings = torch.load('../../datasets/e2e_data/src1_test_embeddings.pt')[:500, :] # 500 samples from the test dataset
generated_embeddings_path = '<your_path_to_embeddings_of_generated_samples>'
generated_embeddings = torch.load(generated_embeddings_path)
with torch.no_grad():
    print(torch.mean(torch.nn.CosineSimilarity(dim = 1)(test_embeddings, generated_embeddings)))
