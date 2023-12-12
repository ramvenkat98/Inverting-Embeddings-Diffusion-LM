# Adapting Diffusion-LM for Embedding Inversion

This was done as part of the final project for CS236. We apply classifier-free-guidance to the recently introduced diffusion-LM model and study the performance on an embedding inversion task. 

The command to train such a model and reproduce the result in the project report is as follows:
```
OPENAI_LOGDIR=<output_dir_path>  TOKENIZERS_PARALLELISM=false python scripts/train.py   --checkpoint_path <output_dir_path> --model_arch transformer --modality e2e-tgt --save_interval 10000 --lr 0.0001 --batch_size 64  --diffusion_steps 2000 --noise_schedule sqrt  --use_kl False --learn_sigma False  --image_size 8 --num_channels 128 --seed 102 --dropout 0.1 --in_channel 256 --out_channel 256 --padding_mode pad --experiment random  --lr_anneal_steps 200000 --weight_decay 0.0 --num_res_blocks 2  --predict_xstart True --training_mode e2e --vocab_size 1024 --cfg True --e2e_train ../datasets/e2e_data
```

This will start training, and save relevant model parameters, weights, etc. to the specified output directory. To sample conditioned on some embeddings, we can run
```
python scripts/text_sample.py --model_path <model_path_in_out_dir> --batch_size 500 --num_samples 500 --top_p -1.0 --out_dir <sample_out_dir> --embedding_path <path_to_embeddings_to_invert>
```

You can then use `get_embeddings.py` and `compute_similarity.py` accordingly to evaluate the generated samples and find their cosine similarity.


## Everything below this is from the original repo, only the changes above are from our fork.
---
## Diffusion-LM Improves Controllable Text Generation

https://arxiv.org/pdf/2205.14217.pdf 



-----------------------------------------------------
## Conda Setup:
```python 
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb
```

-----------------------------------------------------
## Train Diffusion-LM:

```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000  --seed 101 --noise_schedule sqrt  --in_channel 128 --modality roc --submit no --padding_mode pad  --app "--predict_xstart True --training_mode e2e  --vocab_size 11043  --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64```


-------------------
## Decode Diffusion-LM:
mkdir generation_outputs 

``python scripts/batch_decode.py {path-to-diffusion-lm} -1.0 ema``


------------------- 
## Controllable Text Generation 
First, train the classsifier used to guide the generation (e.g. a syntactic parser) 

``  
python train_run.py --experiment e2e-tgt-tree  --app "--init_emb {path-to-diffusion-lm} --n_embd {16} --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10
``

Then, we can use the trained classifier to guide generation. 
(currently, need to update the classifier directory in scripts/infill.py. I will clean this up in the next release.)

``python 
python scripts/infill.py --model_path {path-to-diffusion-lm} --eval_task_ 'control_tree' --use_ddim True  --notes "tree_adagrad" --eta 1. --verbose pipe``



-----------------------------------------------------

For details of the methods and results, please refer to our paper. 


```bibtex
@article{Li-2022-DiffusionLM,
  title={Diffusion-LM Improves Controllable Text Generation},
  author={Xiang Lisa Li and John Thickstun and Ishaan Gulrajani and Percy Liang and Tatsunori Hashimoto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14217}
}
```
