import os
import itertools
from rdkit import Chem
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import torch
from Bio.PDB import *
import numpy as np
import os

from clip_utils import *
import wandb

wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="clip-test",
    # Track hyperparameters and run metadata
    config={
        "epochs": CFG.epochs,
    })

directory = '/scratch/users/jihyeonj/PDBBind_processed/'

ligpaths = []
protpaths = []

# iterate over files in
# that directory
for dir in os.listdir(directory):
    if dir !='.DS_Store':
        foldr = os.path.join(directory, dir)
    for i in os.listdir(foldr):
        if i.endswith('.sdf'):
            ligpaths.append(os.path.join(foldr, i))
        elif i.endswith('.pdb'):
            protpaths.append(os.path.join(foldr, i))
            
batch_size = 10
dim = 768
embeddings = torch.randn(batch_size, dim)
out = embeddings @ embeddings.T
#print(F.softmax(out, dim=-1))


ligtxts = []
for p in ligpaths:
    suppl = Chem.SDMolSupplier(p, sanitize=False)
    smi = Chem.MolToSmiles(suppl[0])

    ligtxts.append(smi)
    
protpaths_s = protpaths
ligtxts_s = ligtxts


def main():
    
    train_prot, train_ligs, valid_prot, valid_ligs = make_train_valid_dfs(protpaths_s, ligtxts_s)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_prot, train_ligs, tokenizer, mode="train")
    valid_loader = build_loaders(valid_prot, valid_ligs, tokenizer, mode="valid")

    train_f = open('train_loader.pkl','wb')
    valid_f = open('valid_loader.pkl', 'wb')
    pickle.dump(train_loader, train_f)
    pickle.dump(valid_loader, valid_f)

    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
	wandb.log({"loss": train_loss})
	with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
        
        

if __name__ == "__main__":
    main()
