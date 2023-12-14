import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import albumentations as A
from rdkit import Chem

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch
from Bio.PDB import *
import numpy as np


class CFG:
    debug = False
    image_path = '/scratch/users/jihyeonj/PDBBind_processed/'
    captions_path ='/scratch/users/jihyeonj/PDBBind_processed/'
    batch_size = 32
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
    
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def load_coords_from_pdb(
    pdb,
    atoms,
    method="raw",
    also_bfactors=False,
    normalize_bfactors=True,
):
    """
    Returns array of shape (1, n_res, len(atoms), 3)
    """

    coords = []
    bfactors = []
    if method == "raw":  # Raw numpy implementation, faster than biopdb
        # Indexing into PDB format, allowing XXXX.XXX
        coords_in_pdb = [slice(30, 38), slice(38, 46), slice(46, 54)]
        # Indexing into PDB format, allowing XXX.XX
        bfactor_in_pdb = slice(60, 66)

        with open(pdb, "r") as f:
            resi_prev = 1
            counter = 0
            for l in f:
                l_split = l.rstrip("\n").split()
                if len(l_split) > 0 and l_split[0] == "ATOM" and l_split[2] in atoms:
                    resi = l_split[5]
                    if resi == resi_prev:
                        counter += 1
                    else:
                        counter = 0
                    if counter < len(atoms):
                        xyz = [
                            np.array(l[s].strip()).astype(float) for s in coords_in_pdb
                        ]
                        coords.append(xyz)
                        if also_bfactors:
                            bfactor = np.array(l[bfactor_in_pdb].strip()).astype(float)
                            bfactors.append(bfactor)
                    resi_prev = resi
            coords = torch.Tensor(np.array(coords)).view(1, -1, len(atoms), 3)

    return coords


def get_dmap(pdb, atoms, batched=True, out="torch", device=None):
    """
        Returns a n-dim array of shape (bs, 1, n, n)

    """
    coords = load_coords_from_pdb(pdb, atoms=atoms).view(1, -1, 3)
    coords = coords.contiguous()
    dmaps = torch.cdist(coords, coords).unsqueeze(1)
    return dmaps.detach().cpu().numpy()[0][0]


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        bkbone_atms = [["N"], ["CA"], ["C"]]
        
        all = []
        for atm in bkbone_atms:
            dmap = get_dmap(self.image_filenames[idx], atm)
            all.append(dmap)
        max_len= max([len(x) for x in all])
        output = [np.pad(x, (0,max_len-len(x)), 'constant') for x in all]
        
        
        img = np.moveaxis(output,0,-1)
        img = self.transforms(image=img)['image']
        item['image'] = torch.tensor(img).permute(2,0,1).float()
        item['caption'] = self.captions[idx]


        return item

    def __len__(self):
        return len(self.captions)

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
        
def make_train_valid_dfs(protpaths_s, ligtxts_s):
    max_id = len(protpaths_s)
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_prot = list(map(protpaths_s.__getitem__, train_ids))
    train_ligs = list(map(ligtxts_s.__getitem__, train_ids))
    valid_prot = list(map(protpaths_s.__getitem__, valid_ids))
    valid_ligs = list(map(ligtxts_s.__getitem__, valid_ids))
    return train_prot, train_ligs, valid_prot, valid_ligs


def build_loaders(prot, lig, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        prot,
        lig,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter
