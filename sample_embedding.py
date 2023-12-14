from clip_utils import *

def get_image_embeddings(val_img, val_txt, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(val_img, val_txt, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=1):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    bkbone_atms = [['CA'], ['C'], ['N']]
    full_dist = []
    for a in bkbone_atms:
        full_dist.append(get_dmap(matches[0], a))
    plt.imshow(np.asarray(full_dist)[0,:,:])
    
    
    """
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        print(match)
        bkbone_atms = [['CA'], ['C'], ['N']]
        full_dist = []
        for a in bkbone_atms:
            full_dist.append(get_dmap(match, a))
        fin_img = np.moveaxis(np.array(full_dist),0,-1)

        ax.imshow(fin_img)
        ax.axis("off")
    """
    plt.show()





_, _, val_prot, val_lig = make_train_valid_dfs()

model, image_embeddings = get_image_embeddings(val_prot, val_lig, "best.pt")

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
encoded_query = tokenizer([query])
batch = {
    key: torch.tensor(values).to(CFG.device)
    for key, values in encoded_query.items()
}
text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
text_embeddings = model.text_projection(text_features)

find_matches(model, 
             image_embeddings,
             query=ligtxts[100],
             image_filenames=val_prot,
             n=1)