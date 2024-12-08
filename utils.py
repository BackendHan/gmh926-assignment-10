# utils.py
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from open_clip import create_model_and_transforms, tokenizer
from PIL import Image
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model, preprocess_train, preprocess_val = create_model_and_transforms("ViT-B/32", pretrained="openai")
    model = model.to(device)
    model.eval()
    return model, preprocess_val

def load_data(emb_path, img_folder):
    # emb_path: "image_embeddings.pickle"
    # img_folder: the folder containing coco_images_resized
    df = pd.read_pickle(emb_path)
    embeddings = np.stack(df['embedding'].values)  # [N, D]
    file_names = df['file_name'].values
    return df, embeddings, file_names

def normalize_features(features):
    # 对特征向量进行归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms

def get_pca_transform(embeddings, k=100):
    pca = PCA(n_components=k)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced

def encode_text(model, texts):
    import open_clip
    text_tokenizer = open_clip.get_tokenizer('ViT-B-32')  # 使用open_clip.get_tokenizer获取分词器
    text_tokens = text_tokenizer(texts)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb = F.normalize(text_emb, p=2, dim=1)
    return text_emb.cpu().numpy()


def encode_image(model, preprocess, img):
    # img: PIL Image
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(img_tensor)
        image_emb = F.normalize(image_emb, p=2, dim=1)
    return image_emb.cpu().numpy()

def compute_similarity(query_vector, database_vectors):
    # query_vector: [1, D]
    # database_vectors: [N, D]
    # 假设query和database_vectors已经归一化
    # 余弦相似度 = Q · E
    scores = query_vector @ database_vectors.T
    return scores[0]

def get_topk_images(scores, file_names, k=5):
    idxs = np.argsort(scores)[::-1][:k]
    result = [(file_names[i], scores[i]) for i in idxs]
    return result
