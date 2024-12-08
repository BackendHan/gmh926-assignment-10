# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import torch
from utils import load_model, load_data, normalize_features, get_pca_transform, encode_text, encode_image, compute_similarity, get_topk_images

# 配置路径（请根据实际情况修改）
EMBEDDING_PATH = "image_embeddings.pickle"  # 已有的embeddings文件
IMAGE_FOLDER = "coco_images_resized"        # 解压后图片所在文件夹

# 加载模型与数据
model, preprocess_val = load_model()
df, embeddings, file_names = load_data(EMBEDDING_PATH, IMAGE_FOLDER)

# 假设原始embeddings已经归一化了，如果没有请归一化
embeddings = normalize_features(embeddings)

# 准备PCA特征
pca_k = 100
pca, reduced_embeddings = get_pca_transform(embeddings, k=pca_k)
reduced_embeddings = normalize_features(reduced_embeddings)  # PCA后再归一化

st.title("Simplified Google Image Search")

text_query = st.text_input("Text Query", "")
uploaded_file = st.file_uploader("Upload an image query", type=["jpg","jpeg","png"])
lambda_val = st.slider("Weight of text vs image query (lambda)", 0.0, 1.0, 0.5)

feature_type = st.selectbox("Feature type", ["CLIP", "PCA"])

if st.button("Search"):
    text_query_vec = None
    image_query_vec = None

    # 文本查询
    if text_query.strip() != "":
        text_query_vec = encode_text(model, [text_query])

    # 图像查询
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        image_query_vec = encode_image(model, preprocess_val, img)

    # 合成最终query向量
    if text_query_vec is not None and image_query_vec is not None:
        combined = lambda_val * text_query_vec + (1 - lambda_val) * image_query_vec
        # 归一化
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        query_vector = combined
    elif text_query_vec is not None:
        query_vector = text_query_vec
    elif image_query_vec is not None:
        query_vector = image_query_vec
    else:
        st.write("Please provide at least one query (text or image).")
        st.stop()

    # 根据选择的特征类型处理
    if feature_type == "PCA":
        query_vector_pca = pca.transform(query_vector)
        query_vector_pca = query_vector_pca / np.linalg.norm(query_vector_pca, axis=1, keepdims=True)
        scores = compute_similarity(query_vector_pca, reduced_embeddings)
    else:
        # 使用原始特征
        scores = compute_similarity(query_vector, embeddings)

    results = get_topk_images(scores, file_names, k=5)

    # 显示结果
    for fname, score in results:
        st.write(f"Image: {fname}, Score: {score:.4f}")
        img_path = os.path.join(IMAGE_FOLDER, fname)
        st.image(img_path, width=200)
