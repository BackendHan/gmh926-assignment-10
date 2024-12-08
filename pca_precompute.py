import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib

# 加载embeddings
df = pd.read_pickle('image_embeddings.pickle')
embeddings = np.stack(df['embedding'].values)

# 执行PCA降维，比如降到100维
k = 100
pca = PCA(n_components=k)
reduced_embeddings = pca.fit_transform(embeddings)

# 保存pca模型和降维后的特征
joblib.dump(pca, 'pca_model.joblib')
np.save('reduced_embeddings.npy', reduced_embeddings)
