# graphMERT-python
A simple example test implementation of the princeton graphMERT paper. 

https://arxiv.org/abs/2510.09580

All rights w/ authors:
GraphMERT: Efficient and Scalable Distillation of Reliable
Knowledge Graphs from Unstructured Data
Margarita Belova, Jiaxin Xiao, Shikhar Tuli and Niraj K. Jha
from
Department of Electrical and Computer Engineering, 
Princeton University
ArXiv link arXiv:2510.09580 from 10 October 2025


# Based on the notebooks use of Romeo + Juliet text for the graphMERT model training.

## GraphMERT Node Embeddings (t-SNE View)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8,8))
plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='Spectral', alpha=0.7)
for i, text in enumerate(corpus[:50]):  # optional show partial labels
    plt.text(emb_2d[i,0]+0.01, emb_2d[i,1]+0.01, str(labels[i]), fontsize=6)
plt.title("GraphMERT Node Embeddings (t-SNE View)")
plt.show()

<img width="683" height="682" alt="image" src="https://github.com/user-attachments/assets/a8ae3cb5-a646-4c22-bff4-44fd21c555a2" />


     
## GraphMERT Semantic Graph Visualization

import networkx as nx

G = nx.Graph()
for i, emb in enumerate(embeddings):
    G.add_node(i, text=corpus[i], cluster=int(labels[i]))
 
from scipy.spatial.distance import pdist, squareform
dist = squareform(pdist(embeddings))
threshold = np.percentile(dist, 5)
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        if dist[i,j] < threshold:
            G.add_edge(i, j)

plt.figure(figsize=(10,8))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=labels, cmap='Spectral', with_labels=False, node_size=40)
plt.title("GraphMERT Semantic Graph Visualization")
plt.show()

<img width="1019" height="842" alt="image" src="https://github.com/user-attachments/assets/0ff4cd7b-42b0-40ca-aedf-9ef2dfbaa3c5" />


## Query search on the graphs results
### This is what we want bacause the search of the graph is linear based on chained knowledge with nodes having data of them.
<img width="1220" height="1042" alt="image" src="https://github.com/user-attachments/assets/3d0da53b-b962-423f-b3e3-118c749ba4b9" />



     

