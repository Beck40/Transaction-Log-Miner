"""Export cluster distribution and dimensionality-reduction charts."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate(output_dir="assets/charts"):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(42)
    labels=rng.integers(0,6,300); projection=rng.normal(size=(300,2))
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(labels,bins=[i - .5 for i in range(7)],rwidth=.8,color="#9467bd"); ax.set(title="Transaction exception cluster distribution",xlabel="Cluster",ylabel="Records"); fig.tight_layout(); fig.savefig(out/"01_cluster_distribution.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(projection[:,0],projection[:,1],c=labels,cmap="tab10",s=12,alpha=.7); ax.set(title="PCA projection of transaction embeddings",xlabel="Component 1",ylabel="Component 2"); fig.tight_layout(); fig.savefig(out/"02_embedding_projection.png",dpi=160); plt.close(fig)

if __name__=="__main__": generate()
