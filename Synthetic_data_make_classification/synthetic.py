import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from matplotlib import rcParams

X,y=make_classification(n_samples=1000,n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
type(X)
df=pd.concat([pd.DataFrame(X),pd.Series(y)], axis=1)
df.columns=['x1', 'x2', 'y']
df.sample(5)

def plot(df:pd.DataFrame, x1:str, x2:str, y:str, title:str='', save:bool=False, figname='figure.png', figblock=False):
    plt.figure(figsize=(10,5))
    plt.scatter(x=df[df[y]==0][x1], y=df[df[y]==0][x2], label='y=0')
    plt.scatter(x=df[df[y]==1][x1], y=df[df[y]==1][x2], label='y=1')
    plt.title(title, fontsize=15)
    plt.legend()
    if save:
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show(block=figblock)


plot(df=df, x1='x1', x2='x2', y='y', title='Dataset with 2 classes', save=True)

#making noise
X,y=make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.15, random_state=42)
df=pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
df.columns=['x1', 'x2', 'y']
plot(df=df, x1='x1', x2='x2', y='y', title='Dataset with 2 classes with 15% noise', save=True, figblock=False)

#class imbalance
X,y=make_classification(n_samples=1000,n_features=2,n_redundant=0, n_clusters_per_class=1, weights=[0.95],random_state=42)
df=pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
df.columns=['x1','x2','y']
plot(df=df, x1='x1', x2='x2', y='y', title='Dataset with 2 classes with weight [0.95]', save=True, figblock=False)

#class separation ratio
X,y=make_classification(n_samples=1000,n_features=2,n_redundant=0, n_clusters_per_class=1, class_sep=5,random_state=42)
df=pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
df.columns=['x1','x2','y']
plot(df=df, x1='x1', x2='x2', y='y', title='Dataset with 2 classes with class separation 5', save=True, figblock=False)

#class separation ratio
X,y=make_classification(n_samples=1000,n_features=2,n_redundant=0, n_clusters_per_class=1, class_sep=20,random_state=42)
df=pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
df.columns=['x1','x2','y']
plot(df=df, x1='x1', x2='x2', y='y', title='Dataset with 2 classes with class separation 20', save=True, figblock=True)
