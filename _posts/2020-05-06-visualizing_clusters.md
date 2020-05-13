---
layout: post
title:  "Visualising Clusters in a Low Dimensional Space"
date:   2020-05-12
categories: clustering 
author: William D'Souza
---


![header_image]({{ site.url }}/images/tsne_header.png)


Unsupervised learning techniques are incredibly powerful if implemented properly. Many people are afraid to use these techniques because it is hard for them to validate the model results in their own head. There is always a known label (or answer) when training supervised models, so it is much easier to quantify how well a supervised model performs from inspecting various measures. The uncertainty of unsupervised learning can leave you with discomfort because there is no real way to quantify the model's results, other than extracting insights from it and “making sense” with what we see.

To endeavour into unsupervised techniques, there needs to be a trust in the math behind it. After that, it can be a little bit easier to cope with. In regards to clustering, there is another practice that you can do to help you gain more confidence in your results. 

In most real situations, we will usually have a large number of features for our model. Unfortunately, our simple brains cannot comprehend what anything will look like if it surpasses 3 dimensions! Visualising your clusters in a low dimensional space will help you see how defined the clusters are, the space between them, and if choosing a clustering method is the right way to attack your problem.

## Libraries and Preprocessing

In general, it is important to pick out the features that you want to cluster on for your end analysis. If they are not fully present, then you may need to engineer them. All the following code is an example of NHL player data from 2004-2018 with some feature engineering to create a new dataset. **It is important to note that this way of wrangling your data will most likely not be applicable for you. Use methods that will get your data to the state you need. Data wrangling is a creative stage that everyone may do a little bit differently.**

``` 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

nhl = pd.read_csv('nhl.csv', encoding= 'latin-1')

## Features for Clusters
clustering_columns = ['player',
    'pim_gp',
    'even_g_gp',
    'pp_g_gp',
    'sh_g_gp', 
    'gw_g_gp',
    'even_a_gp',
    'pp_a_gp',
    'sh_a_gp',
    'sh_perc',
    'avg_toi',
    'avg_blocks',
    'avg_hits',
    'fow']

#summing up stats to be cumulative
grouped = nhl_cleaned.groupby('Player').sum().reset_index()

# Run through each player, calculate per game stats, and append to df
cleaned_df = pd.DataFrame(columns = clustering_columns) 
player_list = list(set(grouped['Player']))
for player in player_list:
    player_df = grouped.loc[grouped['Player'] == player].reset_index()
    player = player_df['Player'][0]
    games_played = player_df['GP'][0]
    pim_gp = float(player_df['PIM'][0] / games_played) 
    even_g_gp = float(player_df['even_goals'][0] / games_played)
    pp_g_gp = float(player_df['pp_goals'][0] / games_played)
    sh_g_gp = float(player_df['sh_goals'][0] / games_played)
    gw_g_gp = float(player_df['gw_goals'][0] / games_played)
    total_g = player_df['even_goals'][0] + player_df['pp_goals'][0] + player_df['sh_goals'][0]  + player_df['gw_goals'][0]
    if player_df['shots'][0] == 0:
        sh_perc = 0
    else:
        sh_perc = float(total_g / player_df['shots'][0])

    even_a_gp = float(player_df['even_assists'][0] / games_played)
    pp_a_gp = float(player_df['pp_assists'][0] / games_played) 
    sh_a_gp = float(player_df['sh_assists'][0] / games_played)
    avg_toi = float(player_df['TOI'][0] / games_played)
    avg_pp_a_gpblocks = float(player_df['BLK'][0] / games_played) 
    avg_hits = float(player_df['HIT'][0] / games_played)
    total_fo = player_df['FOW'][0] + player_df['FOL'][0]
    if total_fo == 0:
        fow_perc = float(0)
    else:
        fow_perc = float(player_df['FOW'][0] / total_fo)

    cleaned_df = cleaned_df.append({
    'player' : player,
    'pim_gp': pim_gp,
    'even_g_gp' : even_g_gp,
    'pp_g_gp' : pp_g_gp,
    'sh_g_gp' : sh_g_gp, 
    'gw_g_gp' : gw_g_gp,
    'even_a_gp' : even_a_gp,
    'pp_a_gp' : pp_a_gp,
    'sh_a_gp' : sh_a_gp,
    'sh_perc' : sh_perc,
    'avg_toi' : avg_toi,
    'avg_blocks': avg_blocks,
    'avg_hits': avg_hits,
    'fow': fow_perc 
    }, ignore_indexut some level of preprocessing may need to be done  = True) 
```

The resulting data frame looks like this:

![useful image]({{ site.url }}/images/cluster_df.png)

To give you an understanding of what each column represents, here is a brief explanation for each of the variables.

1. **player**: NHL Player
2. **pim_gp**: Penalty Minutes per Game Played
3. **even_g_gp**: Even Goals per Game Played
4. **pp_g_gp**: Powerplay Goals per Game Played
5. **sh_g_gp**: Shorthanded Goals per Game Played
6. **gw_g_gp**: Game Winning Goals per Game Played
7. **even_a_gp**: Even Assists per Game Played
8. **pp_a_gp**: Powerplay Assists per Game Played
9. **sh_a_gp**: Shorthanded Assists per Game Played
10. **sh_perc**: Shooting %
11. **avg_toi**: Average Time on Ice
12. **avg_blocks**: Blocks per Game Played
13. **avg_hits**: Hits per Game Played
14. **fow**: Faceoff Win %

## Scaling the Data, Choosing the Method & Number of Clusters

It is important to scale your data. The results you yield if you don't will be extremely misleading. The general reason to scale is that not all measurements are the same and it is inappropriate to use raw values for situations where you use different measures in one application. An example of this is using weight and height, one measurement is in kilograms/pounds and the other in feet/meters. According to my application, I normalized the data using a Min-Max method. 

I also decided to use hierarchical clustering over k-means. Hierarchical clustering is extremely powerful and does not necessarily need you to state the number of clusters. It even produces beautiful dendrograms that can help you interpret your clusters

Depending on your application, you may look at a *Standard Scaler* or even a *Robust Scaler*. The number of clusters was evaluated using the silhouette score. The optimal number for the silhouette score is typically the one with the highest coefficient. Although the "optimal" number was not the final one chosen in this application, **It is important to note that the number of clusters you choose can also be driven by practical or objective needs** 

```
# Scaling Data
ss = MinMaxScaler()
transformed = ss.fit_transform(cleaned_df)

for i in range (2,10):
    results = AgglomerativeClustering(n_clusters = i, linkage = 'average').fit(transformed)
    silhouette_avg = silhouette_score(transformed, results.labels_)
    print("For n_clusters =", i, "The average silhouette_score is :", silhouette_avg)

For n_clusters = 2 The average silhouette_score is : 0.6306192030168505
For n_clusters = 3 The average silhouette_score is : 0.5275206624683815
For n_clusters = 4 The average silhouette_score is : 0.5253108653327441
For n_clusters = 5 The average silhouette_score is : 0.5105659565494436
For n_clusters = 6 The average silhouette_score is : 0.4779116479699615
For n_clusters = 7 The average silhouette_score is : 0.46592731727826187
For n_clusters = 8 The average silhouette_score is : 0.42161463250245973
For n_clusters = 9 The average silhouette_score is : 0.3541173097027004
```

## Visualising the Clusters

The code below creates the following visuals that project the clusters in a 2-D and 3-D space. We use dimensionality reduction to *compress* the data into a lower number of features. T-SNE is a powerful tool to visualise the clusters, but is extremely sensitive and needs to be tuned using iterative methods. PCA is a popular method and is extremely powerful. Visualising your clusters will give you that extra confidence in your end goal because it will help you understand if clustering your data will yield purposeful results for your objective!

```
# Performing t-sne on data
time_start = time.time()       #init time 
tsne = TSNE(n_components=2, verbose=1, perplexity= 70, n_iter=700, random_state = 2) ## Chose params through iterative testing
tsne_results = tsne.fit_transform(df_scaled)
print('Time elapsed: {} seconds'.format(time.time()-time_start))
df_scaled.head()

# Performing PCA on data
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_scaled)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# Merging t-sne and pca results in sclaed dataframe
df_scaled['tsne-2d-one'] = tsne_results[:,0]
df_scaled['tsne-2d-two'] = tsne_results[:,1]
df_scaled['pca-one'] = pca_result[:,0]
df_scaled['pca-two'] = pca_result[:,1] 
df_scaled['pca-three'] = pca_result[:,2]
df_scaled['clusters'] = clustering.labels_

## Visualising clusters in a 2-D Space using tsne
plt.figure(figsize=(10,5))
sns.scatterplot(
x = df_scaled["tsne-2d-one"], 
y = df_scaled["tsne-2d-two"],
hue = df_scaled['clusters'],
palette=sns.color_palette("hls",  4),
legend="full",
alpha=0.3)
plt.show()


## Visualising Clusters in a 2-D & 3-D Space using PCA
plt.figure(figsize=(16,7))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue = 'clusters',
    data=df_scaled,
    palette=sns.color_palette("hls",  4),
    legend="full",
    alpha=0.3)


ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df_scaled["pca-one"], 
    ys=df_scaled["pca-two"], 
    zs=df_scaled["pca-three"], 
    c=df_scaled['clusters'], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
```
![useful image]({{ site.url }}/images/tsne.png)
![useful image]({{ site.url }}/images/pca_1.png)
![useful image]({{ site.url }}/images/pca_2.png)


**Using clustering for this dataset is up for discussion. As well, there are possibly many things that can be done differently from selecting features to choice of the clustering method. Hopefully, this is just an informative tutorial for you to realise the importance of visualising your clusters!**
