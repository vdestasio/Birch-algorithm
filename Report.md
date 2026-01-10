# BIRCH
I implemented:
- Phase 1 with merging refinement (but no path thing to rebuild the tree and no outlier or delay option)
- NO phase 2 because they didn't say anywhere how to choose the lower threshold
- Phase 3: kmeans on the centroids of the leaves, but weighted to keep into account the number of points!!
- Phase 4 (assignment of all points to closest centroid)

I manage the outliers ONLY in phase 4! (and it's optional)

I have to choose:
- Do I rebuild the tree when it exceeds a certain memory size? Or do I choose manually threshold and branching factor as in the sklearn implmentation? PROBABLY CHOOSE MANUALLY
- If I apply hierarchical clustering, how do I assure that the clustering features are not too many and that I can actually apply the global clustering algorithm? should I impose a max number of points at the end of phase 1?

# Presentation structure
- Exaplain what is a clustering feature and what is a CF-tree
- Explain the phases of the algorithm
- Explain the parameters of the algorithm and what we can substitute
- Explain the choices in my implementation
- Comparison with other algorithms (k means and dbscan, agglomerative clustering):
    - On a lot of points, what can we use and what we can't use
    - If there is noise, what can we use and what we can't use
    - Possibly show how changing the parameters changes solution (ex. higher threshold -> smaller tree)
DO not compare time....

# Dataset
https://cs.joensuu.fi/sipu/datasets/