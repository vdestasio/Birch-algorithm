# BIRCH
I implemented:
- Phase 1 with merging refinement (but no path thing to rebuild the tree)
- NO phase 2 because they didn't say anywhere how to choose the lower threshold
- Phase 3 hierarchical clustering/kmeans on the centroids of the leaves
- Phase 4 (assignment of all points to closest centroid)

I have to choose:
- Do I rebuild the tree when it exceeds a certain memory size? Or do I choose manually threshold and branching factor as in the sklearn implmentation?
- Do I manage the outliers? How close to the original implementation?
- If I apply hierarchical clustering, how do I assure that the clustering features are not too many and that I can actually apply the global clustering algorithm? should I impose a max number of points at the end of phase 1?

# Dataset
https://cs.joensuu.fi/sipu/datasets/