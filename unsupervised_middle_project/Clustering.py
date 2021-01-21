class Clustering:
    def __init__(self, labels, n_clusters, title, inertia=None, palette=None, fowlkes_mallows=None):
        self.labels = labels
        self.n_clusters = n_clusters
        self.title = title
        self.inertia = inertia
        self.palette = palette
        self.fowlkes_mallows = fowlkes_mallows

    def get_labels(self):
        return self.labels

    def get_n_clusters(self):
        return self.n_clusters

    def get_title(self) -> str:
        return self.title

    def get_inertia(self):
        return self.inertia

    def get_palette(self):
        return self.palette

    def get_fowlkes_mallows(self):
        return self.fowlkes_mallows
