import networkx as nx
from node2vec import Node2Vec
import numpy as np
import polars as pl
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CAVExtractor:
    """
    Pillar 3: Geographic Aura Vector (CAV).
    Captures spatial context via Node2Vec on a world graph.
    """
    def __init__(self, dimensions: int = 8, walk_length: int = 30, num_walks: int = 200):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embeddings = {} # country -> vector

    def build_world_graph(self, 
                          adjacency_df: Optional[pl.DataFrame] = None, 
                          flight_df: Optional[pl.DataFrame] = None,
                          trade_df: Optional[pl.DataFrame] = None) -> nx.Graph:
        """
        Build a weighted graph of countries.
        """
        G = nx.Graph()
        
        # 1. Adjacency (Static)
        if adjacency_df is not None:
            for row in adjacency_df.to_dicts():
                G.add_edge(row["c1"], row["c2"], weight=0.3)
        
        # 2. Flights
        if flight_df is not None:
            for row in flight_df.to_dicts():
                # weight = 0.5 * log(flights + 1)
                w = 0.5 * np.log1p(row["flights"])
                if G.has_edge(row["origin"], row["dest"]):
                    G[row["origin"]][row["dest"]]["weight"] += w
                else:
                    G.add_edge(row["origin"], row["dest"], weight=w)
        
        # If graph is empty, add some default nodes/edges for testing
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph. Adding placeholder nodes.")
            G.add_edge("USA", "CAN", weight=1.0)
            G.add_edge("USA", "GBR", weight=0.8)
            G.add_edge("TUR", "DEU", weight=0.7)
            G.add_edge("TUR", "SYR", weight=0.5)

        return G

    def train_node2vec(self, G: nx.Graph):
        """
        Train Node2Vec to get country embeddings.
        """
        logger.info(f"Training Node2Vec on world graph with {G.number_of_nodes()} nodes...")
        node2vec = Node2Vec(G, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks, workers=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        self.embeddings = {node: model.wv[node] for node in G.nodes()}
        logger.info("Node2Vec training complete.")

    def compute_aura(self, countries: List[str]) -> np.ndarray:
        """
        Compute the 8-dimensional aura vector for a plasmid based on countries where it's seen.
        """
        if not countries:
            return np.zeros(self.dimensions, dtype=np.float32)
            
        vecs = []
        for c in countries:
            if c in self.embeddings:
                vecs.append(self.embeddings[c])
        
        if not vecs:
            return np.zeros(self.dimensions, dtype=np.float32)
            
        return np.mean(vecs, axis=0).astype(np.float32)
