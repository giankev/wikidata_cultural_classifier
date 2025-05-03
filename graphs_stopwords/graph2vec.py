# preprocess graphs before training
# extract features
# embedd them
# save on specified file
# save embedded vectors inside specified folder

import os
import numpy as np
import pandas as pd
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from karateclub import Graph2Vec as _KCGraph2Vec

from wiki_extractor import WikidataExtractor
from text2graph import Text2Graph
from dataset_parser import DatasetParser

# Paths
INPUT_CSV = os.path.expanduser('~/content/valid.csv')
OUTPUT_CSV = os.path.expanduser('~/content/valid_g2v.csv')
GRAPH2VEC_FOLDER = os.path.expanduser('~/content/graph2vec_out/')

# Wikidata properties to include in graph
PROPS = {
    "P495": "Country of Origin",
    "P17": "Country",
    "P131": "Located in Adm. Entity",
    "P361": "Part Of",
    "P31": "Instance Of",
    "P279": "Subclass Of",
    "P443": "Pronunciation audio",
    "P1435": "Heritage designation",
    "P27": "Country of citizenship",
    "P1705": "Native Label",
    "P2012": "Cuisine"
}

MAX_WORKERS = 8
os.makedirs(GRAPH2VEC_FOLDER, exist_ok=True)

class Graph2Vec:
    """
    Wrapper around KarateClub's Graph2Vec for whole-graph embeddings.
    """
    def __init__(self, dimensions: int = 128, wl_iterations: int = 2, workers: int = 1):
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.workers = workers

    def embed(self, graph_input) -> np.ndarray:
        # Convert Text2Graph to NetworkX if needed
        if isinstance(graph_input, Text2Graph):
            G = self._to_networkx(graph_input)
        else:
            G = graph_input
        # Re-label nodes to integer indices (0..n-1) for KarateClub compatibility
        G = nx.convert_node_labels_to_integers(G)
        # Instantiate model
        model = _KCGraph2Vec(
            dimensions=self.dimensions,
            wl_iterations=self.wl_iterations,
            workers=self.workers
        )
        # Fit and handle potential vocabulary errors
        try:
            model.fit([G])
            emb = model.get_embedding()[0]
        except Exception as e:
            print(f"[Graph2Vec WARNING] {e}")
            emb = np.zeros(self.dimensions, dtype=float)
        return emb

    def _to_networkx(self, t2g: Text2Graph) -> nx.Graph:
        G = nx.Graph()
        for n in t2g.nodes:
            G.add_node(n)
        for (u, v), w in t2g.edge_freq.items():
            if u != v:
                G.add_edge(u, v, weight=w)
        return G


def process_row(row):
    item_url = row['item']
    metadata = {k: row.get(k, '') for k in ['name','description','type','category','subcategory','label']}
    try:
        # Build Text2Graph
        wd_item = WikidataExtractor(item_url)
        t2g = Text2Graph(wd_item, language='en', properties_to_check=PROPS)
        t2g.fetch_sections()
        t2g.build_graph()
        t2g.collapse_linear_paths()

        # Extract graph features
        feats = t2g.get_all_features_as_dict()
        parser = DatasetParser(wd_item)
        feats['number_sitelinks'] = parser.get_number_sitelinks()
        feats['sitelinks_translation_entropy'] = parser.sitelinks_translation_entropy()

        # Compute Graph2Vec embedding
        g2v = Graph2Vec(dimensions=128, wl_iterations=2, workers=4)
        vec = g2v.embed(t2g)
        # Signal validity if vector is not all zeros
        g2v_valid = not np.all(vec == 0)
        feats['g2v_valid'] = g2v_valid

        # Save embedding file
        qid = wd_item.get_entity_id()
        np.save(os.path.join(GRAPH2VEC_FOLDER, f"{qid}.npy"), vec)

        # Combine metadata
        feats.update(metadata)
        feats['item'] = item_url
        feats['qid'] = qid
        feats['language'] = 'en'
        return feats
    except Exception as e:
        print(f"[ERROR] {item_url}: {e}")
        return None

if __name__ == '__main__':
    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} items with {MAX_WORKERS} threads...")

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            print(f"\rComplete: {completed}/{len(df)}", end='', flush=True)
            if res:
                results.append(res)

    # Save to CSV (including g2v_valid flag)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote features to {OUTPUT_CSV}")
