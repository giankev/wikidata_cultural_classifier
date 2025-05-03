# build graphs based on inline links

import os
import requests
import re
from bs4 import BeautifulSoup
from statistics import mean, median, pstdev
import math
import networkx as nx
from pyvis.network import Network
from wiki_extractor import WikidataExtractor
from sklearn.cluster import KMeans
import time
from requests.utils import requote_uri

# features and embedding
from statistics import mean, median, pstdev
import networkx as nx
from karateclub import Graph2Vec
import numpy as np
from networkx.algorithms import community

GRAPH_WEB_DIR = os.path.expanduser("~/content/web_graphs")

class GraphBuilder:
    def __init__(self, output_dir=GRAPH_WEB_DIR):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_sitelinks_count(self, wiki_title: str) -> int:
        """
        Fetch number of sitelinks for a Wikidata item by its English Wikipedia title,
        retrying on HTTP429 with exponential backoff.
        """
        params = {
            'action': 'wbgetentities',
            'sites': 'enwiki',
            'titles': wiki_title,
            'props': 'sitelinks',
            'format': 'json'
        }
        backoff = 1
        for attempt in range(1, 4):
            resp = requests.get('https://www.wikidata.org/w/api.php', params=params)
            if resp.status_code == 429:
                print(f"[WARN] Rate limit hit for '{wiki_title}', retry {attempt}/3 after {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            entities = resp.json().get('entities', {})
            if not entities:
                return 0
            ent = next(iter(entities.values()))
            sl = ent.get('sitelinks', {})
            return sum(1 for k in sl.keys() if k.endswith('wiki'))
        # se dopo 3 tentativi ancora 429 (o errori transient), restituisci 0
        print(f"[ERROR] Unable to fetch sitelinks for '{wiki_title}' after 3 retries, defaulting to 0")
        return 0


    def parse_sections(self, html: str):
        """
        Split HTML into sections: intro and each h2/h3.
        Returns list of (section_title, list_of_paragraph_tags)
        """
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('div', id='bodyContent')
        if not content:
            return []
        sections = []
        # Gather paragraphs and headers in order
        elements = content.find_all(['p', 'h2', 'h3'])
        current_title = 'Introduction'
        current_pars = []
        for el in elements:
            if el.name in ('h2', 'h3'):
                sections.append((current_title, current_pars))
                headline = el.find(class_='mw-headline')
                current_title = headline.get_text() if headline else el.get_text()
                current_pars = []
            else:  # paragraph
                current_pars.append(el)
        # append last section
        sections.append((current_title, current_pars))
        return sections

    def build_graph(self, identifier: str):
        """
        Build hierarchical graph for a Wikidata QID
        """
        # ---===== fetch english wikipedia page =====---
        wd = WikidataExtractor(identifier)
        slinks = wd.get_sitelinks()
        enwiki = slinks.get('enwiki')
        if not enwiki:
            raise ValueError(f"No English Wikipedia page for {identifier}")
        url = enwiki['url'].replace("?", "%3F") # ? char in links is encoded as %3F for some reason
        title = enwiki['title']

        central_weight = self.fetch_sitelinks_count(title)

        resp = requests.get(url)
        resp.raise_for_status()
        html = resp.text

        # ---===== graph init =====---
        self.nodes = {title: {'weight': central_weight, 'label': title}}
        self.edge_freq = {}
        sitelinks_counts = []
        page_char_count = 0

        # ---===== build graph =====---
        sections = self.parse_sections(html)
        for sect_title, pars in sections:
            # Add section node
            sect_node = f"{title}::{sect_title}"
            self.nodes[sect_node] = {'weight': 1, 'label': sect_title}
            self._add_edge(title, sect_node)
            for p in pars:
                text = p.get_text()
                page_char_count += len(text)

                # extract linked items
                links = []
                for a in p.find_all('a', href=True, title=True):
                    href = a['href']
                    if href.startswith('/wiki/') and ':' not in href:
                        wiki_title = requests.utils.unquote(href.split('/wiki/')[-1])

                        # account for redirects
                        full_url = requote_uri("https://en.wikipedia.org" + href)
                        resp = requests.get(full_url, allow_redirects=True, timeout=5)
                        resp.raise_for_status()
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        can = soup.find('link', rel='canonical')
                        if can and can.has_attr('href'):
                            # extract only title
                            wiki_title = requests.utils.unquote(can['href'].split('/wiki/')[-1])
                            wiki_title = wiki_title.partition("#")[0] # remove url comments for wikidata query
                        else:
                            # fallback all’href original
                            wiki_title = requests.utils.unquote(href.split('/wiki/')[-1])

                        alias = a.get_text()

                        if wiki_title not in self.nodes:
                            try:
                                w = self.fetch_sitelinks_count(wiki_title)
                                if w == 0:
                                    print(wiki_title)
                            except requests.HTTPError as e:
                                print(f"[WARN] Failed to fetch sitelinks for '{wiki_title}': {e}")
                                w = 0
                            self.nodes[wiki_title] = {'weight': w, 'label': alias}
                            sitelinks_counts.append(w)
                            time.sleep(0.1)
                        links.append((wiki_title, alias))
                        self._add_edge(sect_node, wiki_title)

                # sentence-level connections
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                for sent in sentences:
                    present = [wiki for wiki, alias in links if alias in sent]
                    for i in range(len(present)):
                        for j in range(i + 1, len(present)):
                            u, v = present[i], present[j]
                            self._add_edge(u, v)
                            self._add_edge(v, u)

        # ---===== compute enwiki stats =====---
        stats = {
            'page_char_count': page_char_count,
            'mean_sitelinks': mean(sitelinks_counts) if sitelinks_counts else 0,
            'median_sitelinks': median(sitelinks_counts) if sitelinks_counts else 0,
            'std_sitelinks': pstdev(sitelinks_counts) if len(sitelinks_counts) > 1 else 0,
        }

        self.features = [stats['page_char_count'], stats['mean_sitelinks'], stats['median_sitelinks'], stats['std_sitelinks']]
        return title, stats

    def _add_edge(self, u, v):
        key = (u, v)
        self.edge_freq[key] = self.edge_freq.get(key, 0) + 1


    def graph_to_web_print(self, title: str, output_file="graph.html"):
        G = nx.DiGraph()
        color_map = {
            'root': '#FF5733',        # red-orange
            'section': '#33A1FF',     # blue
            'item': '#99A1FF',        # green
        }

        # Add nodes with attributes
        for node, data in self.nodes.items():
            node_type = data.get('type', 'item')
            weight = data.get('weight', 1)
            G.add_node(
                node,
                title=f"{node} (weight: {weight})",
                label=f"{node}\n({weight})",   # label shown inside node
                size=10,
                color=color_map.get(node_type, "#DDDDDD")
            )

        # Add edges
        for (u, v), w in self.edge_freq.items():
            G.add_edge(u, v, value=w)

        net = Network(directed=True, height="900px", width="100%", notebook=False)
        net.from_nx(G)

        net.repulsion(node_distance=120, central_gravity=0.3)  # better layout

        out_path = os.path.join(self.output_dir, output_file)
        net.write_html(out_path)
        print(f"Interactive graph written to {out_path}")


# -----===== FEATURES =====-----
def extract_graph_features(G: nx.DiGraph) -> dict:
    feats = {}
    feats['num_nodes'] = G.number_of_nodes()
    feats['num_edges'] = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    feats['average_degree'] = mean(degs) if degs else 0
    feats['density'] = nx.density(G)
    feats['average_clustering'] = nx.average_clustering(G.to_undirected())
    try:
        feats['assortativity'] = nx.degree_pearson_correlation_coefficient(G)
    except:
        feats['assortativity'] = 0
    feats['transitivity'] = nx.transitivity(G.to_undirected())
    try:
        parts = community.greedy_modularity_communities(G.to_undirected())
        feats['modularity'] = community.modularity(G.to_undirected(), parts)
    except:
        feats['modularity'] = 0
    weights = [data.get('value',1) for _,_,data in G.edges(data=True)]
    feats['edge_weight_mean'] = mean(weights) if weights else 0
    feats['edge_weight_median'] = median(weights) if weights else 0
    feats['edge_weight_std'] = pstdev(weights) if len(weights)>1 else 0
    in_deg = [d for _,d in G.in_degree()]
    out_deg = [d for _,d in G.out_degree()]
    feats['in_out_ratio'] = (mean(in_deg)/mean(out_deg)) if out_deg and mean(out_deg)!=0 else 0
    return feats

def compute_graph2vec_embedding(G: nx.DiGraph, dimensions: int=64, epochs: int=50) -> np.ndarray:
    """
    Restituisce embedding Graph2Vec su grafo con nodi rimappati in interi.
    Se il grafo ha meno di 2 nodi, restituisce un vettore zero.
    """
    G_ug = G.to_undirected()
    H = nx.convert_node_labels_to_integers(G_ug)
    # Se meno di 2 nodi, skip embedding
    if H.number_of_nodes() < 2:
        return np.zeros(dimensions)
    model = Graph2Vec(dimensions=dimensions, workers=1, epochs=epochs)
    try:
        model.fit([H])
    except AssertionError as e:
        print(f"[WARN] Graph2Vec error: {e}, returning zeros embedding.")
        return np.zeros(dimensions)
    emb = model.get_embedding()
    return emb[0]

def build_graph_features(qid: str) -> dict:
    builder = GraphBuilder()
    title, stats = builder.build_graph(qid)
    G = nx.DiGraph()
    for node,data in builder.nodes.items(): G.add_node(node, **data)
    for (u,v), w in builder.edge_freq.items(): G.add_edge(u,v, value=w)
    topo = extract_graph_features(G)
    emb = compute_graph2vec_embedding(G)
    result = {**stats, **topo, 'graph2vec_embedding': emb.tolist()}
    return result

# ----- ESEMPIO -----
if __name__ == '__main__':
    builder = GraphBuilder()
    title, stats = builder.build_graph('Q177')  # Pizza
    builder.graph_to_web_print(title, output_file="pizza_graph.html")
    #print("Features:")
    #for k, v in stats.items():
    #    print(f"  {k}: {v}")
    #feat = build_graph_features("Q55641393")
    #for k, v in feat.items():
    #    print(f"  {k}: {v}")
