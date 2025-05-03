# using stopwords and tokens build a graphs
# input: wikipedia page description - titles and sections

import os
import re
import math
import requests

# Formatting
import unicodedata
import numpy as np
from colorama import Fore, Back

# Graph
from collections import Counter, defaultdict

# Graph visualization
from pyvis.network import Network
import networkx as nx

# Wikipedia
import wikipedia
import pycountry
from wiki_extractor import WikidataExtractor
from dataset_parser import DatasetParser
import urllib.parse

# Stopwords - multi language
import stopwordsiso as stopwords

# Setup
SECTION_BLACKLIST = ["Bibliografia", "Bibliography", "Altri progetti", "References", "Note", "See also", "Notes", "Further reading"]
GRAPH_WEB_DIR = "web_graphs"

class Text2Graph:
    """
    Given a wikidata item, extracts the desired lanugage wikipedia page content, and builds a graph over it
    """
    def __init__(self, item, language="en", properties_to_check=None):
        self.item     = item
        self.language = language
        self.props    = properties_to_check or {}
        self.title    = (item.get_label(lang=language) or "").lower()

        # Init stopwords
        if stopwords.has_lang(language):                                        # check if there is a stopwords for the language
            self.stopwords = stopwords.stopwords(self.__get_stopword_lang())    # Selected language stopwords
        else:
            self.stopwords = stopwords.stopwords("en")

        # Others
        self.sections  = {}           # populated by fetch_sections()
        self.nodes     = set()        # will hold all node-keys
        self.edge_freq = Counter()    # Counter[(u,v)] → weight

        # section_texts maps each section title -> list of paragraph strings
        self.sections: dict[str, list[str]] = {}

    """
    Get possible stopword needed languages from the selected language and the item coutnry or country of origin
    """
    def __get_stopword_lang(self):
        langs = set()

        if isinstance(self.item, WikidataExtractor):
            claims = self.item.get_claims()

            country_props = ["P17", "P495"]  # Paese e Paese di origine

            for prop in country_props:
                for claim in claims.get(prop, []):
                    try:
                        country_qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                        country_name = self.item.get_label_for_qid(country_qid, lang="en")
                        
                        # Usa pycountry per trovare la lingua associata al paese
                        country = pycountry.countries.get(name=country_name)
                        if not country:
                            # Tenta anche con il nome comune
                            country = pycountry.countries.search_fuzzy(country_name)[0]
                        
                        if country:
                            # Cerca la lingua associata alla country
                            langs_for_country = pycountry.languages
                            matches = [l for l in langs_for_country if hasattr(l, 'alpha_2') and country.alpha_2.lower() in l.alpha_2.lower()]
                            for lang in matches:
                                langs.add(lang.alpha_2)
                    except Exception as e:
                        # print(f"Error {claim}: {e}")
                        continue

        if self.language not in langs:
            langs.add(self.language)

        return list(langs)
    
    # -------- Graph construction --------
    
    def fetch_sections(self):
        """
        Fetch and clean the wikitext for the Wikipedia page in self.language,
        then split into a dict: section_title -> list of paragraph strings.
        """
        # Get the exact sitelink URL for this language
        wiki_url = self.item.get_wikipedia_url(lang=self.language)
        if not wiki_url:
            return {}

        # Extract the page title from the URL
        parsed = urllib.parse.urlparse(wiki_url)
        page_title = urllib.parse.unquote(parsed.path.rsplit('/', 1)[-1])

        # Prepare API call
        api = f"https://{self.language}.wikipedia.org/w/api.php"
        session = getattr(self, '_wiki_session', None)
        if session is None:
            session = requests.Session()
            self._wiki_session = session

        resp = session.get(api, params={
            'action':    'parse',
            'page':      page_title,
            'prop':      'wikitext',
            'redirects': True,
            'format':    'json'
        })
        data = resp.json().get('parse', {})
        raw = data.get('wikitext', {}).get('*', '')
        if not raw:
            return {}

        # Remove HTML comments <!-- ... -->
        raw = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL)

        # Remove <ref>...</ref> and self-closing <ref/>
        raw = re.sub(r'<ref[^>]*?>.*?</ref>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'<ref[^/>]*/>', '', raw)

        # Remove tables/infoboxes {| ... |}
        raw = re.sub(r'\{\|[\s\S]*?\|\}', '', raw)

        # Iteratively strip templates {{ ... }}
        #    (repeat until no more templates)
        pattern = re.compile(r'\{\{[^{}]*\}\}')
        while True:
            raw, count = pattern.subn('', raw)
            if count == 0:
                break

        # Remove File: and Category: links entirely
        raw = re.sub(r'\[\[File:[^\]]+\]\]', '', raw)
        raw = re.sub(r'\[\[Category:[^\]]+\]\]', '', raw)

        # Convert internal links [[A|B]] → B, and [[A]] → A
        def _link_repl(m):
            text = m.group(2) or m.group(1)
            return text
        raw = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', _link_repl, raw)
        raw = re.sub(r'\[\[([^\]]+)\]\]', r'\1', raw)

        # Remove any leftover HTML tags
        raw = re.sub(r'<[^>]+>', '', raw)

        # SPLIT INTO SECTIONS
        heading_re = re.compile(r'^\s*==+\s*(.+?)\s*==+\s*$', re.MULTILINE)
        parts = heading_re.split(raw)

        self.sections.clear()

        # Introduction (before first heading)
        intro = parts[0].strip()
        if intro:
            paras = [p.strip() for p in re.split(r'\n\s*\n', intro) if p.strip()]
            self.sections["Introduction"] = paras

        # Remaining (title, body) pairs
        for title, body in zip(parts[1::2], parts[2::2]):
            sec_title = title.strip()
            paras     = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
            self.sections[sec_title] = paras

        return self.sections
        
    """
    Tokenizer:
    - Lowercases
    - Normalizes accented characters
    - Removes punctuation
    - Splits on whitespace
    - Keeps only alphabetic tokens
    """
    @staticmethod
    def tokenize(text: str) -> list[str]:
        # Normalize text (e.g., é → e + ')
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ASCII', 'ignore').decode('utf-8')  # remove accents

        # Lowercase and remove non-alphanumeric characters (punctuation)
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text.lower())

        # Tokenize and keep only alphabetic tokens
        tokens = text.split()
        return [t for t in tokens if t.isalpha()]


    """
    1) Add central node: the item title
    2) Add Wikidata-feature edges:
        - For each property in self.props:
            central -> prop_label
            prop_label -> each claim value
    3) For each sentence in each section:
        - tokenize, drop stopwords
        - add edge central -> first_token
        - add edges token[i] -> token[i+1]
    """
    def build_graph(self):
        # --- 1) Central node ---
        central = self.title or "unknown"
        self.nodes.add(central)

        # --- 2) Wikidata features ---
        dp = DatasetParser(self.item)

        # Claims
        target = dp.claims_target(self.props)
        values = dp.get_claims_value(target)
        for pid, vlist in values.items():
            for v in vlist:
                if isinstance(v, str) and v.startswith("Q"):
                    val_label = self.item.get_label_for_qid(v, lang=self.language)
                else:
                    val_label = str(v)

                val_node = val_label.lower()
                self.nodes.add(val_node)
                self.edge_freq[(central, val_node)] += 1

        # --- 3) Text-based edges ---
        for sec, paras in self.sections.items():

            # drop blacklisted sections
            if sec in SECTION_BLACKLIST:
                continue

            # build using section texts
            for para in paras:
                for sent in re.split(r'[\.!?]+', para):
                    sent = sent.strip()
                    if not sent:
                        continue

                    toks = self.tokenize(sent)
                    toks = [t for t in toks if t not in self.stopwords]
                    if not toks:
                        continue

                    # central → first token
                    first = toks[0]
                    self.nodes.add(first)
                    self.edge_freq[(central, first)] += 1

                    # chain adjacent tokens
                    for u, v in zip(toks, toks[1:]):
                        self.nodes.update([u, v])
                        self.edge_freq[(u, v)] += 1

    """
    Collapse nodes with less than 3 total connections (in + out).
    Chains like "pasta -> alla -> gricia" become "pasta alla gricia".
    Disconnected nodes are removed.
    """
    def collapse_linear_paths(self):
        import copy
        G = nx.DiGraph()

        # Build the initial graph from current edge_freq
        for (u, v), w in self.edge_freq.items():
            G.add_edge(u, v, weight=w)

        visited = set()
        new_nodes = set()
        new_edge_freq = Counter()

        for node in list(G.nodes()):
            if node in visited:
                continue

            # Count total connections
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            total_deg = in_deg + out_deg

            # Skip complex/hub nodes
            if total_deg >= 3 or in_deg > 1 or out_deg > 1:
                new_nodes.add(node)
                continue

            # Start path collapse forward
            path = [node]
            current = node
            while True:
                neighbors = list(G.successors(current))
                if len(neighbors) != 1:
                    break
                nxt = neighbors[0]

                # Only keep merging if the next node is also linear (degree < 3)
                if (G.in_degree(nxt) + G.out_degree(nxt)) >= 3 or G.in_degree(nxt) > 1 or G.out_degree(nxt) > 1:
                    break

                path.append(nxt)
                visited.add(nxt)
                current = nxt

            # Collapse the chain into one node
            collapsed = ' '.join(path)
            new_nodes.add(collapsed)

            # Connect input edges (excluding internal)
            for u in G.predecessors(path[0]):
                if u not in path:
                    new_edge_freq[(u, collapsed)] += G[u][path[0]]['weight']

            # Connect output edges (excluding internal)
            for v in G.successors(path[-1]):
                if v not in path:
                    new_edge_freq[(collapsed, v)] += G[path[-1]][v]['weight']

            visited.update(path)

        # Add unvisited original nodes
        for node in G.nodes():
            if node not in visited:
                new_nodes.add(node)

        # Add unvisited original edges
        for u, v in G.edges():
            if u not in visited and v not in visited:
                new_edge_freq[(u, v)] += G[u][v]['weight']

        # Update graph
        self.nodes = new_nodes
        self.edge_freq = new_edge_freq

    """
    Change the wanted language
    """
    def set_language(self, lang = "en"):
        self.language = lang
        self.title = self.item.get_label(lang = language)

    # -------- Graph visualization --------

    """
    Display the extracted titles and paragraphs
    """
    def print_sections(self, truncate = True):
        print(f"{Fore.CYAN}Contents of the {Fore.BLACK + Back.LIGHTYELLOW_EX}{self.language}{Fore.CYAN + Back.RESET} wikipedia page of {Fore.BLACK + Back.LIGHTYELLOW_EX}{self.title}{Back.RESET}.")
        print(f"{Fore.RED + Back.LIGHTWHITE_EX}Red titles are not considered for graph construction!{Back.RESET} \n")
        for sec, paras in self.sections.items():
            # Print section title
            if sec in SECTION_BLACKLIST:
                print(Fore.RED + sec)
            else:
                print(Fore.LIGHTWHITE_EX + sec)
            
            # Print section text
            for p in paras:
                if truncate:
                    print(Fore.RESET, p[:60] + ("…" if len(p) > 60 else ""))
                else:
                    print(Fore.RESET + p)
            print(Fore.RESET)

    """
    Convert the built graph to an html page for easy displaying
    """
    def graph_to_web_print(self, output_file="graph.html"):
        # build your NetworkX graph as before …
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for (u, v), w in self.edge_freq.items():
            G.add_edge(u, v, value=w)

        net = Network(directed=True, height="900px", width="100%")
        net.from_nx(G)

        if not os.path.exists(GRAPH_WEB_DIR):                                                   # ensure output directory exists
            os.makedirs(GRAPH_WEB_DIR)          

        output_path = os.path.join(GRAPH_WEB_DIR, output_file)                                  # set ouput directory
        net.write_html(output_path)                                                             # instead of net.show(), use write_html
        print(f"Interactive graph written to {output_file}. Open that in your browser.")

    # -------- Basic graph features extraction ----------

    def get_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for (u, v), w in self.edge_freq.items():
            G.add_edge(u, v, weight=w)
        return G

    def get_num_nodes(self):
        return len(self.nodes)

    def get_num_edges(self):
        return len(self.edge_freq)

    def get_average_degree(self):
        if self.get_num_nodes() == 0:
            return 0
        return 2 * self.get_num_edges() / self.get_num_nodes()

    def get_density(self):
        G = self.get_graph()
        return nx.density(G)

    def get_average_clustering(self):
        G = self.get_graph()
        if len(G) < 2:
            return 0
        undirected_G = G.to_undirected()
        return nx.average_clustering(undirected_G)

    def get_centrality_measures(self):
        G = self.get_graph()
        centrality = {}

        try:
            deg = nx.degree_centrality(G)
            bet = nx.betweenness_centrality(G)
            clo = nx.closeness_centrality(G)

            centrality['degree_centrality'] = np.mean(list(deg.values()))
            centrality['betweenness_centrality'] = np.mean(list(bet.values()))
            centrality['closeness_centrality'] = np.mean(list(clo.values()))
        except Exception as e:
            print(f"Error computing centrality measures: {e}")
            centrality = {'degree_centrality': 0, 'betweenness_centrality': 0, 'closeness_centrality': 0}

        return centrality

    def get_assortativity(self):
        G = self.get_graph()
        try:
            if len(G) > 1:
                return nx.degree_assortativity_coefficient(G)
            else:
                return None
        except Exception as e:
            print(f"Error computing assortativity: {e}")
            return None

    def get_transitivity(self):
        G = self.get_graph()
        return nx.transitivity(G)

    def get_edge_weight_stats(self):
        if not self.edge_freq:
            return {"mean": 0, "std": 0, "max": 0, "min": 0}
        weights = list(self.edge_freq.values())
        return {
            "mean": np.mean(weights),
            "std": np.std(weights),
            "max": np.max(weights),
            "min": np.min(weights)
        }

    # -------- Mesoscale Features --------

    def get_average_clustering(self) -> float:
        """Average clustering coefficient."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_weighted_edges_from([(u, v, w) for (u, v), w in self.edge_freq.items()])
        UG = G.to_undirected()
        return nx.average_clustering(UG, weight='weight')

    def get_modularity(self) -> float:
        """Modularity score of partition (using Louvain if available)."""
        try:
            import community as community_louvain
            G = nx.Graph()
            G.add_edges_from(self.edge_freq.keys())
            partition = community_louvain.best_partition(G)
            return community_louvain.modularity(partition, G)
        except ImportError:
            print("Install 'python-louvain' package for modularity calculation.")
            return -1.0
        except Exception:
            return -1.0

    # -------- Topological Features --------

    def get_node_count(self) -> int:
        """Total number of nodes."""
        return len(self.nodes)


    def get_edge_count(self) -> int:
        """Total number of edges."""
        return len(self.edge_freq)


    def get_density(self) -> float:
        """Graph density (for directed graphs)."""
        n = len(self.nodes)
        m = len(self.edge_freq)
        if n <= 1:
            return 0.0
        return m / (n * (n - 1))


    def get_average_degree(self) -> float:
        """Average degree."""
        if not self.nodes:
            return 0.0
        return (2 * len(self.edge_freq)) / len(self.nodes)

    def get_average_shortest_path_length(self) -> float:
        """Average shortest path length (on largest weakly connected component)."""
        G = nx.DiGraph()
        G.add_edges_from(self.edge_freq.keys())
        try:
            if nx.is_weakly_connected(G):
                return nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph)
        except Exception:
            return -1.0

    # -------- Other potentially useful features --------

    def get_graph_entropy(self) -> float:
        """Shannon entropy of node degree distribution."""
        G = nx.DiGraph()
        G.add_edges_from(self.edge_freq.keys())
        degree_sequence = [deg for _, deg in G.degree()]
        if not degree_sequence:
            return 0.0
        total = sum(degree_sequence)
        probs = [deg / total for deg in degree_sequence]
        return -sum(p * math.log2(p) for p in probs if p > 0)


    def get_in_out_degree_ratio(self) -> float:
        """Mean ratio of in-degree to out-degree."""
        G = nx.DiGraph()
        G.add_edges_from(self.edge_freq.keys())
        ratios = []
        for node in G.nodes():
            indeg = G.in_degree(node)
            outdeg = G.out_degree(node)
            if outdeg > 0:
                ratios.append(indeg / outdeg)
        if ratios:
            return sum(ratios) / len(ratios)
        else:
            return 0.0

    # -------- "Cultural Signal" hint --------

    def get_label_overlap_with_country(self) -> float:
        """Overlap between node labels and country name or nationality (basic heuristic)."""
        if not hasattr(self.item, 'get_label_for_qid'):
            return 0.0
        country_qids = ["P17", "P495"]  # Country and Country of Origin
        terms = set()
        for pid in country_qids:
            target = self.props.get(pid)
            if target:
                try:
                    country_name = self.item.get_label_for_qid(target, lang=self.language)
                    if country_name:
                        tokens = self.tokenize(country_name)
                        terms.update(tokens)
                except Exception:
                    continue
        if not terms:
            return 0.0

        count = 0
        for node in self.nodes:
            node_tokens = self.tokenize(node)
            if any(tok in terms for tok in node_tokens):
                count += 1
        return count / len(self.nodes) if self.nodes else 0.0

    # -------- DNN helper methods --------

    def get_all_features_as_dict(self) -> dict:
        """Raccoglie tutte le feature in un unico dict, pronto per DNN."""
        features = {
            'num_nodes':                     self.get_num_nodes(),
            'num_edges':                     self.get_num_edges(),
            'average_degree':                self.get_average_degree(),
            'density':                       self.get_density(),
            'average_clustering':            self.get_average_clustering(),
          # 'num_connected_components':      self.get_num_connected_components(),
          # 'largest_component_size':        self.get_largest_component_size(),
          # 'average_shortest_path_length':  self.get_average_shortest_path_length(),
          # 'diameter':                      self.get_diameter(),
            'assortativity':                 self.get_assortativity() or 0,
            'transitivity':                  self.get_transitivity(),
          # 'isolated_nodes':                self.get_isolated_nodes_count(),
            'language':                      self.language,
            'graph_entropy':                 self.get_graph_entropy(),
            'modularity':                    self.get_modularity(),
        }

        # centrality
        cent = self.get_centrality_measures()
        features.update({
            'avg_degree_centrality':     cent['degree_centrality'],
            'avg_betweenness_centrality':cent['betweenness_centrality'],
            'avg_closeness_centrality':  cent['closeness_centrality'],
        })

        # edge weights
        ew = self.get_edge_weight_stats()
        features.update({
            'edge_weight_mean': ew['mean'],
            'edge_weight_std':  ew['std'],
          # 'edge_weight_max':  ew['max'],
          # 'edge_weight_min':  ew['min'],
        })

        # la tua feature culturale
        features['label_overlap_with_country'] = self.get_label_overlap_with_country()

        return features

    # -------- get helper methods --------

    def get_title(self):
        return self.title
    
    def get_lang(self):
        return self.language



# Example
if __name__ == '__main__':
    wikidata_url = "https://www.wikidata.org/wiki/Q55641393"
    item = WikidataExtractor(wikidata_url)
    properties_to_check = {
        "P495": "Country of Origin",
        "P17": "Country",
        "P361": "Part Of",
        "P31": "Instance Of",
        "P279": "Subclass Of",
        "P1705": "Native Label",
        "P2012": "Cuisine"
    }

    #url = item.get_wikipedia_url()
    language = "en" # text language
    text2graph = Text2Graph(item, language, properties_to_check)

    # Build graph
    secs = text2graph.fetch_sections()
    text2graph.build_graph()
    text2graph.collapse_linear_paths()

    # Save graph visualization
    file_name = f"{text2graph.get_title()}_{text2graph.get_lang()}_graph.html"
    text2graph.print_sections(truncate = False)
    text2graph.graph_to_web_print(output_file=file_name)

    # Extract features
    print("\n--- All Graph Features ---\n")
    features = text2graph.get_all_features_as_dict()
    for name, value in features.items():
        print(f"{name}: {value}")