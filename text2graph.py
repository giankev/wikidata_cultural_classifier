import os
import re
import math

# Formatting
import unicodedata
from colorama import Fore, Back

# Graph
from collections import Counter, defaultdict

# Graph visualization
from pyvis.network import Network
import networkx as nx

# Wikipedia
import wikipedia
from wiki_extractor import WikidataExtractor
from dataset_parser import DatasetParser

# Stopwords - multi language
import stopwordsiso as stopwords

# Setup
SECTION_BLACKLIST = ["Bibliografia", "Bibliography", "Altri progetti", "References", "Note"]
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
        if stopwords.has_lang(language):                        # check if there is a stopwords for the language
            self.stopwords = stopwords.stopwords(language)      # Selected language stopwords
        else:
            self.stopwords = stopwords.stopwords("en")

        # Others
        self.sections  = {}           # populated by fetch_sections()
        self.nodes     = set()        # will hold all node-keys
        self.edge_freq = Counter()    # Counter[(u,v)] → weight

        # section_texts maps each section title -> list of paragraph strings
        self.sections: dict[str, list[str]] = {}

    """
    Splits the page content into sections
    - each section has a title - paragraphs mapping
    """
    def fetch_sections(self):
        wikipedia.set_lang(self.language)
        page = wikipedia.page(self.title)
        raw = page.content

        # Compile a regex that matches lines like "== Some Title ==", use it to extract title and sections from page content
        heading_re = re.compile(r'^\s*==+\s*(.+?)\s*==+\s*$', re.MULTILINE)

        # Split the text on those headings.
        #    re.split returns: [intro_text, title1, body1, title2, body2, ...]
        parts = heading_re.split(raw)

        #sections: dict[str, list[str]] = {}
        self.sections.clear()

        # The very first chunk is everything before the first heading
        intro = parts[0].strip()
        if intro:
            # break intro into paragraphs on blank lines
            paras = [p.strip() for p in re.split(r'\n\s*\n', intro) if p.strip()]
            self.sections["Introduction"] = paras

        # The remaining parts come in pairs: (title, body)
        for title, body in zip(parts[1::2], parts[2::2]):
            title = title.strip()
            # split body into paragraphs
            paras = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
            self.sections[title] = paras

        return self.sections
        
    """
    Improved tokenizer:
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
        self.title = item.get_label(lang = language)

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

    """
    Get functions
    """
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
    language = "en" # text and stopwords language
    text2graph = Text2Graph(item, language, properties_to_check)

    secs = text2graph.fetch_sections()
    text2graph.build_graph()
    text2graph.collapse_linear_paths()

    file_name = f"{text2graph.get_title()}_{text2graph.get_lang()}_graph.html"
    text2graph.print_sections(truncate = False)
    text2graph.graph_to_web_print(output_file=file_name)