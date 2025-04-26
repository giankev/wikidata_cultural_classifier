import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from wiki_extractor import WikidataExtractor
from text2graph import Text2Graph

INPUT_CSV    = '~/content/train.csv'
OUTPUT_FOLDER = 'graph_features'
OUTPUT_CSV   = os.path.join(OUTPUT_FOLDER, 'features.csv')

PROPS = {
    "P495": "Country of Origin",
    "P17":  "Country",
    "P361": "Part Of",
    "P31":  "Instance Of",
    "P279": "Subclass Of",
    "P1705":"Native Label",
    "P2012":"Cuisine"
}

MAX_WORKERS = 8

def process_row(row) -> dict | None:
    """
    Build & analyze the English graph for a single DataFrame row.
    Returns a feature-dict or None on error.
    """
    item_url = row['item']
    metadata = {
        'name':        row.get('name', ''),
        'description': row.get('description', ''),
        'type':        row.get('type', ''),
        'category':    row.get('category', ''),
        'subcategory': row.get('subcategory', ''),
        'label':       row.get('label', '')
    }

    try:
        wd_item = WikidataExtractor(item_url)
        t2g     = Text2Graph(wd_item, language='en', properties_to_check=PROPS)
        t2g.fetch_sections()
        t2g.build_graph()
        t2g.collapse_linear_paths()

        feats = t2g.get_all_features_as_dict()
        feats.update(metadata)
        feats['item']     = item_url
        feats['language'] = 'en'
        return feats

    except Exception as e:
        print(f"[ERROR] {item_url}: {e}")
        return None

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    df = pd.read_csv(INPUT_CSV)
    num_items = len(df)
    print(f"Processing {num_items} items with {MAX_WORKERS} threads...")

    all_features = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): idx
                   for idx, row in df.iterrows()}

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"\rComplete: {completed}/{num_items}", end='', flush=True)
            if result is not None:
                all_features.append(result)

    print("\nWriting outputs…")
    pd.DataFrame(all_features).to_csv(OUTPUT_CSV, index=False)

    print(f"Done. Features written to: {OUTPUT_CSV}")
