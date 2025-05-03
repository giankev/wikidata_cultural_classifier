# extract graphs topological and mesoscale features, add graphs embedding
# out: augmented csv data file

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from graph import build_graph_features  # graph helper
INPUT_CSV = os.path.expanduser('~/content/train.csv')
OUTPUT_CSV = os.path.expanduser('~/content/train_augmented_graphs.csv')
WORKERS = 10 # num threads (can be bigger than hw num cores/threads)

if __name__ == '__main__':
    # Carica e scarta la colonna description
    df = pd.read_csv(INPUT_CSV).drop(columns=['description'])
    rows = df.to_dict('records')
    total = len(rows)
    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # submit dei job: QID estratto dall'item URL
        future_to_row = {
            executor.submit(build_graph_features, row['item'].split('/')[-1]): row
            for row in rows
        }

        # Processa mano a mano che i risultati tornano
        for idx, future in enumerate(as_completed(future_to_row), start=1):
            row = future_to_row[future]
            qid = row['item'].split('/')[-1]
            try:
                feats = future.result()
            except Exception as e:
                print(f"[ERROR] build_graph_features failed for {qid}: {e}", flush=True)
                # skip
                continue
            merged = {**row, **feats}
            results.append(merged)
            print(f"Processed {idx}/{total} – QID {qid}", end="\r", flush=True)

    # Crea DataFrame e salva
    df_aug = pd.DataFrame(results)
    df_aug.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAugmented CSV saved to {OUTPUT_CSV}")
