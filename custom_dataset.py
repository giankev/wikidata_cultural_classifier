from wiki_extractor import WikidataExtractor
from dataset_parser import DatasetParser
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class CustomData:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # Funzione Worker per creare un'istanza di WikidataExtractor
    def create_extractor_task(self, idx_identifier):
        """
        Tenta di creare un oggetto WikidataExtractor.
        Ritorna (indice, oggetto_extractor) se successo.
        Ritorna (indice, None) se fallisce (e stampa un errore).
        """
        idx, identifier = idx_identifier
        try:
            # La chiamata a __init__ fa il fetch. Se fallisce, solleva eccezione.
            extractor = WikidataExtractor(identifier)
            return idx, extractor
        except Exception as e:
            # Se WikidataExtractor fallisce (es. rete, ID non trovato), cattura l'errore.
            print(f"WARN: Failed to create extractor for index {idx} (ID/URL: {identifier}): {e}")
            return idx, None

    def query_wikidata_for_items(self) -> dict:
      '''
      Restituisce un dic [index, wikidataextractor_obj]
      '''
        # Prepara gli argomenti per i threads
      args = list(self.df['item'].items())  # lista di (index, wikidata_url)
      fetched_results = {}
      #Nota che per alcuni valori di MAXWORKERS possono creare errori del tipo "HTTP Error 429: Too Many Requests".
      #Questo dipende anche dal numero di item a cui si fa richiesta al server
      MAXWORKERS = 5 # a volte se impostato a 10 non crea a problemi, penso che dipenda dal carico del server nel momento delle richeiste.

      # Esecuzione in parallelo di #max_workers threads
      with ThreadPoolExecutor(max_workers=MAXWORKERS) as executor:
          futures = {executor.submit(self.create_extractor_task, arg): arg[0] for arg in args}
          for future in as_completed(futures):
              idx = futures[future]
              try:
                  _, result_obj = future.result()
                  fetched_results[idx] = result_obj
              except Exception as e:
                  print(f"ERROR: Critical error getting result for index {idx}: {e}")
                  fetched_results[idx] = None

      return fetched_results

    def check_fetch_results(self, fetched_results: dict) -> None:
      total_attempted = len(fetched_results) # Numero di item per cui abbiamo provato il fetch
      # Conta quanti valori nel dizionario sono None
      failures = sum(1 for result_obj in fetched_results.values() if result_obj is None)
      successes = total_attempted - failures

      print(f"\nFetch Summary:")
      print(f"  Attempted: {total_attempted}")
      print(f"  Successful Fetches (Extractor created): {successes}")
      print(f"  Failed Fetches (Extractor is None): {failures}")

    def add_feature(self) -> dict: #aggiunge le ulteriori feature calcolate dai metodi della classe DatasetParser
      print("Adding feature...")
      fetched_results = self.query_wikidata_for_items()
      self.check_fetch_results(fetched_results)

      df_copia = self.df.copy()
      df_copia['number_sitelinks'] = pd.NA
      df_copia['sitelinks_translation_entropy'] = pd.NA
      df_copia['number_claims'] = pd.NA
      df_copia['po_P495'] = pd.NA
      df_copia['po_P1343'] = pd.NA

      for idx, extractor_instance in fetched_results.items():
          if extractor_instance:
              try:

                  parser = DatasetParser(extractor_instance)
                  num_links = parser.get_number_sitelinks()
                  entropy = parser.sitelinks_translation_entropy()
                  num_claims = parser.get_number_claims()
                  po_P495 = parser.get_presence_of_P495()
                  po_P1343 = parser.get_presence_of_P31()

                  # aggiunta delle feature calcolate da 'DasetParser'
                  df_copia.at[idx, 'number_sitelinks'] = num_links
                  df_copia.at[idx, 'sitelinks_translation_entropy'] = entropy
                  df_copia.at[idx, 'number_claims'] = num_claims
                  df_copia.at[idx, 'po_P495'] = po_P495
                  df_copia.at[idx, 'po_P1343'] = po_P1343


              except Exception as e:
                  print(f"ERROR: Parsing failed for index {idx} (Entity: {extractor_instance.get_entity_id()}): {e}")

              
      print("Feature added...")
      return df_copia

    def preprocess_data(self, df,columns_to_drop: list = None, type_col: str = 'type', category_col: str = 'category') -> pd.DataFrame:
        """
        Applica preprocessing a un DataFrame:
        1. Elimina righe con NaN in QUALSIASI colonna.
        2. Droppa colonne specificate.
        3. Codifica la colonna 'type' (binaria 0/1).
        4. Applica One-Hot Encoding alla colonna 'category'.

        Args:
            df (pd.DataFrame): Il DataFrame di input.
            columns_to_drop (list, optional): Lista di nomi di colonne da eliminare.
                                              Defaults a ["item", "name", "description", "subcategory"].
            type_col (str, optional): Nome della colonna tipo binaria. Defaults a 'type'.
            category_col (str, optional): Nome della colonna categoria multi-classe. Defaults a 'category'.

        Returns:
            pd.DataFrame: Il DataFrame preprocessato.
        """
        print(f"\n--- Preprocessing DataFrame (Initial rows: {len(df)}) ---")
        original_row_count = len(df)
        df_cleaned = df.dropna()
        final_row_count = len(df_cleaned)
        rows_deleted = original_row_count - final_row_count

        if rows_deleted > 0:
            print(f"Handling Missing Values: Eliminato {rows_deleted} righe con valori NaN.")
            print(f"  Righe rimanenti dopo dropna: {final_row_count}")
        else:
            print("Handling Missing Values: Nessuna riga con NaN trovata.")

        if columns_to_drop is None:
            columns_to_drop = ["item", "name", "description", "subcategory"]

        df_cleaned = df_cleaned.drop(columns=columns_to_drop)

        type_mapping = {"concept": 0, "entity": 1}
        df_cleaned['type'] = df_cleaned['type'].map(type_mapping)

        categories_list = [
            'Literature', 'Philosophy and Religion', 'Fashion', 'Food', 'Comics and Anime',
            'Visual Arts', 'Media', 'Performing Arts', 'Biology', 'Films', 'Music', 'Sports',
            'Geography', 'Architecture', 'Politics', 'History', 'Transportation',
            'Gestures and Habits', 'Books'
        ]

        df_encoded = pd.get_dummies(df_cleaned['category'], prefix='category', dtype=int, drop_first=False)
        df_cleaned = pd.concat([df_cleaned, df_encoded], axis=1)
        df_cleaned = df_cleaned.drop('category', axis=1)

        print(f"--- Preprocessing Completo (Final rows: {len(df_cleaned)}, Final columns: {len(df_cleaned.columns)}) ---")
        return df_cleaned
