from wikidata.client import Client
import requests

class WikidataExtractor:
    """
    Classe per estrarre informazioni da Wikidata a partire da un URL o un ID di entità.
    """
    _client = Client()

    def __init__(self, identifier):
        # Caricamento dell'item
        self.entity_id = self.extract_entity_id(identifier)
        self.item = self._fetch_item(self.entity_id)
        self.url_api = "https://en.wikipedia.org/w/api.php"

    @staticmethod
    def extract_entity_id(url_or_id: str) -> str:
        """
        Estrae l'ID di entità (es. Q42) da un URL di Wikidata o restituisce l'ID se è già fornito.
        """
        if url_or_id.startswith(('http://', 'https://')):
            return url_or_id.strip().split("/")[-1]
        return url_or_id

    def request_data(self):
      """
      Richiesta dati da Wikipedia tramite API, ritorna il contenuto della pagina.
      """

      default_return = '' # O None, se preferisci

      # Verifica preliminare sull'item Wikidata e sitelink enwiki
      if self.item is None or not hasattr(self.item, 'data'):
          # print(f"DEBUG [{self.entity_id}]: Impossibile richiedere testo Wiki, item Wikidata non valido.")
          return default_return
      sitelinks = self.item.data.get('sitelinks', {})
      enwiki_data = sitelinks.get('enwiki')
      if not enwiki_data or 'title' not in enwiki_data:
          # print(f"DEBUG [{self.entity_id}]: Sitelink 'enwiki' non trovato.")
          return default_return

      wiki_title = enwiki_data['title']
      params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": wiki_title,
        "format": "json",
        "redirects": 1
      }

      try:
          # 1. Fai la richiesta GET e ottieni l'OGGETTO Response
          response = requests.get(self.url_api, params=params, timeout=10)

          # 2. CONTROLLA lo status code PRIMA di fare .json()
          response.raise_for_status() # Solleva HTTPError per 4xx/5xx

          # 3. SOLO SE lo status è OK (2xx), procedi con il parsing JSON
          res_json = response.json()

          # 4. Estrai dati dal JSON (con controlli)
          pages = res_json.get("query", {}).get("pages", {})
          if not pages:
                print(f"WARN [{self.entity_id}]: Risposta JSON da Wikipedia API non contiene 'pages' per '{wiki_title}'.")
                return default_return

          page = next(iter(pages.values()))
          if page.get('missing') is not None or page.get('invalid') is not None:
                # print(f"DEBUG [{self.entity_id}]: Pagina Wikipedia '{wiki_title}' mancante/invalida.")
                return default_return

          text = page.get("extract", default_return) # Usa default se 'extract' manca
          return text

      except requests.exceptions.HTTPError as http_err:
          # Errore specifico HTTP catturato da raise_for_status()
          print(f"ERROR [{self.entity_id}]: Errore HTTP Wikipedia per '{wiki_title}': {http_err.response.status_code} {http_err}")
          return default_return
      except requests.exceptions.RequestException as req_err:
          # Altri errori di rete (timeout, connessione, ecc.)
          print(f"ERROR [{self.entity_id}]: Errore di rete Wikipedia per '{wiki_title}': {req_err}")
          return default_return
      except json.JSONDecodeError as json_err:
          # Errore specifico se la risposta (anche con status 200) non è JSON valido
          print(f"ERROR [{self.entity_id}]: Errore decodifica JSON Wikipedia per '{wiki_title}': {json_err}")
          # Stampa l'inizio del testo ricevuto per debug
          try:
                print(f"       Response Status: {response.status_code}, Text: {response.text[:200]}...")
          except NameError: # response potrebbe non essere definito se l'errore è prima
                pass
          return default_return
      except Exception as e:
          # Cattura altri errori imprevisti durante l'estrazione dal JSON
          print(f"ERROR [{self.entity_id}]: Errore generico processando Wikipedia per '{wiki_title}': {e}")
          return default_return

    def _fetch_item(self, entity_id: str):
        """
        Recupera l'item da Wikidata, caricando etichette, descrizioni e sitelinks.
        """
        try:
            return self._client.get(entity_id, load=True)
        except Exception as e:
            raise ValueError(f"Errore nel recupero dell'entità {entity_id}: {e}")

    def get_entity_id(self) -> str:
        """
        Restituisce l'etichetta dell'entità nella lingua specificata (default: en).
        """
        return self.entity_id

    def get_label(self) -> str:
        """
        Restituisce l'etichetta dell'entità nella lingua specificata (default: en).
        """
        return self.item.label

    def get_description(self) -> str:
        """
        Restituisce la descrizione dell'entità nella lingua specificata (default: en).
        """
        return self.item.description

    def get_sitelinks(self):
        """
        Restituisce un dizionario dei sitelink di Wikipedia:
          {lang: url, ...}
        """
        raw = self.item.data.get('sitelinks', {})
        result = {}
        for site_key, site_data in raw.items():
            if site_key.endswith('wiki') and not site_key.startswith('commons'):
                lang = site_key.replace('wiki', '')
                title = site_data.get('title', '')
                url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                result[site_key] = {'url': url, 'title': title}

        return result

    def get_claims(self) -> dict:
        """
        Restituisce tutte le claim dell'entità come dict:
        {property_id: [valore1, valore2, ...], ...}
        """
        raw = self.item.data.get('claims', {})
        claims = {}
        for prop, cl_list in raw.items():
            prop_id = prop
            values = cl_list
            claims[prop_id] = values
        return claims

    def get_number_claims(self) -> int:
        """
        Restituisce il numero di claim dell'entità.
        """
        return len(self.get_claims())



if __name__ == '__main__':
    # Esempio di utilizzo
    # PASTA ALLA GRICIA -> cultural exclusive
    wikidata_url = "https://www.wikidata.org/wiki/Q55641393"
    # PIZZA -> cultural representative
    #wikidata_url = "https://www.wikidata.org/wiki/Q177"
    entity = WikidataExtractor(wikidata_url)
    print("Entity ID:", entity.get_entity_id())
    print("Label:", entity.get_label())
    print("Description:", entity.get_description())
    print("Sitelinks:", entity.get_sitelinks())
    print("Claims:", entity.get_claims())
    print("Numero di claims totali:", entity.get_number_claims())
