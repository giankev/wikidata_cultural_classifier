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
        self.text = self.request_data()

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

        url = self.item.data['sitelinks']['enwiki']['title']
        if not url or 'title' not in self.item.data['sitelinks']['enwiki']:
          return '' 

        params = {
          "action": "query",
          "prop": "extracts",
          "explaintext": True,
          "titles": url,
          "format": "json",
          "redirects": 1
        }

        try:
              res = requests.get(self.url_api, params=params).json()
              res.raise_for_status() # Solleva eccezione per errori HTTP (4xx, 5xx)
              page = next(iter(res["query"]["pages"].values()))
              text = page.get("extract", "")
              return text
        except Exception as e:
              print(f"Errore nella chiamata all'API: {e}")
              return ''

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

    def get_text(self):
        """
        Richiesta dati da Wikipedia tramite API, ritorna il contenuto della pagina.
        """
        return self.text


if __name__ == '__main__':
    # Esempio di utilizzo
    # PASTA ALLA GRICIA -> cultural exclusive
    # wikidata_url = "https://www.wikidata.org/wiki/Q55641393"
    # PIZZA -> cultural representative
    wikidata_url = "https://www.wikidata.org/wiki/Q177"
    entity = WikidataExtractor(wikidata_url)
    text =  entity.get_text()
    print("Entity ID:", entity.get_entity_id())
    print("Label:", entity.get_label())
    print("Description:", entity.get_description())
    print("Sitelinks:", entity.get_sitelinks())
    print("Claims:", entity.get_claims())
    print("Numero di claims totali:", entity.get_number_claims())
    print("\n", text)
