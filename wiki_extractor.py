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

    def request_data(self, title: str = "enwiki"):
        """
        Richiesta dati da Wikipedia tramite API, ritorna il contenuto della pagina.
        """
        params = {
          "action": "query",
          "prop": "extracts",
          "explaintext": True,
          "titles": title,
          "format": "json",
          "redirects": 1
        }

        try:
              res = requests.get(self.url_api, params=params).json()
              page = next(iter(res["query"]["pages"].values()))
              text = page.get("extract", "")
              return text
        except Exception as e:
              print(e)
              return None

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

    def get_label(self, lang = "en") -> str:
        """
        Restituisce l'etichetta dell'entità nella lingua specificata (default: 'en').
        Se non esiste una label in quella lingua, tenta di restituire quella inglese
        o, in ultima istanza, la prima label disponibile.
        """
        labels = self.item.data.get("labels", {})
        # Try the requested language
        if lang in labels:
            return labels[lang]["value"]
        # Fallback to English if different
        if lang != "en" and "en" in labels:
            return labels["en"]["value"]
        # Finally, try to return the first available label
        if labels:
            return next(iter(labels.values()))["value"]
        # Else there is nothing to return
        return None

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
    
    def get_wikipedia_url(self, lang='en'):
        """
        Restituisce il link alla pagina wikipedia dell'item
        lang default = english, se l'argomento è NULL restituisce tutti i sitelinks
        """
        sitelinks = self.get_sitelinks()
        if sitelinks:
            if lang:
                # filter only the specified language
                sitelink = sitelinks.get(f'{lang}wiki')
                if sitelink:
                    wiki_url = sitelink.get('url')
                    if wiki_url:
                        return requests.utils.unquote(wiki_url)
            else:
                # return all of the urls
                wiki_urls = {}
                for key, sitelink in sitelinks.items():
                    wiki_url = sitelink.get('url')
                    if wiki_url:
                        wiki_urls[key] = requests.utils.unquote(wiki_url)
                return wiki_urls
        return None   

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
    
    def get_label_for_qid(self, qid, lang="en"):
        try:
            entity = self._client.get(qid, load=True)
            return entity.label.get(lang, qid)
        except:
            return qid  # fallback to QID



if __name__ == '__main__':
    # Esempio di utilizzo
    wikidata_url = "https://www.wikidata.org/wiki/Q55641393"
    entity = WikidataExtractor(wikidata_url)
    print("Entity ID:", entity.get_entity_id())
    print("Label:", entity.get_label())
    print("Description:", entity.get_description())
    print("Sitelinks:", entity.get_sitelinks())
    print("Claims:", entity.get_claims())
