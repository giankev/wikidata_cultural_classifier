from collections import Counter
import math
from difflib import SequenceMatcher

class DatasetParser:

  def __init__(self, item):
    '''
    Inizializzazione del costruttore tramite oggetto WikidataExtractor.
    '''
    self.entity_id = item.get_entity_id()
    self.sitelinks = item.get_sitelinks()
    self.claims = item.get_claims()
    self.label = item.get_label()
    self.description = item.get_description()

  def claims_target(self, properties_to_check: dict = {}) -> dict:
    """
    Restituisce i claims target che contengono le properties specificate.
    """
    if properties_to_check == {}:
        return {}

    target_claims = {}

    for property_id, data_value in self.claims.items():
        if property_id in properties_to_check:
            target_claims[property_id] = data_value

    return target_claims

  def get_claims_value(self, target_claims) -> dict:
    """
    Restituisce i claims con i relativi valori.
    ex. Country of Origin (P495): ['Q38']
    """

    if target_claims == {}:
        return {}

    claims_value = {}
    for property_id, data_value in target_claims.items():
        claims_value[property_id] = []
        for value in data_value:
            datavalue = value["mainsnak"]["datavalue"]
            if datavalue["type"] == "wikibase-entityid":
                claims_value[property_id].append(datavalue["value"]["id"])  # Estrae il QID (es. 'Q38' per Italy)
            elif datavalue["type"] == "string":
                claims_value[property_id].append(datavalue["value"])  # Estrae stringhe
            elif datavalue["type"] == "quantity":
                claims_value[property_id].append(datavalue["value"]["amount"])  # Estrae quantità numeriche

    return claims_value

  @staticmethod
  def get_similarity(s1, s2):
    """Calcola similarità di due stringhe usando SequenceMatcher (range 0-1)."""
    if not s1 or not s2:
        return 0.0

    # 2. Calcolo della similarità
    #    Crea un oggetto SequenceMatcher con le due stringhe.
    #    Il primo argomento 'None' indica di non ignorare nessun carattere "spazzatura".
    #    Chiama il metodo .ratio() per ottenere il punteggio di similarità.
    return SequenceMatcher(None, s1, s2).ratio()

  def sitelinks_translation_entropy(self, similarity_threshold: float = 0.9) -> float:
      """
      Restituisce l'entropia sulla distrubuzione dei titoli per comprendere se dato un item, corrispondono piu'
      traduzioni nelle diverse lingue.
      Un valore vicino a 0 indica alta dominanza di uno/pochi titoli (bassa diversità).
      Un valore alto indica alta diversità di titoli.
      ex. "pasta alla gricia" in molte lingue non ha nessuna traduzione (quindi avra' una bassa entropia), questo puo' significare un item cultural exclusive.
      """

      if self.sitelinks == {}:
        return None

      entropy = 0.0
      cleaned_titles = []
      for lang, value in self.sitelinks.items():
        if lang == "commonswiki" or not lang.endswith("wiki"):
          continue

        title = value.get("title", {})
        cleaned_titles.append(title.lower())

      if cleaned_titles == []:
        return entropy

      title_counts = Counter(cleaned_titles)
      total_titles = sum(title_counts.values())
      num_unique_titles = len(title_counts)
      if num_unique_titles <= 1:
        return entropy

      sorted_unique_titles = [title for title, count in title_counts.most_common()]
      final_grouped_counts = Counter()
      processed_mask = {title: False for title in sorted_unique_titles}

      for i, representative_norm in enumerate(sorted_unique_titles):
          if processed_mask[representative_norm]:
              continue

          processed_count = 0
          current_group_count = title_counts[representative_norm]
          processed_mask[representative_norm] = True
          processed_count += title_counts[representative_norm]

          for j in range(i + 1, len(sorted_unique_titles)):
                other_norm = sorted_unique_titles[j]
                if processed_mask[other_norm]:
                    continue

                similarity = self.get_similarity(representative_norm, other_norm)
                if similarity >= similarity_threshold:
                    current_group_count += title_counts[other_norm]
                    processed_mask[other_norm] = True
                    processed_count += title_counts[other_norm]

          final_grouped_counts[representative_norm] = current_group_count

      for title in final_grouped_counts:
        probability = final_grouped_counts[title] / total_titles
        entropy -= probability * math.log2(probability)

      #per normalizzare il valore tra 0 e 1
      #max_entropy = math.log2(num_unique_titles)
      #entropy_norm = entropy / max_entropy

      return entropy

  def get_number_sitelinks(self) -> int:
    """
    Restituisce il numero di sitelinks.
    """
    return len(self.sitelinks)

  def get_label(self) -> str:
    """
    Restituisce l'etichetta dell'entità
    """
    return self.label

  def get_description(self) -> str:
    """
    Restituisce la descrizione dell'entità
    """
    return self.description

  def get_number_claims(self) -> int:
    """
    Restituisce il numero di claim dell'entità.
    """
    return len(self.claims)

  def get_presence_of_P495(self) -> bool:
    """
    Restituisce True se l'entità ha il claim specificato, False altrimenti.
    """
    return 1 if ("P495" in self.claims.keys()) else 0

  def get_presence_of_P1343(self) -> bool:
    """
    Restituisce True se l'entità ha il claim specificato, False altrimenti.
    """
    return 1 if ("P1343" in self.claims.keys()) else 0

  def get_presence_of_P2596(self) -> bool:
    """
    Restituisce True se l'entità ha il claim specificato, False altrimenti.
    """
    return 1 if ("P2596" in self.claims.keys()) else 0

  def get_presence_of_P17(self) -> bool:
    """
    Restituisce True se l'entità ha il claim specificato, False altrimenti.
    """
    return 1 if ("P17" in self.claims.keys()) else 0

  def get_number_of_P31(self) -> int:
    """
    Restituisce il numero di claim specificato.
    """

    p31_statements = self.claims.get("P31", [])
    return len(p31_statements)

  def get_sum_of_cultural_claims(self) -> int:
    """
    Restituisce la somma dei claims culturali.
    """
    return sum(1 for claim in self.claims.keys() if claim in ["P495", "P17", "P2596", "P1435"])



if __name__ == '__main__':
    # Esempio di utilizzo
    # PASTA ALLA GRICIA -> cultural exclusive
    # wikidata_url = "https://www.wikidata.org/wiki/Q55641393"
    # PIZZA -> cultural representative
    wikidata_url = "https://www.wikidata.org/wiki/Q177"
    item = WikidataExtractor(wikidata_url)
    properties_to_check = {
        "P495": "Country of Origin",
        "P17": "Country",
        "P131": "Located in Adm. Entity",
        "P361": "Part Of",
        "P31": "Instance Of",
        "P279": "Subclass Of",
        "P443": "Pronuncation audio",
        "P1435": "Heritage designation",
        "P27": "Country of citizenship",
        "P1705": "Native Label",
        "P2012": "Cuisine",
        "P2596": "Culture",
        "P1343": "Described by source"
    }

    dataset_parser = DatasetParser(item)
    target_claims = dataset_parser.claims_target(properties_to_check)
    claims_value = dataset_parser.get_claims_value(target_claims)
    number_sitelinks = dataset_parser.get_number_sitelinks()
    sitelinks_translation_entropy = dataset_parser.sitelinks_translation_entropy()
    number_claims = dataset_parser.get_number_claims()
    presence_of_P495 = dataset_parser.get_presence_of_P495()
    presence_of_P1343 = dataset_parser.get_presence_of_P1343()
    presence_of_P2596 = dataset_parser.get_presence_of_P2596()
    presence_of_P17 = dataset_parser.get_presence_of_P17()
    number_of_P31 = dataset_parser.get_number_of_P31()
    sum_cultural_claims = dataset_parser.get_sum_of_cultural_claims()

    print("Entity ID:", dataset_parser.entity_id)
    print("Label:", dataset_parser.get_label())
    print("Description:", dataset_parser.get_description())
    print("Target claims:", target_claims)
    print("Target claims value:", claims_value)
    print("Number sitelinks:", number_sitelinks) #numero di pagine wikipedia nelle varie lingue che sono presenti
    print("Entropy of sitelinks:", sitelinks_translation_entropy)
    print("Numero di claims totali:", number_claims)
    print("Presence of P495:", presence_of_P495)
    print("Presence of P1343:", presence_of_P1343)
    print("Presence of P2596:", presence_of_P2596)
    print("Presence of P17:", presence_of_P17)
    print("Number of P31:", number_of_P31)
    print("sum of cultural claims:", sum_cultural_claims)
