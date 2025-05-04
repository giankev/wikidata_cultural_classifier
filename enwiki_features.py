# extract number of page characters, number of inline word links, and statistics about sitelinks

import requests
from bs4 import BeautifulSoup
from statistics import mean, median, pstdev
from wiki_extractor import WikidataExtractor
import math
from requests.utils import requote_uri
from requests.exceptions import HTTPError, RequestException

# ---------- MULTITHREADING & CSV ----------
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import pickle
import time

# ---------- CONFIG ----------

INPUT_CSV = os.path.expanduser('~/content/train.csv')
OUTPUT_CSV = os.path.expanduser('~/content/train_augmented.csv')

WD_CACHE_FILE = os.path.expanduser('~/content/wd_cache.pkl')
HTML_CACHE_FILE = os.path.expanduser('~/content/html_cache.pkl')

MAX_WORKERS = 10
SAVE_EVERY_N = 30
ENABLE_CACHING = False

# ---------- GLOBAL VARS ----------
EXPECTED_KEYS = [
    'title',
    'page_length',
    'num_links',
    'mean_sitelinks_count',
    'median_sitelinks_count',
    'std_sitelinks_count'
]

session = requests.Session()
wd_cache = {}
html_cache = {}

if os.path.exists(WD_CACHE_FILE):
    with open(WD_CACHE_FILE, 'rb') as f:
        wd_cache = pickle.load(f)

if os.path.exists(HTML_CACHE_FILE):
    with open(HTML_CACHE_FILE, 'rb') as f:
        html_cache = pickle.load(f)


# ---------- HELPERS ----------
def fill_missing_stats(stats):
    """
    Ensure that all EXPECTED_KEYS are present in stats.
    If a key is missing, fills it with 0 (or empty string for 'title').
    """
    return {k: stats.get(k, 0 if k != 'title' else '') for k in EXPECTED_KEYS}


def extract_linked_words(html: str) -> dict:
    """
    Parse the HTML of a Wikipedia page and return a dict mapping
    each unique link-title to its href found in <p> tags.
    """
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find('div', id='bodyContent')
    if not content_div:
        return {}

    links = {}
    paragraphs = content_div.find_all("p")
    for p in paragraphs:
        for a in p.find_all("a", href=True):
            title = a.get('title')
            if title and title not in links:
                links[title] = a['href']
    return links


def get_paragraph_len(html: str) -> int:
    """
    Return the total number of characters across all <p> tags
    inside the main content div; -1 if content div not found.
    """
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find('div', id='bodyContent')
    if not content_div:
        return -1
    return sum(len(p.text) for p in content_div.find_all("p"))


def retry_request(func, *args, tries=3, backoff=1, **kwargs):
    """
    Call func(*args, **kwargs). On HTTPError 429 or timeout, wait for `backoff` seconds,
    double backoff, retry up to `tries` times.
    """
    delay = backoff
    for attempt in range(1, tries+1):
        try:
            resp = func(*args, **kwargs)
            resp.raise_for_status()
            return resp
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 and attempt < tries:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except RequestException:  # network problem, timeout, etc.
            if attempt < tries:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def extract_linked_items_stats(links: dict) -> dict:
    """
    Given a dict of {title: href} from extract_linked_words,
    query each linked Wikipedia related wikidata item, fetches 
    its Wikidata sitelink count, and returns mean/median/std of those counts.
    """
    sitelinks_counts = []

    for title, href in links.items():
        if href.startswith('/wiki/') and ':' not in href:
            wiki_title = requests.utils.unquote(href.split('/wiki/')[-1])

            # account for redirects
            full_url = requote_uri("https://en.wikipedia.org" + href)
            resp = retry_request(requests.get, full_url, allow_redirects=True, timeout=5)
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

            params = {
                'action': 'wbgetentities',
                'sites': 'enwiki',
                'titles': wiki_title,
                'props': 'sitelinks',
                'format': 'json'
            }
            try:
                wdresp = session.get('https://www.wikidata.org/w/api.php', params=params, timeout=10)
                wdresp.raise_for_status()
                ent_data = wdresp.json().get('entities', {})
                if ent_data:
                    ent = next(iter(ent_data.values()))
                    sl = ent.get('sitelinks', {})
                    count_sl = sum(1 for k in sl.keys() if k.endswith('wiki'))
                    sitelinks_counts.append(count_sl)
            except Exception:
                continue

            time.sleep(0.2)

    return {
        'mean_sitelinks_count': mean(sitelinks_counts) if sitelinks_counts else 0,
        'median_sitelinks_count': median(sitelinks_counts) if sitelinks_counts else 0,
        'std_sitelinks_count': pstdev(sitelinks_counts) if len(sitelinks_counts) > 1 else 0,
    }


def process_item(identifier: str) -> dict:
    """
    For a single Wikidata Q‑ID:
    1. Look up its enwiki URL via WikidataExtractor.
    2. Fetch the page HTML.
    3. Extract <p>‑link stats via extract_linked_words.
    4. Compute page_length + num_links.
    5. Compute linked‐items sitelink statistics.
    Returns a dict with all EXPECTED_KEYS.
    """
    wd = WikidataExtractor(identifier)
    sitelinks = wd.get_sitelinks()
    enwiki = sitelinks.get('enwiki')
    if not enwiki:
        raise ValueError(f"No English Wikipedia sitelink for '{identifier}'")

    url = enwiki['url'].replace("?", "%3F") # ? char in links is encoded as %3F for some reason
    title = enwiki['title']

    try:
        resp = retry_request(session.get, url, timeout=10)
        html = resp.text
    except Exception as e:
        raise RuntimeError(f"Error fetching HTML from {url}: {e}")

    linked_words = extract_linked_words(html)
    links = linked_words if isinstance(linked_words, dict) else {}

    stats_page = {
        'title': title,
        'page_length': get_paragraph_len(html),
        'num_links': len(links),
    }

    items_stats = extract_linked_items_stats(links)

    return {**stats_page, **items_stats}


def _process_row(idx, item_url):
    """
    Worker wrapper for multiprocessing: given row index & Wikidata URL,
    calls process_item and fills missing stats on error.
    Returns (idx, stats_dict).
    """
    try:
        stats = process_item(item_url)
        stats = fill_missing_stats(stats)
    except Exception as e:
        print(f"\n[ERROR] Index {idx} ({item_url}): {e}")
        stats = fill_missing_stats({})
    return idx, stats


def save_caches():
    print("\n[CACHE] Saving cached files...")
    with open(WD_CACHE_FILE, 'wb') as f:
        pickle.dump(wd_cache, f)
    with open(HTML_CACHE_FILE, 'wb') as f:
        pickle.dump(html_cache, f)


def enwiki_augment(input_csv = INPUT_CSV, output_csv = OUTPUT_CSV, num_workers = MAX_WORKERS):
    df = pd.read_csv(input_csv).drop(columns=['description'], errors='ignore')
    total = len(df)
    results = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_row, idx, row['item']): idx
            for idx, row in df.iterrows()
        }

        for i, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            _, stats = future.result()
            results[idx] = stats

            sys.stdout.write(f"\rProcessed {i}/{total}")
            sys.stdout.flush()

            if ENABLE_CACHING and i % SAVE_EVERY_N == 0:
                save_caches()

    stats_df = pd.DataFrame.from_dict(results, orient='index')
    out = pd.concat([df, stats_df], axis=1)
    out.to_csv(output_csv, index=False)
    print(f"\n[INFO] Saved augmented CSV to {output_csv}")

    if ENABLE_CACHING:
        save_caches()

# ---------- main ----------
if __name__ == '__main__':
    enwiki_augment()