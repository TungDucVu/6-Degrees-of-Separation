import wikipediaapi
import spacy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Wikipedia setup
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MyProjectName (merlin@example.com)',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# --- caching ---
page_cache = {}
names_cache = {}
cache_lock = threading.Lock()

def get_wiki(name):
    with cache_lock:
        if name in page_cache:
            return page_cache[name]
    page = wiki_wiki.page(name)
    with cache_lock:
        page_cache[name] = page
    return page

def get_links(page):
    if not page.exists():
        return set()
    return set(page.links.keys())

# Only CPU-bound SpaCy part (run sequentially)
def get_names(available_links):
    persons = []
    for link in available_links:
        if " " not in link:  # simple heuristic to skip non-names
            continue
        doc = nlp(link)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append(ent.text)
    return persons

# Fetch names from page (threaded for I/O)
# def fetch_names(title):
#     with cache_lock:
#         if title in names_cache:
#             return names_cache[title]
#     page = get_wiki(title)  # network I/O
#     links = get_links(page)
#     names = get_names(links)  # CPU-bound
#     with cache_lock:
#         names_cache[title] = names
#     return names

# --- bidirectional BFS with persistent threads ---
def bidirectional_bfs_threaded(source, target, max_depth=6, max_workers=10):
    frontier_src = deque([(source, [source])])
    frontier_tgt = deque([(target, [target])])
    visited_src = {source: [source]}
    visited_tgt = {target: [target]}
    found_path = []

    lock = threading.Lock()
    done = threading.Event()

    # Persistent executor
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def expand_frontier(frontier, visited_self, visited_other, forward=True):
        nonlocal found_path
        while frontier and not done.is_set():
            batch = []
            for _ in range(min(len(frontier), max_workers)):
                batch.append(frontier.popleft())

            futures = {executor.submit(get_wiki, current): (current, path) for current, path in batch}

            for future in as_completed(futures):
                current, path = futures[future]
                if done.is_set():
                    return
                try:
                    page = future.result()
                except Exception:
                    continue

                neighbors = get_names(get_links(page))  # sequential SpaCy
                for neighbor in neighbors:
                    if neighbor not in visited_self:
                        new_path = path + [neighbor]
                        visited_self[neighbor] = new_path
                        if neighbor in visited_other:
                            with lock:
                                if not found_path:
                                    if forward:
                                        found_path = new_path + visited_other[neighbor][::-1][1:]
                                    else:
                                        found_path = visited_other[neighbor] + new_path[::-1][1:]
                                    done.set()
                        frontier.append((neighbor, new_path))

    # Start threads for both directions
    t1 = threading.Thread(target=expand_frontier, args=(frontier_src, visited_src, visited_tgt, True))
    t2 = threading.Thread(target=expand_frontier, args=(frontier_tgt, visited_tgt, visited_src, False))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    executor.shutdown(wait=True)
    return found_path if found_path else None

# Example run
source = "Donald Trump"
target = "Albert Einstein"

start = time.perf_counter()
path = bidirectional_bfs_threaded(source, target, max_workers=10)
end = time.perf_counter()

print(path)
print(f"Run time: {end-start:.2f} seconds")
