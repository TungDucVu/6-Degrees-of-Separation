import wikipediaapi
import spacy
from collections import deque
import threading
import time

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Wiki setup
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MyProjectName (merlin@example.com)',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# --- caching ---
page_cache = {}
names_cache = {}

def get_wiki(name):
    if name in page_cache:
        return page_cache[name]
    page = wiki_wiki.page(name)
    page_cache[name] = page
    return page

def get_links(page):
    if not page.exists():
        return set()
    return set(page.links.keys())

def get_names(available_links):
    if not available_links:
        return []
    persons = []
    for link in available_links:
        if "_" not in link and " " not in link:  # quick skip
            continue
        doc = nlp(link)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append(ent.text)
    return persons

def get_names_from_page(title):
    if title in names_cache:
        return names_cache[title]
    page = get_wiki(title)
    links = get_links(page)
    names = get_names(links)
    names_cache[title] = names
    return names

# --- bidirectional BFS with threads ---
def bidirectional_bfs(source, target, max_depth=6):
    frontier_src = deque([(source, [source])])
    frontier_tgt = deque([(target, [target])])
    visited_src = {source: [source]}
    visited_tgt = {target: [target]}
    found_path = []

    lock = threading.Lock()
    done = threading.Event()

    def expand_frontier(frontier, visited_self, visited_other, forward=True):
        nonlocal found_path
        while frontier and not done.is_set():
            current, path = frontier.popleft()
            if len(path) > max_depth:
                continue
            neighbors = get_names_from_page(current)
            for neighbor in neighbors:
                if neighbor not in visited_self:
                    new_path = path + [neighbor]
                    visited_self[neighbor] = new_path
                    # check if met the other side
                    if neighbor in visited_other:
                        lock.acquire()
                        try:
                            if not found_path:
                                if forward:
                                    found_path = new_path + visited_other[neighbor][::-1][1:]
                                else:
                                    found_path = visited_other[neighbor] + new_path[::-1][1:]
                                done.set()
                        finally:
                            lock.release()
                    frontier.append((neighbor, new_path))

    t1 = threading.Thread(target=expand_frontier, args=(frontier_src, visited_src, visited_tgt, True))
    t2 = threading.Thread(target=expand_frontier, args=(frontier_tgt, visited_tgt, visited_src, False))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return found_path if found_path else None

# Example run
source = "Donald Trump"
target = "Albert Einstein"

# record runtime + run algo
start = time.perf_counter()
path = bidirectional_bfs(source, target)
end = time.perf_counter()


print(path)
print(f"Run time: {end-start:.6f} seconds")
