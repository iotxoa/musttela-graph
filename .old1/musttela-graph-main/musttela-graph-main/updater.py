import json
import os
import time
import requests
import arxiv
from semanticscholar import SemanticScholar

# --- CONFIGURACIÓN ---
JSON_FILE = "docs/graph_data.json"
SEEDS_FILE = "seeds.json"
ARXIV_QUERY = 'cat:cs.CY AND ("AI" OR "Journalism" OR "Media" OR "Ethics" OR "Communication")'
S2_KEYWORDS = ["Algorithmic Journalism", "AI Media Ethics", "News Automation"]

# --- TELEGRAM CONFIG ---
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except: return [] if filepath == SEEDS_FILE else {"nodes": [], "links": []}
    return [] if filepath == SEEDS_FILE else {"nodes": [], "links": []}

def save_graph(data):
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_id(text):
    if not text: return "unknown"
    return "".join(x for x in text if x.isalnum()).lower()

def add_node(graph, id, name, group, val=10, meta={}):
    for n in graph['nodes']:
        if n['id'] == id: return False
    graph['nodes'].append({"id": id, "name": name, "group": group, "val": val, **meta})
    return True

def add_link(graph, source, target, value=1):
    for l in graph['links']:
        if (l['source'] == source and l['target'] == target) or \
           (l['source'] == target and l['target'] == source):
            return
    graph['links'].append({"source": source, "target": target, "value": value})

def send_telegram(new_items):
    if not TG_TOKEN or not TG_CHAT_ID or not new_items: return
    
    msg = f"🚨 <b>MUSTTELA:</b> {len(new_items)} Novedades\n\n"
    for item in new_items[:3]:
        msg += f"📄 {item['name']}\n🔗 {item['url']}\n\n"
    if len(new_items) > 3: msg += f"<i>...y {len(new_items)-3} más.</i>"
    
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
                      json={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'})
    except: pass

# --- PROCESADORES (SIN KEYBERT) ---

def process_arxiv(result, graph, new_list):
    pid = result.entry_id.split('/')[-1]
    if not add_node(graph, pid, result.title, "paper", 30, {
        "abstract": result.summary.replace("\n", " "), "url": result.pdf_url, "date": result.published.isoformat().split('T')[0]
    }): return

    print(f"[ArXiv] {result.title[:30]}...")
    new_list.append({"name": result.title, "url": result.pdf_url})

    for a in result.authors:
        aid = f"auth_{clean_id(a.name)}"
        add_node(graph, aid, a.name, "author", 15)
        add_link(graph, pid, aid, 5)

    # Usar categorías de ArXiv como temas
    for cat in result.categories:
        tid = f"topic_{clean_id(cat)}"
        add_node(graph, tid, cat, "topic", 10)
        add_link(graph, pid, tid, 2)

def process_s2(paper, graph, new_list, is_seed=False):
    if not paper or not paper.paperId: return
    pid = paper.paperId
    
    if not add_node(graph, pid, paper.title, "paper", 40 if is_seed else 30, {
        "abstract": paper.abstract or "Sin resumen", 
        "url": paper.url or f"https://semanticscholar.org/paper/{pid}",
        "date": str(paper.year) + "-01-01" if paper.year else "2024-01-01"
    }): return

    print(f"[S2] {paper.title[:30]}...")
    new_list.append({"name": paper.title, "url": paper.url})

    if paper.authors:
        for a in paper.authors:
            aid = f"auth_{clean_id(a.name)}"
            add_node(graph, aid, a.name, "author", 15)
            add_link(graph, pid, aid, 5)

    # Usar Fields of Study como temas
    if paper.fieldsOfStudy:
        for field in paper.fieldsOfStudy:
            tid = f"topic_{clean_id(field)}"
            add_node(graph, tid, field, "topic", 10)
            add_link(graph, pid, tid, 2)

def main():
    print("--- MUSTTELA V14 LITE ---")
    graph = load_json(JSON_FILE)
    if not isinstance(graph, dict): graph = {"nodes": [], "links": []}
    
    new_items = []
    s2 = SemanticScholar()
    ax = arxiv.Client()

    # 1. SEEDS
    seeds = load_json(SEEDS_FILE)
    for sid in seeds:
        try: process_s2(s2.get_paper(sid), graph, new_items, True)
        except: pass
        time.sleep(0.5)

    # 2. ARXIV
    search = arxiv.Search(query=ARXIV_QUERY, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
    for r in ax.results(search): process_arxiv(r, graph, new_items)

    # 3. S2
    for q in S2_KEYWORDS:
        try:
            for p in s2.search_paper(q, limit=5): process_s2(p, graph, new_items)
        except: pass

    if new_items:
        save_graph(graph)
        send_telegram(new_items)
        print(f"Hecho. {len(new_items)} nuevos.")
    else:
        print("Sin novedades.")

if __name__ == "__main__":
    main()
