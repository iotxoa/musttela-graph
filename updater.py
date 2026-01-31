import json
import os
import time
import arxiv
from semanticscholar import SemanticScholar
from keybert import KeyBERT

# --- CONFIGURACIÓN ---
JSON_FILE = "docs/graph_data.json"
SEEDS_FILE = "seeds.json"

# Tu Query Original de ArXiv (INTACTA)
ARXIV_QUERY = 'cat:cs.CY AND ("AI" OR "Journalism" OR "Media" OR "Ethics" OR "Communication")'

# Nuevas Keywords para buscar en Semantic Scholar (Journals)
S2_KEYWORDS = ["Algorithmic Journalism", "AI Media Ethics", "News Automation"]

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return [] if filepath == SEEDS_FILE else {"nodes": [], "links": []}
    return [] if filepath == SEEDS_FILE else {"nodes": [], "links": []}

def save_graph(data):
    # Asegurar que docs existe
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_id(text):
    """Limpia strings para usarlos como IDs"""
    if not text: return "unknown"
    return "".join(x for x in text if x.isalnum()).lower()

# --- GESTOR DE GRAFO (Anti-Duplicados) ---
def add_node(graph, id, name, group, val=10, meta={}):
    # Comprobar si ya existe
    for n in graph['nodes']:
        if n['id'] == id:
            return False # Ya existe
    
    node = {
        "id": id,
        "name": name,
        "group": group,
        "val": val,
        **meta
    }
    graph['nodes'].append(node)
    return True

def add_link(graph, source, target, value=1):
    # Comprobar si el link ya existe (en cualquier dirección)
    for l in graph['links']:
        if (l['source'] == source and l['target'] == target) or \
           (l['source'] == target and l['target'] == source):
            return
    graph['links'].append({"source": source, "target": target, "value": value})

# --- PROCESADORES ---

def process_arxiv_result(result, graph, kw_model):
    """Logica original para ArXiv"""
    paper_id = result.entry_id.split('/')[-1]
    
    # Intentamos añadir. Si devuelve False, es que ya existía.
    added = add_node(graph, paper_id, result.title, "paper", 30, {
        "abstract": result.summary.replace("\n", " "),
        "url": result.pdf_url,
        "date": result.published.isoformat()
    })
    
    if not added: return False # Saltamos si ya lo teníamos

    print(f"[ArXiv] Nuevo: {result.title[:40]}...")

    # Autores
    for author in result.authors:
        auth_id = f"auth_{clean_id(author.name)}"
        add_node(graph, auth_id, author.name, "author", 15)
        add_link(graph, paper_id, auth_id, 5)

    # Keywords (KeyBERT)
    keywords = kw_model.extract_keywords(
        result.summary, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=5
    )
    for kw, score in keywords:
        kw_clean = kw.lower().strip()
        kw_id = f"topic_{clean_id(kw_clean)}"
        add_node(graph, kw_id, kw_clean, "topic", 10)
        add_link(graph, paper_id, kw_id, 2)
    
    return True

def process_s2_result(paper, graph, kw_model, is_seed=False):
    """Nueva lógica para Semantic Scholar"""
    if not paper or not paper.paperId: return False

    p_id = paper.paperId
    # Si es seed, le damos prioridad visual (opcional)
    val = 40 if is_seed else 30
    
    abstract = paper.abstract if paper.abstract else "Sin resumen."
    
    added = add_node(graph, p_id, paper.title, "paper", val, {
        "abstract": abstract,
        "url": paper.url if paper.url else f"https://www.semanticscholar.org/paper/{p_id}",
        "date": str(paper.year) + "-01-01" if paper.year else "2024-01-01"
    })

    if not added: return False

    source_tag = "[SEED]" if is_seed else "[S2]"
    print(f"{source_tag} Procesado: {paper.title[:40]}...")

    # Autores
    if paper.authors:
        for auth in paper.authors:
            if not auth.name: continue
            auth_id = f"auth_{clean_id(auth.name)}"
            add_node(graph, auth_id, auth.name, "author", 15)
            add_link(graph, p_id, auth_id, 5)

    # Temas (Híbrido: API + KeyBERT)
    tags = []
    # 1. Intentar usar categorías oficiales de S2
    if paper.fieldsOfStudy:
        tags += paper.fieldsOfStudy[:3]
    
    # 2. Rellenar con IA si faltan
    if len(tags) < 4 and abstract:
        kws = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
        tags += [k[0] for k in kws]

    for tag in tags:
        tag_clean = tag.lower().strip()
        tag_id = f"topic_{clean_id(tag_clean)}"
        add_node(graph, tag_id, tag_clean, "topic", 10)
        add_link(graph, p_id, tag_id, 2)

    return True

# --- MAIN ---
def main():
    print("--- MUSTTELA ENGINE V2 (HYBRID) ---")
    
    # 1. Cargar datos existentes
    graph = load_json(JSON_FILE)
    if not isinstance(graph, dict): graph = {"nodes": [], "links": []}
    
    # 2. Inicializar IAs y Clientes
    print("Cargando KeyBERT y Clientes API...")
    kw_model = KeyBERT()
    s2_client = SemanticScholar()
    ax_client = arxiv.Client()
    
    new_count = 0

    # 3. PROCESAR SEEDS (Históricos Manuales)
    seeds = load_json(SEEDS_FILE) # Lee seeds.json de la raíz
    if seeds:
        print(f"Verificando {len(seeds)} papers históricos...")
        for seed_id in seeds:
            try:
                # Buscamos por DOI o ArXiv ID en S2
                paper = s2_client.get_paper(seed_id)
                if process_s2_result(paper, graph, kw_model, is_seed=True):
                    new_count += 1
                time.sleep(0.5) # Respetar API limits
            except Exception as e:
                print(f"Error buscando seed {seed_id}: {e}")

    # 4. PROCESAR NOVEDADES ARXIV (Tu query original)
    print("Buscando en ArXiv...")
    search = arxiv.Search(
        query=ARXIV_QUERY,
        max_results=12,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    for result in ax_client.results(search):
        if process_arxiv_result(result, graph, kw_model):
            new_count += 1

    # 5. PROCESAR NOVEDADES S2 (Complementario)
    print("Buscando en Semantic Scholar...")
    for q in S2_KEYWORDS:
        try:
            results = s2_client.search_paper(q, limit=5)
            for item in results:
                if process_s2_result(item, graph, kw_model):
                    new_count += 1
        except:
            pass

    # 6. GUARDAR
    if new_count > 0:
        save_graph(graph)
        print(f"--- ÉXITO: {new_count} nuevos items añadidos. ---")
    else:
        print("--- Todo al día. No hay items nuevos. ---")

if __name__ == "__main__":
    main()
