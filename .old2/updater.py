import json
import os
import arxiv
from keybert import KeyBERT

JSON_FILE = "docs/graph_data.json"

# 1. CONSTANTES DE BÚSQUEDA
QUERY = 'cat:cs.CY AND ("AI" OR "Journalism" OR "Media" OR "Ethics" OR "Communication")'

# 2. PAPERS HISTÓRICOS / IMPRESCINDIBLES (Añade aquí los IDs de ArXiv que quieras)
# Ejemplo: "1706.03762" es "Attention Is All You Need"
MANUAL_SEEDS = [
    "1706.03762", # Attention Is All You Need (Base de los Transformers)
    "2005.14165", # GPT-3 Paper
    "1911.01547", # The Bitter Lesson (Sutton) - Concepto clave
    "2201.11903", # Chain-of-Thought Prompting
]

def load_graph():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return {"nodes": [], "links": []}
    return {"nodes": [], "links": []}

def save_graph(data):
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_id(text):
    return text.lower().strip().replace(" ", "_").replace(".", "").replace("-", "_")

def process_paper(result, graph, existing_ids, kw_model):
    """Procesa un resultado de ArXiv y lo añade al grafo"""
    paper_id = result.entry_id.split('/')[-1]
    
    # Si ya existe, ignoramos (para no duplicar)
    if paper_id in existing_ids:
        return False

    print(f"> Procesando: {result.title[:40]}...")

    # NODO PAPER
    graph['nodes'].append({
        "id": paper_id,
        "name": result.title,
        "group": "paper",
        "val": 30,
        "abstract": result.summary.replace("\n", " "),
        "url": result.pdf_url,
        "date": result.published.isoformat()
    })
    existing_ids.add(paper_id)

    # AUTORES
    for author in result.authors:
        auth_name = author.name
        auth_id = f"auth_{clean_id(auth_name)}"
        
        if auth_id not in existing_ids:
            graph['nodes'].append({
                "id": auth_id,
                "name": auth_name,
                "group": "author",
                "val": 15
            })
            existing_ids.add(auth_id)
        
        # Enlace Paper -> Autor
        graph['links'].append({"source": paper_id, "target": auth_id, "value": 5})

    # TOPICS (Keywords)
    keywords = kw_model.extract_keywords(
        result.summary, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=6 # Aumentado a 6 para más conexiones
    )

    for kw, score in keywords:
        kw_clean = kw.lower().strip()
        kw_id = f"topic_{clean_id(kw_clean)}"

        if kw_id not in existing_ids:
            graph['nodes'].append({
                "id": kw_id,
                "name": kw_clean,
                "group": "topic",
                "val": 10
            })
            existing_ids.add(kw_id)

        # Enlace Paper -> Topic
        graph['links'].append({
            "source": paper_id,
            "target": kw_id,
            "value": 3
        })
    
    return True

def main():
    print("--- INICIANDO MUSTTELA V11 (HYBRID FETCH) ---")
    graph = load_graph()
    existing_ids = {n['id'] for n in graph['nodes']}
    
    print("Cargando modelo NLP...")
    kw_model = KeyBERT()
    client = arxiv.Client()

    new_count = 0

    # 1. PROCESAR MANUALES (HISTÓRICOS)
    if MANUAL_SEEDS:
        print(f"Verificando {len(MANUAL_SEEDS)} papers históricos...")
        search_manual = arxiv.Search(id_list=MANUAL_SEEDS)
        for result in client.results(search_manual):
            if process_paper(result, graph, existing_ids, kw_model):
                new_count += 1

    # 2. PROCESAR NOVEDADES (DIARIAS)
    print("Buscando novedades...")
    search_daily = arxiv.Search(
        query=QUERY,
        max_results=20, # Aumentado para densificar
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    for result in client.results(search_daily):
        if process_paper(result, graph, existing_ids, kw_model):
            new_count += 1

    # Guardar siempre para reordenar JSON si hace falta
    if new_count > 0:
        save_graph(graph)
        print(f"--- {new_count} nuevos items añadidos. ---")
    else:
        print("--- Todo actualizado. No hay items nuevos. ---")

if __name__ == "__main__":
    main()