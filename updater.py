import json
import os
import arxiv
from keybert import KeyBERT

# --- CONFIGURACIÓN ---
JSON_FILE = "docs/graph_data.json" # Asegúrate que apunta a docs/
QUERY = 'cat:cs.CY AND ("AI" OR "Journalism" OR "Media" OR "Ethics")'

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
    """Convierte texto en un ID válido (sin espacios, minúsculas)"""
    return text.lower().strip().replace(" ", "_").replace(".", "")

def main():
    print("--- INICIANDO MUSTTELA V3 (CONECTIVIDAD TOTAL) ---")
    graph = load_graph()
    
    # Cache de IDs para no duplicar
    existing_ids = {n['id'] for n in graph['nodes']}
    
    print("Cargando IA...")
    kw_model = KeyBERT()

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=10, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    new_count = 0
    print(f"Buscando papers nuevos...")

    for result in client.results(search):
        paper_id = result.entry_id.split('/')[-1]
        
        if paper_id in existing_ids:
            continue

        print(f"> Procesando: {result.title[:40]}...")

        # 1. NODO PAPER
        graph['nodes'].append({
            "id": paper_id,
            "name": result.title,
            "group": "paper",
            "val": 25,
            "abstract": result.summary.replace("\n", " "),
            "url": result.pdf_url,
            "date": result.published.isoformat(),
            "year": result.published.year
        })
        existing_ids.add(paper_id)

        # 2. CONEXIÓN POR AUTORES (Nodos Autor)
        for author in result.authors:
            auth_name = author.name
            auth_id = f"auth_{clean_id(auth_name)}"
            
            # Si el autor no existe, lo creamos
            if auth_id not in existing_ids:
                graph['nodes'].append({
                    "id": auth_id,
                    "name": auth_name,
                    "group": "author",
                    "val": 15
                })
                existing_ids.add(auth_id)
            
            # Link: Paper <--> Autor
            graph['links'].append({
                "source": paper_id, 
                "target": auth_id,
                "value": 3
            })

        # 3. CONEXIÓN POR TEMAS (KeyBERT)
        keywords = kw_model.extract_keywords(
            result.summary, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=4
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

            # Link: Paper <--> Tema
            graph['links'].append({
                "source": paper_id,
                "target": kw_id,
                "value": round(score * 10, 2)
            })
        
        new_count += 1

    if new_count > 0:
        save_graph(graph)
        print(f"--- ¡Hecho! {new_count} papers integrados en la red. ---")
    else:
        print("--- Sin novedades hoy. ---")

if __name__ == "__main__":
    main()
