import json
import os
import arxiv
from keybert import KeyBERT

JSON_FILE = "docs/graph_data.json"
# Nota: He añadido "Journalism" y "Communication" explícitamente para tus temas
QUERY = 'cat:cs.CY AND ("AI" OR "Journalism" OR "Media" OR "Ethics" OR "Communication")'

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

def main():
    print("--- INICIANDO MUSTTELA V4 (CLUSTERING) ---")
    graph = load_graph()
    existing_ids = {n['id'] for n in graph['nodes']}
    
    print("Cargando modelo NLP...")
    kw_model = KeyBERT() # Modelo ligero

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=12, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    new_count = 0
    print("Buscando papers...")

    for result in client.results(search):
        paper_id = result.entry_id.split('/')[-1]
        
        if paper_id in existing_ids:
            continue

        print(f"> {result.title[:40]}...")

        # NODO PAPER
        graph['nodes'].append({
            "id": paper_id,
            "name": result.title,
            "group": "paper",
            "val": 30, # Nodos más grandes para clic fácil
            "abstract": result.summary.replace("\n", " "),
            "url": result.pdf_url,
            "date": result.published.isoformat()
        })
        existing_ids.add(paper_id)

        # AUTORES (Nodos dorados)
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
            
            # Enlace Fuerte Paper-Autor
            graph['links'].append({"source": paper_id, "target": auth_id, "value": 5})

        # TEMAS (Nodos grises - El pegamento del grafo)
        # Extraemos 5 keywords para aumentar probabilidad de coincidencia
        keywords = kw_model.extract_keywords(
            result.summary, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=5
        )

        for kw, score in keywords:
            kw_clean = kw.lower().strip()
            # Unificación simple (ej: "ai ethics" y "ethics of ai" -> se intentan juntar)
            kw_id = f"topic_{clean_id(kw_clean)}"

            if kw_id not in existing_ids:
                graph['nodes'].append({
                    "id": kw_id,
                    "name": kw_clean,
                    "group": "topic",
                    "val": 10
                })
                existing_ids.add(kw_id)

            # Enlace Paper-Tema
            graph['links'].append({
                "source": paper_id,
                "target": kw_id,
                "value": 2 # Valor menor para que no se peguen tanto como autor-paper
            })
        
        new_count += 1

    if new_count > 0:
        save_graph(graph)
        print(f"--- {new_count} nuevos papers añadidos. ---")
    else:
        print("--- No hay papers nuevos. ---")

if __name__ == "__main__":
    main()