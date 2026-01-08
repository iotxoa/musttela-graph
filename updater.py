import json
import os
import arxiv
from keybert import KeyBERT
from datetime import datetime

# --- CONFIGURACIÓN ---
# Archivo donde se guardará la base de datos visual
JSON_FILE = "docs/graph_data.json"

# Tu búsqueda personalizada.
# Busca en categorías de Computación y Sociedad (cs.CY) o IA (cs.AI)
QUERY = 'cat:cs.CY AND ("AI" OR "Journalism" OR "Communication" OR "Ethics" OR "Media")'

def load_graph():
    """Carga el JSON existente o crea uno nuevo si no existe."""
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"nodes": [], "links": []}
    return {"nodes": [], "links": []}

def save_graph(data):
    """Guarda los datos actualizados en el JSON."""
    # Asegurarnos de que la carpeta existe
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)
    
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("--- INICIANDO SISTEMA MUSTTELA ---")
    graph = load_graph()
    
    # Recopilar IDs de papers que ya tenemos para no duplicar
    existing_ids = {node['id'] for node in graph['nodes'] if node.get('group') == 'paper'}
    
    # Inicializar el modelo de Inteligencia Artificial (KeyBERT)
    print("Cargando modelo neuronal (esto puede tardar un poco)...")
    kw_model = KeyBERT()

    # Configurar cliente de ArXiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=10,  # Bajamos 10 papers por ejecución para probar
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    new_count = 0
    print(f"Buscando papers recientes sobre: {QUERY} ...")

    for result in client.results(search):
        # El ID suele ser una URL, nos quedamos con la última parte numérica
        paper_id = result.entry_id.split('/')[-1]
        
        # Si ya lo tenemos, saltamos
        if paper_id in existing_ids:
            continue

        print(f"> Procesando: {result.title[:60]}...")

        # 1. CREAR NODO DE PAPER
        paper_node = {
            "id": paper_id,
            "name": result.title,
            "group": "paper",
            "val": 20, # Tamaño del nodo
            "abstract": result.summary.replace("\n", " "),
            "url": result.pdf_url,
            "date": result.published.isoformat(),
            "authors": [a.name for a in result.authors]
        }
        graph['nodes'].append(paper_node)

        # 2. EXTRAER CONOCIMIENTO (TEMAS)
        # Usamos KeyBERT para leer el abstract y sacar 4 conceptos clave
        keywords = kw_model.extract_keywords(
            result.summary, 
            keyphrase_ngram_range=(1, 2), # Aceptamos 1 o 2 palabras (ej: "Ethics", "Social Media")
            stop_words='english', 
            top_n=4
        )

        for kw, score in keywords:
            kw_clean = kw.lower().strip()
            kw_id = f"topic_{kw_clean.replace(' ', '_')}"

            # Si el tema no existe en el grafo, lo creamos
            if not any(n['id'] == kw_id for n in graph['nodes']):
                graph['nodes'].append({
                    "id": kw_id,
                    "name": kw_clean,
                    "group": "topic",
                    "val": 8  # Los temas son nodos más pequeños
                })

            # Crear la conexión (Paper) --[trata sobre]--> (Tema)
            graph['links'].append({
                "source": paper_id,
                "target": kw_id,
                "value": round(score * 10, 2) # Grosor de la línea según relevancia
            })
        
        new_count += 1

    # Guardar resultados
    if new_count > 0:
        save_graph(graph)
        print(f"--- ÉXITO: Se han añadido {new_count} papers nuevos a la base de datos. ---")
    else:
        print("--- No se encontraron papers nuevos hoy. ---")

if __name__ == "__main__":
    main()
