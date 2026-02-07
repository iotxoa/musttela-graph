"""
MUSTTELA - Script de Llenado Inicial
Ejecutar UNA VEZ para construir el corpus base de ~500 papers
"""

import json
import time
from datetime import datetime, timedelta
import arxiv
from semanticscholar import SemanticScholar

from config import *
from utils import *

# ==================== PROCESADORES ====================

def process_arxiv_paper(result, graph, added_date=None):
    """Procesa un paper de ArXiv y lo añade al grafo."""
    
    # ID del paper
    pid = result.entry_id.split('/')[-1].replace('v1', '').replace('v2', '')
    
    # Verificar calidad
    paper_data = {
        'title': result.title,
        'abstract': result.summary
    }
    if not passes_quality_filter(paper_data):
        return False
    
    # Metadata del nodo
    metadata = {
        "abstract": result.summary.replace("\n", " ")[:500],  # Limitar tamaño
        "url": result.pdf_url,
        "date": result.published.isoformat().split('T')[0],
        "source": "arxiv",
        "citation_count": 0  # ArXiv no tiene citation count
    }
    
    # Añadir fecha de agregado si es reciente
    if added_date:
        metadata["added_date"] = format_date_iso(added_date)
    
    # Añadir nodo
    if not add_node(graph, pid, result.title, "paper", 30, metadata):
        return False
    
    print(f"  [ArXiv] {truncate(result.title, 50)}")
    
    # Autores
    for author in result.authors[:5]:  # Limitar a 5 autores principales
        aid = f"auth_{clean_id(author.name)}"
        if add_node(graph, aid, author.name, "author", 15):
            add_link(graph, pid, aid, 5)
    
    # Categorías como topics
    for category in result.categories[:3]:  # Top 3 categorías
        tid = f"topic_{clean_id(category)}"
        if add_node(graph, tid, category, "topic", 10):
            add_link(graph, pid, tid, 2)
    
    return True

def process_s2_paper(paper, graph, is_seed=False, added_date=None):
    """Procesa un paper de Semantic Scholar y lo añade al grafo."""
    
    if not paper or not paper.paperId:
        return False
    
    pid = paper.paperId
    
    # Verificar calidad
    paper_data = {
        'title': paper.title,
        'abstract': paper.abstract or ""
    }
    if not passes_quality_filter(paper_data):
        return False
    
    # Metadata del nodo
    metadata = {
        "abstract": (paper.abstract or "Sin resumen")[:500],
        "url": paper.url or f"https://semanticscholar.org/paper/{pid}",
        "date": str(paper.year) + "-01-01" if paper.year else "2020-01-01",
        "source": "s2",
        "citation_count": paper.citationCount or 0
    }
    
    # Si es seed, marcarlo especialmente
    if is_seed:
        metadata.update({
            "is_seed": True,
            "seed_color": SEED_CONFIG["color"],
            "seed_badge": SEED_CONFIG["badge"]
        })
        val = SEED_CONFIG["size"]
    else:
        val = 30
    
    # Añadir fecha de agregado si es reciente
    if added_date:
        metadata["added_date"] = format_date_iso(added_date)
    
    # Añadir nodo
    if not add_node(graph, pid, paper.title, "paper", val, metadata):
        return False
    
    prefix = "  [SEED]" if is_seed else "  [S2]  "
    print(f"{prefix} {truncate(paper.title, 50)}")
    
    # Autores
    if paper.authors:
        for author in paper.authors[:5]:  # Top 5 autores
            aid = f"auth_{clean_id(author.name)}"
            if add_node(graph, aid, author.name, "author", 15):
                add_link(graph, pid, aid, 5)
    
    # Fields of Study como topics
    if paper.fieldsOfStudy:
        for field in paper.fieldsOfStudy[:3]:  # Top 3 fields
            tid = f"topic_{clean_id(field)}"
            if add_node(graph, tid, field, "topic", 10):
                add_link(graph, pid, tid, 2)
    
    return True

# ==================== BÚSQUEDA POR PERÍODO ====================

def search_arxiv_temporal(ax_client, year_start, year_end, max_results):
    """
    Busca papers de ArXiv en un rango temporal específico.
    """
    print(f"\n🔍 ArXiv {year_start}-{year_end} (max: {max_results})")
    
    all_keywords = []
    for category in KEYWORDS.values():
        all_keywords.extend(category)
    
    # Construir query temporal
    date_filter = f'submittedDate:[{year_start}0101 TO {year_end}1231]'
    
    # Query combinada con keywords principales
    core_keywords = ' OR '.join([f'"{kw}"' for kw in KEYWORDS['core_journalism'][:3]])
    query = f'{date_filter} AND ({core_keywords})'
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers_found = []
    try:
        for result in ax_client.results(search):
            papers_found.append(result)
            if len(papers_found) >= max_results:
                break
        print(f"  ✓ Encontrados: {len(papers_found)}")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
    
    return papers_found

def search_s2_temporal(s2_client, year_start, year_end, keywords, max_per_keyword, min_citations=0):
    """
    Busca papers de Semantic Scholar en un rango temporal.
    """
    print(f"\n🔍 S2 {year_start}-{year_end} (max: {max_per_keyword} por keyword)")
    
    all_papers = []
    rate_limiter = RateLimiter(calls_per_second=10)  # S2 permite ~100 req/5min
    
    for keyword in keywords[:5]:  # Limitar a 5 keywords principales
        try:
            rate_limiter.wait()
            
            # Buscar
            results = s2_client.search_paper(
                query=keyword,
                year=f"{year_start}-{year_end}",
                fields=S2_CONFIG["fields"],
                limit=max_per_keyword
            )
            
            # Filtrar por citas mínimas
            for paper in results:
                if paper.citationCount and paper.citationCount >= min_citations:
                    all_papers.append(paper)
            
            print(f"  '{keyword[:30]}': {len([p for p in results if p in all_papers])} válidos")
            
        except Exception as e:
            print(f"  ⚠️  Error con '{keyword}': {e}")
            continue
    
    print(f"  ✓ Total encontrados: {len(all_papers)}")
    return all_papers

# ==================== EXPANSIÓN DE SEEDS ====================

def expand_seed_references(s2_client, graph, seed_id, max_references=10):
    """
    Encuentra papers que citan a un seed y los añade al grafo.
    """
    print(f"\n📚 Expandiendo referencias de seed: {seed_id}")
    
    try:
        # Obtener papers que citan a este seed
        paper = s2_client.get_paper(seed_id, fields=['citations'])
        
        if not paper or not paper.citations:
            print("  No se encontraron citaciones")
            return 0
        
        count = 0
        for citation in paper.citations[:max_references]:
            if citation and citation.citingPaper:
                citing = citation.citingPaper
                
                # Solo añadir si es relevante por keywords
                title_abstract = f"{citing.title} {citing.abstract or ''}"
                all_keywords = []
                for kw_list in KEYWORDS.values():
                    all_keywords.extend(kw_list)
                
                if should_include_by_keywords(title_abstract, all_keywords[:10]):
                    if process_s2_paper(citing, graph):
                        # Enlazar con el seed
                        add_link(graph, citing.paperId, seed_id, 3)
                        count += 1
        
        print(f"  ✓ {count} referencias añadidas")
        return count
        
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return 0

# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("🚀 MUSTTELA - LLENADO INICIAL DEL GRAFO")
    print("="*60)
    print(f"Objetivo: ~500 papers bien seleccionados")
    print(f"Período: 2020-2026")
    print("="*60 + "\n")
    
    # Inicializar grafo vacío
    graph = {"nodes": [], "links": []}
    
    # Inicializar clientes
    s2 = SemanticScholar()
    ax = arxiv.Client()
    
    papers_added = []
    
    # ==================== 1. PROCESAR SEEDS ====================
    print("\n" + "─"*60)
    print("FASE 1: PROCESAR SEEDS FUNDACIONALES")
    print("─"*60)
    
    seeds = load_json(SEEDS_FILE, default=[])
    print(f"Seeds encontrados: {len(seeds)}")
    
    for seed_id in seeds:
        try:
            time.sleep(0.5)  # Rate limiting
            paper = s2.get_paper(seed_id, fields=S2_CONFIG["fields"])
            if process_s2_paper(paper, graph, is_seed=True):
                papers_added.append({
                    "name": paper.title,
                    "url": paper.url,
                    "type": "seed"
                })
        except Exception as e:
            print(f"  ⚠️  Error con seed {seed_id}: {e}")
    
    print(f"\n✅ Seeds procesados: {len([n for n in graph['nodes'] if n.get('is_seed')])}")
    
    # ==================== 2. EXPANDIR REFERENCIAS DE SEEDS ====================
    if SEED_CONFIG["expand_references"]:
        print("\n" + "─"*60)
        print("FASE 2: EXPANDIR REFERENCIAS DE SEEDS")
        print("─"*60)
        
        for seed_id in seeds[:5]:  # Solo primeros 5 seeds para no saturar
            refs_added = expand_seed_references(
                s2, graph, seed_id, 
                max_references=SEED_CONFIG["max_references_per_seed"]
            )
            time.sleep(1)
    
    # ==================== 3. BÚSQUEDA TEMPORAL ====================
    print("\n" + "─"*60)
    print("FASE 3: BÚSQUEDA TEMPORAL ESTRATÉGICA")
    print("─"*60)
    
    # Juntar todas las keywords
    all_keywords = []
    for kw_list in KEYWORDS.values():
        all_keywords.extend(kw_list)
    
    for period, config in TEMPORAL_STRATEGY.items():
        year_start, year_end = period.split('-') if '-' in period else (period, period)
        
        print(f"\n📅 Período: {period} - {config['description']}")
        
        # ArXiv
        arxiv_papers = search_arxiv_temporal(
            ax, year_start, year_end,
            max_results=config['max_results'] // 3  # 1/3 de ArXiv, 2/3 de S2
        )
        
        for paper in arxiv_papers:
            if process_arxiv_paper(paper, graph):
                papers_added.append({
                    "name": paper.title,
                    "url": paper.pdf_url,
                    "type": "arxiv"
                })
        
        time.sleep(2)
        
        # Semantic Scholar
        s2_papers = search_s2_temporal(
            s2, year_start, year_end,
            keywords=all_keywords,
            max_per_keyword=config['max_results'] // len(KEYWORDS),
            min_citations=config.get('min_citations', 0)
        )
        
        for paper in s2_papers:
            if process_s2_paper(paper, graph):
                papers_added.append({
                    "name": paper.title,
                    "url": paper.url or "#",
                    "type": "s2"
                })
        
        time.sleep(2)
    
    # ==================== 4. GUARDAR Y NOTIFICAR ====================
    print("\n" + "─"*60)
    print("FASE 4: GUARDANDO RESULTADOS")
    print("─"*60)
    
    save_graph(graph, JSON_FILE)
    print_graph_stats(graph)
    
    # Notificación Telegram
    msg = f"🎉 <b>MUSTTELA - Llenado Inicial Completado</b>\n\n"
    msg += f"📊 Papers añadidos: {len([n for n in graph['nodes'] if n['group'] == 'paper'])}\n"
    msg += f"   └─ Seeds: {len([n for n in graph['nodes'] if n.get('is_seed')])}\n"
    msg += f"   └─ ArXiv: {len([p for p in papers_added if p['type'] == 'arxiv'])}\n"
    msg += f"   └─ S2: {len([p for p in papers_added if p['type'] == 's2'])}\n\n"
    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    
    send_telegram(msg)
    
    print("\n" + "="*60)
    print("✅ LLENADO INICIAL COMPLETADO")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
