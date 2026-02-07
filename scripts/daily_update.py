"""
MUSTTELA - Script de Actualización Diaria
Ejecutar automáticamente cada día para añadir papers recientes
"""

import json
import time
from datetime import datetime, timedelta
import arxiv
from semanticscholar import SemanticScholar

from config import *
from utils import *

# ==================== PROCESADORES CON FECHA ====================

def process_recent_arxiv(result, graph):
    """Procesa un paper reciente de ArXiv y lo marca como nuevo."""
    
    pid = result.entry_id.split('/')[-1].replace('v1', '').replace('v2', '')
    
    # Verificar calidad
    paper_data = {
        'title': result.title,
        'abstract': result.summary
    }
    if not passes_quality_filter(paper_data):
        return False
    
    # Verificar que sea realmente reciente
    days_old = (datetime.now() - result.published).days
    if days_old > DAILY_UPDATE["lookback_days"]:
        return False
    
    # Metadata con marcador de "nuevo"
    metadata = {
        "abstract": result.summary.replace("\n", " ")[:500],
        "url": result.pdf_url,
        "date": result.published.isoformat().split('T')[0],
        "source": "arxiv",
        "citation_count": 0,
        "added_date": format_date_iso(datetime.now()),  # IMPORTANTE: Marca cuándo se añadió
        "is_new": True
    }
    
    # Añadir nodo
    if not add_node(graph, pid, result.title, "paper", 30, metadata):
        return False
    
    print(f"  🆕 [ArXiv] {truncate(result.title, 50)}")
    
    # Autores
    for author in result.authors[:5]:
        aid = f"auth_{clean_id(author.name)}"
        if add_node(graph, aid, author.name, "author", 15):
            add_link(graph, pid, aid, 5)
    
    # Categorías
    for category in result.categories[:3]:
        tid = f"topic_{clean_id(category)}"
        if add_node(graph, tid, category, "topic", 10):
            add_link(graph, pid, tid, 2)
    
    return True

def process_recent_s2(paper, graph):
    """Procesa un paper reciente de S2 y lo marca como nuevo."""
    
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
    
    # Metadata con marcador de "nuevo"
    metadata = {
        "abstract": (paper.abstract or "Sin resumen")[:500],
        "url": paper.url or f"https://semanticscholar.org/paper/{pid}",
        "date": str(paper.year) + "-01-01" if paper.year else datetime.now().strftime("%Y-%m-%d"),
        "source": "s2",
        "citation_count": paper.citationCount or 0,
        "added_date": format_date_iso(datetime.now()),  # IMPORTANTE
        "is_new": True
    }
    
    # Añadir nodo
    if not add_node(graph, pid, paper.title, "paper", 30, metadata):
        return False
    
    print(f"  🆕 [S2] {truncate(paper.title, 50)}")
    
    # Autores
    if paper.authors:
        for author in paper.authors[:5]:
            aid = f"auth_{clean_id(author.name)}"
            if add_node(graph, aid, author.name, "author", 15):
                add_link(graph, pid, aid, 5)
    
    # Fields
    if paper.fieldsOfStudy:
        for field in paper.fieldsOfStudy[:3]:
            tid = f"topic_{clean_id(field)}"
            if add_node(graph, tid, field, "topic", 10):
                add_link(graph, pid, tid, 2)
    
    return True

# ==================== LIMPIEZA DE FLAGS "NUEVO" ====================

def cleanup_old_new_flags(graph):
    """
    Limpia los flags 'is_new' de papers que ya tienen más de X días.
    Actualiza también los metadatos de 'newness_level'.
    """
    print("\n🧹 Limpiando flags antiguos...")
    
    updated_count = 0
    threshold_days = DAILY_UPDATE["new_paper_threshold_days"]
    
    for node in graph['nodes']:
        if node['group'] != 'paper':
            continue
        
        added_date = node.get('added_date')
        if not added_date:
            continue
        
        # Calcular días desde que se añadió
        date_obj = parse_date(added_date)
        days_old = (datetime.now() - date_obj).days
        
        # Determinar nivel de novedad
        newness = get_newness_level(added_date)
        
        if newness:
            # Actualizar nivel de novedad
            node['newness_level'] = newness
            level_config = NEWNESS_LEVELS[newness]
            node['newness_badge'] = level_config['badge']
            node['newness_color'] = level_config['color']
        else:
            # Ya no es nuevo, limpiar flags
            if 'is_new' in node:
                del node['is_new']
            if 'newness_level' in node:
                del node['newness_level']
            if 'newness_badge' in node:
                del node['newness_badge']
            if 'newness_color' in node:
                del node['newness_color']
            updated_count += 1
    
    print(f"  ✓ {updated_count} papers ya no son 'nuevos'")
    
    # Contar cuántos hay en cada nivel
    today_count = sum(1 for n in graph['nodes'] if n.get('newness_level') == 'today')
    new_count = sum(1 for n in graph['nodes'] if n.get('newness_level') == 'new')
    recent_count = sum(1 for n in graph['nodes'] if n.get('newness_level') == 'recent')
    
    if today_count or new_count or recent_count:
        print(f"  🔥 HOY: {today_count}")
        print(f"  ✨ NUEVO (1-3 días): {new_count}")
        print(f"  📌 RECIENTE (3-7 días): {recent_count}")

# ==================== BÚSQUEDA DE PAPERS RECIENTES ====================

def search_recent_arxiv(ax_client):
    """Busca papers muy recientes en ArXiv."""
    
    print("\n🔍 Buscando papers recientes en ArXiv...")
    
    # Calcular fecha límite
    lookback = datetime.now() - timedelta(days=DAILY_UPDATE["lookback_days"])
    date_str = lookback.strftime('%Y%m%d')
    
    # Keywords principales
    core_kw = ' OR '.join([f'"{kw}"' for kw in KEYWORDS['core_journalism']])
    
    # Query temporal
    query = f'submittedDate:[{date_str} TO 99991231] AND ({core_kw})'
    
    search = arxiv.Search(
        query=query,
        max_results=DAILY_UPDATE["max_new_papers"] // 2,  # Mitad ArXiv, mitad S2
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    try:
        for result in ax_client.results(search):
            papers.append(result)
        print(f"  ✓ {len(papers)} papers encontrados")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
    
    return papers

def search_recent_s2(s2_client):
    """Busca papers muy recientes en Semantic Scholar."""
    
    print("\n🔍 Buscando papers recientes en S2...")
    
    all_papers = []
    rate_limiter = RateLimiter(calls_per_second=10)
    
    # Usar solo keywords más específicas
    top_keywords = KEYWORDS['core_journalism'] + KEYWORDS['emerging']
    
    for keyword in top_keywords[:6]:  # Top 6 keywords
        try:
            rate_limiter.wait()
            
            # Buscar solo del año actual
            current_year = datetime.now().year
            results = s2_client.search_paper(
                query=keyword,
                year=str(current_year),
                fields=S2_CONFIG["fields"],
                limit=5  # Solo 5 por keyword
            )
            
            for paper in results:
                # Verificar que sea realmente reciente
                if paper.publicationDate:
                    pub_date = parse_date(paper.publicationDate)
                    days_old = (datetime.now() - pub_date).days
                    
                    if days_old <= DAILY_UPDATE["lookback_days"]:
                        all_papers.append(paper)
            
        except Exception as e:
            print(f"  ⚠️  Error con '{keyword}': {e}")
            continue
    
    print(f"  ✓ {len(all_papers)} papers encontrados")
    return all_papers

# ==================== MAIN ====================

def main():
    print("\n" + "="*60)
    print("🔄 MUSTTELA - ACTUALIZACIÓN DIARIA")
    print("="*60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Lookback: {DAILY_UPDATE['lookback_days']} días")
    print("="*60 + "\n")
    
    # Cargar grafo existente
    graph = load_json(JSON_FILE)
    
    if not graph or not graph.get('nodes'):
        print("⚠️  No se encontró grafo existente. Ejecuta primero initial_build.py")
        return
    
    print(f"📊 Grafo actual: {len(graph['nodes'])} nodos")
    
    # Limpiar flags antiguos
    cleanup_old_new_flags(graph)
    
    # Inicializar clientes
    s2 = SemanticScholar()
    ax = arxiv.Client()
    
    new_papers = []
    
    # ==================== 1. ARXIV RECIENTES ====================
    print("\n" + "─"*60)
    print("FASE 1: PAPERS RECIENTES DE ARXIV")
    print("─"*60)
    
    arxiv_papers = search_recent_arxiv(ax)
    
    for paper in arxiv_papers:
        if process_recent_arxiv(paper, graph):
            new_papers.append({
                "name": paper.title,
                "url": paper.pdf_url,
                "source": "arxiv",
                "date": paper.published.isoformat().split('T')[0]
            })
    
    print(f"✅ {len([p for p in new_papers if p['source'] == 'arxiv'])} papers de ArXiv añadidos")
    
    # ==================== 2. S2 RECIENTES ====================
    print("\n" + "─"*60)
    print("FASE 2: PAPERS RECIENTES DE SEMANTIC SCHOLAR")
    print("─"*60)
    
    s2_papers = search_recent_s2(s2)
    
    for paper in s2_papers:
        if process_recent_s2(paper, graph):
            new_papers.append({
                "name": paper.title,
                "url": paper.url or "#",
                "source": "s2",
                "date": str(paper.year) if paper.year else "2026"
            })
    
    print(f"✅ {len([p for p in new_papers if p['source'] == 's2'])} papers de S2 añadidos")
    
    # ==================== 3. GUARDAR Y NOTIFICAR ====================
    print("\n" + "─"*60)
    print("FASE 3: GUARDANDO CAMBIOS")
    print("─"*60)
    
    if new_papers:
        save_graph(graph, JSON_FILE)
        print_graph_stats(graph)
        
        # Notificación Telegram
        msg = f"🔄 <b>MUSTTELA - Actualización Diaria</b>\n\n"
        msg += f"🆕 {len(new_papers)} nuevos papers añadidos\n\n"
        
        # Top 3 papers
        for i, paper in enumerate(new_papers[:3], 1):
            title = truncate(paper['name'], 50)
            url = paper['url']
            source = "📄 ArXiv" if paper['source'] == 'arxiv' else "📚 S2"
            msg += f"{i}. {title}\n{source} - {paper['date']}\n🔗 {url}\n\n"
        
        if len(new_papers) > 3:
            msg += f"<i>...y {len(new_papers) - 3} más.</i>\n\n"
        
        msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        
        send_telegram(msg)
        
        print(f"\n✅ Actualización completada: {len(new_papers)} papers nuevos")
    else:
        print("\nℹ️  No se encontraron papers nuevos hoy")
    
    print("\n" + "="*60)
    print("✅ ACTUALIZACIÓN DIARIA COMPLETADA")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
