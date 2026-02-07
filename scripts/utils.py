"""
MUSTTELA - Funciones de utilidad compartidas
"""

import json
import os
import time
from datetime import datetime, timedelta
import requests
from config import TG_TOKEN, TG_CHAT_ID

# ==================== FILE OPERATIONS ====================

def load_json(filepath, default=None):
    """Carga un archivo JSON, retorna default si no existe o hay error."""
    if default is None:
        default = {"nodes": [], "links": []}
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️  Error al leer {filepath}, usando default")
                return default
    return default

def save_graph(data, filepath):
    """Guarda el grafo en JSON con formato bonito."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Grafo guardado: {len(data['nodes'])} nodos, {len(data['links'])} enlaces")

# ==================== STRING UTILITIES ====================

def clean_id(text):
    """Limpia un texto para usarlo como ID."""
    if not text:
        return "unknown"
    return "".join(x for x in text if x.isalnum() or x in ['-', '_']).lower()

def truncate(text, max_length=100):
    """Trunca un texto a longitud máxima."""
    if not text:
        return ""
    return text[:max_length] + "..." if len(text) > max_length else text

# ==================== GRAPH OPERATIONS ====================

def node_exists(graph, node_id):
    """Verifica si un nodo ya existe en el grafo."""
    return any(n['id'] == node_id for n in graph['nodes'])

def add_node(graph, id, name, group, val=10, meta=None):
    """
    Añade un nodo al grafo si no existe.
    
    Returns:
        bool: True si se añadió, False si ya existía
    """
    if meta is None:
        meta = {}
    
    if node_exists(graph, id):
        return False
    
    node = {
        "id": id,
        "name": name,
        "group": group,
        "val": val,
        **meta
    }
    
    graph['nodes'].append(node)
    return True

def update_node_metadata(graph, node_id, new_metadata):
    """Actualiza los metadatos de un nodo existente."""
    for node in graph['nodes']:
        if node['id'] == node_id:
            node.update(new_metadata)
            return True
    return False

def add_link(graph, source, target, value=1):
    """
    Añade un enlace al grafo si no existe.
    Evita duplicados (A->B es igual que B->A).
    """
    # Verificar si el enlace ya existe (en cualquier dirección)
    for link in graph['links']:
        if (link['source'] == source and link['target'] == target) or \
           (link['source'] == target and link['target'] == source):
            return False
    
    graph['links'].append({
        "source": source,
        "target": target,
        "value": value
    })
    return True

def get_node(graph, node_id):
    """Obtiene un nodo por su ID."""
    for node in graph['nodes']:
        if node['id'] == node_id:
            return node
    return None

# ==================== DATE UTILITIES ====================

def parse_date(date_input):
    """
    Parsea diferentes formatos de fecha a datetime.
    Acepta: datetime objects, ISO strings, year integers
    """
    if isinstance(date_input, datetime):
        return date_input
    
    if isinstance(date_input, str):
        # ISO format: 2024-01-15
        try:
            return datetime.fromisoformat(date_input.replace('Z', '+00:00'))
        except:
            pass
        
        # Fecha con hora
        try:
            return datetime.strptime(date_input.split('T')[0], '%Y-%m-%d')
        except:
            pass
    
    if isinstance(date_input, int):
        # Solo año
        return datetime(date_input, 1, 1)
    
    # Default: hoy
    return datetime.now()

def get_newness_level(added_date):
    """
    Determina el nivel de 'novedad' de un paper basado en cuándo se añadió.
    
    Returns:
        str: 'today', 'new', 'recent', o None
    """
    if not added_date:
        return None
    
    date_obj = parse_date(added_date)
    days_old = (datetime.now() - date_obj).days
    
    if days_old <= 1:
        return "today"
    elif days_old <= 3:
        return "new"
    elif days_old <= 7:
        return "recent"
    return None

def format_date_iso(date_obj):
    """Formatea una fecha a ISO string."""
    if isinstance(date_obj, datetime):
        return date_obj.isoformat().split('T')[0]
    return str(date_obj)

# ==================== QUALITY FILTERS ====================

def passes_quality_filter(paper_data, min_abstract_len=100):
    """
    Verifica si un paper pasa los filtros de calidad básicos.
    """
    # Debe tener título
    if not paper_data.get('title'):
        return False
    
    # Debe tener abstract de longitud mínima
    abstract = paper_data.get('abstract', '')
    if len(abstract) < min_abstract_len:
        return False
    
    # No debe contener keywords de exclusión
    exclude_keywords = ['retracted', 'withdrawn', 'erratum']
    title_lower = paper_data['title'].lower()
    abstract_lower = abstract.lower()
    
    for keyword in exclude_keywords:
        if keyword in title_lower or keyword in abstract_lower:
            return False
    
    return True

def should_include_by_keywords(text, keyword_list):
    """
    Verifica si un texto contiene alguna de las keywords de la lista.
    Case-insensitive.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keyword_list)

# ==================== TELEGRAM NOTIFICATIONS ====================

def send_telegram(message, parse_mode='HTML'):
    """Envía un mensaje a Telegram."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return False
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                'chat_id': TG_CHAT_ID,
                'text': message,
                'parse_mode': parse_mode
            },
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️  Error enviando Telegram: {e}")
        return False

def notify_new_papers(new_papers):
    """Envía notificación de nuevos papers a Telegram."""
    if not new_papers:
        return
    
    msg = f"🚀 <b>MUSTTELA UPDATE</b>\n\n"
    msg += f"📊 {len(new_papers)} nuevos papers añadidos\n\n"
    
    # Mostrar hasta 3 papers
    for i, paper in enumerate(new_papers[:3], 1):
        title = truncate(paper.get('name', 'Sin título'), 60)
        url = paper.get('url', '#')
        msg += f"{i}. {title}\n🔗 {url}\n\n"
    
    if len(new_papers) > 3:
        msg += f"<i>...y {len(new_papers) - 3} más.</i>\n\n"
    
    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    
    send_telegram(msg)

# ==================== RATE LIMITING ====================

class RateLimiter:
    """Simple rate limiter para APIs."""
    
    def __init__(self, calls_per_second=1):
        self.delay = 1.0 / calls_per_second
        self.last_call = 0
    
    def wait(self):
        """Espera el tiempo necesario entre llamadas."""
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

# ==================== STATS ====================

def print_graph_stats(graph):
    """Imprime estadísticas del grafo."""
    papers = [n for n in graph['nodes'] if n['group'] == 'paper']
    authors = [n for n in graph['nodes'] if n['group'] == 'author']
    topics = [n for n in graph['nodes'] if n['group'] == 'topic']
    seeds = [n for n in graph['nodes'] if n.get('is_seed', False)]
    
    print("\n" + "="*50)
    print("📊 ESTADÍSTICAS DEL GRAFO")
    print("="*50)
    print(f"📄 Papers: {len(papers)}")
    print(f"   └─ Seeds: {len(seeds)}")
    print(f"👤 Autores: {len(authors)}")
    print(f"🏷️  Topics: {len(topics)}")
    print(f"🔗 Enlaces: {len(graph['links'])}")
    print("="*50 + "\n")
