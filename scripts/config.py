"""
MUSTTELA - Configuración centralizada
"""

# ==================== PATHS ====================
JSON_FILE = "docs/graph_data.json"
SEEDS_FILE = "config/seeds.json"

# ==================== TELEGRAM ====================
import os
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ==================== KEYWORDS ESTRUCTURADAS ====================
KEYWORDS = {
    # CORE: AI + Journalism (específico)
    "core_journalism": [
        "algorithmic journalism",
        "computational journalism", 
        "automated news writing",
        "robot journalism",
        "AI news generation",
        "GPT journalism"
    ],
    
    # AI + Communication Theories
    "communication_theory": [
        "AI agenda setting",
        "algorithmic gatekeeping",
        "AI framing news",
        "computational framing",
        "AI news values",
        "algorithmic news selection"
    ],
    
    # AI + Media Ethics & Bias
    "ethics_bias": [
        "AI bias journalism",
        "algorithmic bias news",
        "AI media ethics",
        "automated journalism ethics",
        "AI fact-checking bias",
        "algorithmic accountability journalism"
    ],
    
    # AI + Misinformation
    "misinformation": [
        "AI misinformation detection",
        "automated fact-checking",
        "AI deepfakes journalism",
        "algorithmic content moderation news",
        "AI verification journalism"
    ],
    
    # AI + Newsroom/Industry
    "industry_practice": [
        "AI newsroom automation",
        "journalism AI adoption",
        "news organizations AI",
        "AI journalism workflow",
        "automated content production"
    ],
    
    # Emerging/Specific
    "emerging": [
        "large language models journalism",
        "LLM news",
        "GPT-4 journalism",
        "generative AI news production",
        "AI personalized news"
    ]
}

# ==================== ESTRATEGIA TEMPORAL ====================
TEMPORAL_STRATEGY = {
    "2020-2021": {
        "max_results": 50,
        "min_citations": 20,
        "description": "Papers fundacionales y seminales"
    },
    "2022-2023": {
        "max_results": 150,
        "min_citations": 10,
        "description": "Desarrollo del campo post-ChatGPT"
    },
    "2024-2025": {
        "max_results": 250,
        "min_citations": 3,
        "description": "Investigación reciente y preprints"
    },
    "2026": {
        "max_results": 50,
        "min_citations": 0,
        "description": "Lo más reciente disponible"
    }
}

# ==================== ARXIV CONFIG ====================
ARXIV_CONFIG = {
    "categories": ["cs.CY", "cs.AI", "cs.CL"],  # Computers & Society, AI, Computation & Language
    "max_results_per_query": 20,
    "sort_by": "relevance"  # or "submittedDate"
}

# ==================== SEMANTIC SCHOLAR CONFIG ====================
S2_CONFIG = {
    "max_results_per_keyword": 15,
    "fields": [
        "paperId", "title", "abstract", "authors", "year", 
        "citationCount", "fieldsOfStudy", "url", "publicationDate"
    ],
    "min_citation_count": 3  # Mínimo de citas para considerar
}

# ==================== DAILY UPDATE CONFIG ====================
DAILY_UPDATE = {
    "lookback_days": 3,  # Buscar papers de los últimos 3 días
    "max_new_papers": 20,  # Máximo de papers nuevos por día
    "new_paper_threshold_days": 7  # Días que un paper se considera "nuevo"
}

# ==================== NEWNESS LEVELS ====================
# Para el sistema de degradación visual
NEWNESS_LEVELS = {
    "today": {
        "days": 1,
        "badge": "🔥 HOY",
        "color": "#00ff88",
        "size_multiplier": 1.4
    },
    "new": {
        "days": 3,
        "badge": "✨ NUEVO",
        "color": "#44ff99",
        "size_multiplier": 1.2
    },
    "recent": {
        "days": 7,
        "badge": "📌 RECIENTE",
        "color": "#88ffaa",
        "size_multiplier": 1.1
    }
}

# ==================== SEED CONFIG ====================
SEED_CONFIG = {
    "color": "#FFD700",  # Dorado
    "badge": "📚 FUNDACIONAL",
    "size": 50,
    "always_visible": True,
    "expand_references": True,  # Traer papers que citan a los seeds
    "max_references_per_seed": 10
}

# ==================== FILTROS DE CALIDAD ====================
QUALITY_FILTERS = {
    "min_abstract_length": 100,  # Caracteres mínimos en abstract
    "required_fields": ["title", "abstract", "authors"],
    "exclude_keywords": [
        "retracted", "withdrawn", "erratum"
    ]
}
