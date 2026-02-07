```text                                                                              
      __                                     
     /  \      M  U  S  T  T  E  L  A        
    /|oo \     F  A  C  T  O  R  Y           
   (_|  /_)    ──────────────────────        
    _`@/_ \    [ Est. 2025 ]                 
   |     | \                                 
   |     |  \      >> HUNTING FOR KNOWLEDGE  
   |_____|   \     >> IN THE NOISE           
      | |     \                              
      |_|      \                             
     _|_|_                                   
                                             
                                             
  [ SYSTEM_LOG: MODULE_01 // ETHEREAL_ARCHIVE ]
  > Status: ......................... ONLINE
  > Subject: ........................ AI_JOURNALISM
  > Vibe: ........................... ACADEMIC_CYBERPUNK

================================================================
  00. MANIFIESTO (ABSTRACT)
================================================================

  Bienvenido a ETHEREAL ARCHIVE. 
  
  Este es el primer módulo de la factoría. No es una lista de
  PDFs aburrida; es un organismo vivo. Un sistema de dos fases
  que rastrea, caza y digiere papers académicos sobre IA y
  periodismo, separando la señal del ruido mediante un
  protocolo de degradación visual.

  Si el paper es nuevo, brilla. Si es viejo, se desvanece
  en la estructura del grafo.

================================================================
  01. ARQUITECTURA VISUAL & FEATURES
================================================================

       .       .         .           .
           .       +        .     .      .
      .     .    (doi)    .     .     .
          .     ───●───      .    (arxiv)
       .       .   │    .        .    .
     (pdf) ─────── □ ────── (s2)    .
       .       .   │    .        .
          .      (txt)     +    .     .

  [+] VISTA 3D INTERACTIVA
      Grafo de fuerza. La proximidad = relación semántica.

  [+] SISTEMA DE BADGES & DEGRADACIÓN (The Decay)
      Los papers tienen "vida media visual".
      
      🔥 HOY (24h) ........ Verde Brillante (Prioridad Máxima)
      ✨ NUEVO (1-3d) ..... Verde Suave (Alta Visibilidad)
      📌 RECIENTE (3-7d) .. Verde Sutil (Estable)
      💀 ARCHIVADO ........ Gris (Grafo Histórico)

  [+] SEEDS (SEMILLAS)
      Nodos dorados inmutables. Papers fundacionales que
      actúan como anclas de verdad.

  [+] BENTO GRID
      Layout de exploración rápida con filtros de "Novedad".

================================================================
  02. ESTRUCTURA DEL SISTEMA
================================================================

  MUSTTELA-FACTORY/
  │
  ├── scripts/                 # El cerebro (Python)
  │   ├── config.py            # ⚙️ Configuración central
  │   ├── utils.py             # 🔧 Herramientas compartidas
  │   ├── initial_build.py     # 🏗️ FASE 1: Big Bang
  │   └── daily_update.py      # 🔄 FASE 2: Rutina diaria
  │
  ├── docs/                    # La cara (Frontend)
  │   ├── graph_data.json      # 🧠 La memoria del grafo
  │   └── index.html           # 🎨 El lienzo visual
  │
  ├── config/
  │   └── seeds.json           # 🌱 Papers Semilla (Json)
  │
  └── .github/workflows/       # Los autómatas
      ├── initial-build.yml    # Ejecución manual
      └── daily-update.yml     # Cron job (09:00 UTC)

================================================================
  03. FASES OPERATIVAS (THE LOGIC)
================================================================

  ¶ FASE 1: LLENADO INICIAL (The Big Bang)
  --------------------------------------------------------
  Construcción del corpus histórico (~500 papers).
  Se ejecuta una sola vez.

  [ CRONOLOGÍA DE INGESTA ]
  > 2020-21 .. Fundacionales (>20 citas) .. [░░░░░] 50 items
  > 2022-23 .. Post-ChatGPT  (>10 citas) .. [▒▒▒▒▒] 150 items
  > 2024-25 .. Recientes     (>3 citas) ... [▓▓▓▓▓] 250 items
  > 2026+ .... Breaking News (Sin filtro) . [█████] 50 items

  [ KEYWORDS OBJETIVO ]
  > Core: ....... Algorithmic journalism, Computational journ.
  > Theory: ..... AI agenda setting, Algorithmic gatekeeping
  > Ethics: ..... AI bias, Automated journalism ethics
  > Fake: ....... AI misinformation detection, Fact-checking
  > Industry: ... Newsroom automation, AI adoption

  
  ¶ FASE 2: ACTUALIZACIÓN DIARIA (The Hunt)
  --------------------------------------------------------
  El hurón sale a cazar cada día a las 09:00 UTC.
  Solo busca en las últimas 48-72 horas.
  
  [ ALGORITMO ]
  1. Detectar paper nuevo en ArXiv/Semantic Scholar.
  2. Asignar badge 🔥 HOY.
  3. Notificar (opcional).
  4. Limpiar badges de papers antiguos (>7 días).

================================================================
  04. INSTRUCCIONES DE DESPLIEGUE
================================================================

  [A] RESET COMPLETO (TABULA RASA)
  --------------------------------------------------------
  Para empezar desde cero absoluto:

  1. BACKUP (Opcional)
     $ wget .../graph_data.json -O backup.json

  2. PURGA & REEMPLAZO
     Elimina /docs, /scripts y reemplaza con la nueva
     estructura.

  3. CONFIGURACIÓN DE SECRETOS (GitHub Repo Settings)
     Si quieres notificaciones de Telegram:
     > TELEGRAM_TOKEN
     > TELEGRAM_CHAT_ID

  4. EJECUCIÓN (Génesis)
     Ve a GitHub Actions -> "🚀 Llenado Inicial del Grafo"
     Click > "Run Workflow".
     (Tiempo estimado: 30-45 mins. Paciencia, está leyendo).


  [B] MANTENIMIENTO DIARIO
  --------------------------------------------------------
  El sistema es autónomo. Pero si quieres forzarlo:
  Ve a GitHub Actions -> "🔄 Actualización Diaria"
  Click > "Run Workflow".


  [C] OBSERVACIÓN LOCAL
  --------------------------------------------------------
  Para ver el grafo en tu máquina:
  $ pip install -r requirements.txt
  $ python -m http.server 8000 (dentro de /docs)
  > Visita localhost:8000

================================================================
  05. CONFIGURACIÓN & TUNING
================================================================

  Archivo: scripts/config.py
  Aquí controlas la sensibilidad del sistema.

  > MAX_RESULTS ....... Límite de papers por búsqueda
  > DATE_WINDOW ....... Días hacia atrás para buscar
  > BADGE_COLORS ...... Hex codes para los estados
  
  Archivo: config/seeds.json
  > Añade DOIs aquí para forzar la aparición de papers clave.
  > ["10.1145/xxxx", "arXiv:1706.xxxx"]

================================================================
  06. TROUBLESHOOTING (BUGS DEL MATRIX)
================================================================

  [!] ERROR: GitHub Action falla
      > Causa: Rate limit de ArXiv o Semantic Scholar.
      > Solución: Espera unas horas. El script ya limita
        sus peticiones, pero las APIs son caprichosas.

  [!] ERROR: No aparecen papers nuevos
      > Revisa si 'graph_data.json' tiene el campo 'added_date'.
      > Verifica los logs del workflow 'Daily Update'.

  [!] ERROR: Grafo 3D no carga / Pantalla negra
      > Abre consola (F12).
      > Verifica que el JSON no esté corrupto.
      > Límite recomendado: < 5000 nodos para fluidez.

================================================================
  CREDITS & LICENSE
================================================================
  
  Diseñado para investigación académica.
  Los papers pertenecen a sus autores.
  El código es libre.
  
         __      _
       o'')}____//
        `_/      )   < Happy Researching! >
        (_(_/-(_/
        
  MUSTTELA FACTORY © 2025
────────────────────────────────────────────────────────────────
