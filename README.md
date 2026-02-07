# 🚀 MUSTTELA - AI Journalism Research Graph

Sistema de dos fases para visualización y exploración de papers académicos sobre IA y periodismo.

## 📋 Características

- **Vista 3D interactiva** con grafo de fuerza
- **Vista de grid** (bento layout) para exploración
- **Sistema de badges**: ArXiv, S2, Seeds fundacionales, Novedad
- **Degradación temporal** de papers nuevos: 🔥 HOY → ✨ NUEVO → 📌 RECIENTE
- **Seeds destacados** en color dorado
- **Filtros avanzados**: por fuente, favoritos, novedad
- **Actualización automática diaria**

## 🏗️ Estructura del Proyecto

```
musttela-graph/
├── scripts/
│   ├── config.py              # Configuración centralizada
│   ├── utils.py               # Funciones compartidas
│   ├── initial_build.py       # FASE 1: Llenado inicial
│   └── daily_update.py        # FASE 2: Updates diarios
├── docs/
│   ├── graph_data.json        # Datos del grafo
│   └── index.html             # Frontend
├── config/
│   └── seeds.json             # Papers fundacionales
├── requirements.txt
└── .github/workflows/
    ├── initial-build.yml      # Workflow manual
    └── daily-update.yml       # Workflow automático
```

## 🎯 Sistema de Dos Fases

### FASE 1: Llenado Inicial (Día 0)
Construcción del corpus base de ~500 papers bien seleccionados.

**Estrategia temporal:**
- 2020-2021: 50 papers (fundacionales, min. 20 citas)
- 2022-2023: 150 papers (post-ChatGPT, min. 10 citas)
- 2024-2025: 250 papers (investigación reciente, min. 3 citas)
- 2026: 50 papers (lo más reciente)

**Keywords categorizadas:**
- Core Journalism: `algorithmic journalism`, `computational journalism`, etc.
- Communication Theory: `AI agenda setting`, `algorithmic gatekeeping`, etc.
- Ethics & Bias: `AI bias journalism`, `automated journalism ethics`, etc.
- Misinformation: `AI misinformation detection`, `automated fact-checking`, etc.
- Industry: `AI newsroom automation`, `journalism AI adoption`, etc.
- Emerging: `large language models journalism`, `GPT-4 journalism`, etc.

### FASE 2: Actualización Diaria (Día 1+)
Añade solo papers de las últimas 48-72h con sistema de degradación:

- **🔥 HOY** (1 día): Color verde brillante
- **✨ NUEVO** (1-3 días): Color verde suave
- **📌 RECIENTE** (3-7 días): Color verde sutil
- Después de 7 días: sin badge

## 🚀 Instrucciones de Uso

### 1️⃣ RESET COMPLETO (Empezar desde cero)

#### Paso 1: Hacer backup (opcional)
```bash
# Descargar el graph_data.json actual de GitHub por si acaso
wget https://raw.githubusercontent.com/TU_USUARIO/musttela-graph/main/docs/graph_data.json -O backup_graph_data.json
```

#### Paso 2: Reemplazar todo el código local

```bash
# Eliminar archivos antiguos (excepto .git)
rm -rf docs/ scripts/ .github/ config/
rm updater.py

# Copiar todos los archivos nuevos del sistema reorganizado
# (asumiendo que tienes los archivos de este reset en /nueva-estructura/)
cp -r /nueva-estructura/* .

# Verificar estructura
tree -L 2
```

#### Paso 3: Configurar secretos en GitHub

Ve a tu repositorio → Settings → Secrets → Actions → New repository secret

Añade:
- `TELEGRAM_TOKEN`: Tu token de bot de Telegram (opcional)
- `TELEGRAM_CHAT_ID`: Tu chat ID (opcional)

#### Paso 4: Ejecutar llenado inicial

**Opción A: Localmente** (para probar)
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar llenado inicial
cd scripts
python initial_build.py
```

**Opción B: En GitHub Actions** (recomendado)
1. Ve a: Actions → 🚀 Llenado Inicial del Grafo
2. Click en "Run workflow"
3. Espera ~30-45 minutos (depende de cuántos papers encuentre)
4. Revisa que docs/graph_data.json se haya creado

#### Paso 5: Verificar en GitHub Pages

1. Settings → Pages → Source: Deploy from a branch → main → /docs
2. Espera 2-3 minutos
3. Visita: `https://TU_USUARIO.github.io/musttela-graph/`

✅ **Deberías ver el grafo 3D con ~500 papers**

#### Paso 6: Activar actualización automática

El workflow `daily-update.yml` ya está configurado para correr automáticamente a las 09:00 UTC cada día.

Puedes probarlo manualmente:
1. Actions → 🔄 Actualización Diaria
2. Click en "Run workflow"

---

### 2️⃣ USO DIARIO (Después del reset)

Una vez completado el reset, el sistema funciona automáticamente:

1. **Cada día a las 09:00 UTC**:
   - GitHub Actions ejecuta `daily_update.py`
   - Busca papers de las últimas 48h
   - Los marca como "nuevos"
   - Limpia badges de papers >7 días
   - Hace commit y push automático
   - (Opcional) Envía notificación a Telegram

2. **En el frontend**:
   - Papers nuevos aparecen con badge 🔥 HOY
   - Se destacan con color verde brillante
   - Aparecen en sección "Novedades"
   - Después de 3 días → badge ✨ NUEVO
   - Después de 7 días → sin badge

---

## 🎨 Frontend Features

### Vista de Grafo 3D
- **Seeds**: Dorado, tamaño grande, badge 📚 FUNDACIONAL
- **Papers nuevos**: Verde degradado según antigüedad
- **Hover**: Tooltip con título, abstract, badges
- **Click**: Abre URL del paper

### Vista de Grid
- **Sección "Novedades"**: Solo papers con badge
- **Sección "Todos"**: Todos los papers filtrados
- **Favoritos**: Click en ★ para marcar (guardado en localStorage)
- **Badges**: Fuente (ArXiv/S2), Novedad, Seed

### Filtros
- **Favoritos**: Solo papers marcados
- **Fuente**: ArXiv y/o S2
- **Novedad**: 
  - Todos
  - Solo hoy (🔥)
  - Última semana (📌)

---

## 🛠️ Mantenimiento

### Ver logs de workflows
```bash
# En GitHub
Actions → Click en un workflow run → Ver detalles
```

### Ejecutar scripts manualmente
```bash
cd scripts

# Llenado inicial (CUIDADO: añade ~500 papers)
python initial_build.py

# Actualización diaria (solo papers recientes)
python daily_update.py
```

### Modificar configuración

Edita `scripts/config.py` para cambiar:
- Keywords de búsqueda
- Rangos temporales
- Límites de papers por período
- Colores de badges
- Días de "novedad"

### Añadir/quitar seeds

Edita `config/seeds.json`:
```json
[
  "10.1145/3351095.3372859",
  "arXiv:1706.03762",
  "TU_NUEVO_DOI_O_ARXIV_ID"
]
```

---

## 📊 Estadísticas del Grafo

El frontend muestra en el header:
- **PAPERS**: Total de papers
- **AUTHORS**: Total de autores únicos
- **TOPICS**: Total de topics/categorías

---

## 🐛 Troubleshooting

### El workflow falla en GitHub Actions

**Posibles causas:**
1. Rate limit de ArXiv o Semantic Scholar → Espera unas horas
2. Error en scripts → Revisa los logs en Actions
3. Permisos insuficientes → Verifica que el workflow tenga `contents: write`

### No se ven los papers nuevos

1. Verifica que `graph_data.json` tenga campo `added_date`
2. Revisa la consola del navegador (F12) por errores
3. Confirma que el script `daily_update.py` se ejecutó correctamente

### El grafo 3D no carga

1. Abre la consola (F12) y busca errores
2. Verifica que `graph_data.json` esté en `docs/`
3. Confirma que el JSON no esté corrupto: `python -m json.tool docs/graph_data.json`

---

## 📝 Notas Adicionales

### Límites de APIs
- **ArXiv**: ~1 request/segundo (el script ya tiene rate limiting)
- **Semantic Scholar**: ~100 requests/5 minutos (el script ya tiene rate limiting)

### Tamaño del grafo
- Con 500 papers + autores + topics → ~2000-2500 nodos
- El grafo 3D maneja bien hasta 5000 nodos
- Si superas ese límite, considera:
  - Reducir max_results en config.py
  - Implementar paginación en grid view

### Personalización de diseño

El diseño es "brutalist-academic". Para cambiar:
- Edita las CSS variables en `docs/index.html`
- Cambia fonts en el `<link>` de Google Fonts
- Modifica colores en `:root { ... }`

---

## 🎯 Próximos Pasos (Después del Reset)

1. ✅ Ejecutar llenado inicial
2. ✅ Verificar que el grafo se vea correctamente
3. ✅ Probar una actualización diaria manual
4. ⏳ Dejar que corra automáticamente
5. 📈 Monitorear durante una semana
6. 🎨 (Opcional) Personalizar diseño según gustos

---

## 📜 Licencia

Este proyecto es de uso académico/investigación. Los papers pertenecen a sus respectivos autores.

---

## 🙋 Soporte

Si algo no funciona, revisa:
1. Los logs de GitHub Actions
2. La consola del navegador (F12)
3. Este README

¡Happy researching! 🚀📚
