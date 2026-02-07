# 📦 MUSTTELA - SISTEMA COMPLETO DE DOS FASES

## 🎯 Resumen Ejecutivo

Sistema reorganizado para MUSTTELA que implementa:

1. **Llenado inicial inteligente** (~500 papers de 2020-2026 con estrategia temporal)
2. **Actualización diaria automática** (papers recientes con sistema de novedad)
3. **Frontend renovado** (brutalist-academic design, badges, filtros avanzados)
4. **Seeds destacados** (color dorado, siempre visibles)
5. **Degradación temporal** (HOY → NUEVO → RECIENTE → normal)

---

## 📁 Archivos Incluidos

### Scripts Python
- `scripts/config.py` - Configuración centralizada (keywords, colores, límites)
- `scripts/utils.py` - Funciones compartidas (I/O, filtrado, stats)
- `scripts/initial_build.py` - Llenado inicial con estrategia temporal
- `scripts/daily_update.py` - Actualización diaria con detección de novedad

### Frontend
- `docs/index.html` - Interfaz completa (3D graph + grid view + filtros)

### Configuración
- `config/seeds.json` - 10 papers fundacionales
- `requirements.txt` - Dependencias Python

### Workflows
- `.github/workflows/initial-build.yml` - Trigger manual para día 0
- `.github/workflows/daily-update.yml` - Cron diario (09:00 UTC)

### Documentación
- `README.md` - Documentación completa del sistema
- `MIGRATION_GUIDE.md` - Guía paso a paso para migrar desde el sistema actual

---

## 🚀 Quick Start

### Opción 1: Reset Completo (Recomendado)

```bash
# 1. Backup del sistema actual
mkdir backup && cp -r docs/ .github/ updater.py backup/

# 2. Limpiar y copiar nuevo sistema
rm -rf docs/ scripts/ config/ .github/ updater.py
cp -r /ruta/a/musttela-reset/* .

# 3. Force push a GitHub
git add .
git commit -m "🔄 Reset: Sistema de dos fases"
git push origin main --force

# 4. Ejecutar llenado inicial
# Opción A: Localmente
pip install -r requirements.txt
cd scripts && python initial_build.py

# Opción B: GitHub Actions
# Ve a: Actions → "🚀 Llenado Inicial" → Run workflow

# 5. Verificar en GitHub Pages
# https://TU_USUARIO.github.io/musttela-graph/
```

### Opción 2: Probar Localmente Primero

```bash
# 1. Clonar en directorio separado
git clone https://github.com/TU_USUARIO/musttela-graph.git musttela-test
cd musttela-test

# 2. Copiar nuevo sistema
cp -r /ruta/a/musttela-reset/* .

# 3. Ejecutar llenado inicial
pip install -r requirements.txt
cd scripts && python initial_build.py

# 4. Ver resultado
cd ../docs
python -m http.server 8000
# Abrir: http://localhost:8000

# 5. Si funciona, hacer push al repo principal
```

---

## 🎨 Features del Frontend

### Vista 3D
- **Seeds**: Color dorado (#FFD700), tamaño 50, badge 📚 FUNDACIONAL
- **Papers nuevos**: 
  - 🔥 HOY: Verde brillante (#00ff88), tamaño x1.4
  - ✨ NUEVO: Verde suave (#44ff99), tamaño x1.2
  - 📌 RECIENTE: Verde sutil (#88ffaa), tamaño x1.1
- **Papers normales**: Cyan (#4ECDC4)
- **Autores**: Rojo (#FF6B6B)
- **Topics**: Verde agua (#95E1D3)

### Vista Grid
- **Sección "Novedades"**: Solo papers con badge (auto-oculta si vacía)
- **Sección "Todos"**: Grid responsive de cards
- **Badges**: Fuente (ArXiv/S2), Novedad, Seed
- **Favoritos**: Guardados en localStorage
- **Click**: Abre URL del paper

### Filtros
```javascript
{
  favorites: boolean,        // Solo favoritos
  arxiv: boolean,           // Mostrar ArXiv
  s2: boolean,              // Mostrar S2
  newAll: boolean,          // Todos los papers
  newToday: boolean,        // Solo 🔥 HOY
  newWeek: boolean          // Solo 📌 última semana
}
```

---

## ⚙️ Configuración Avanzada

### Modificar Keywords

Edita `scripts/config.py`:

```python
KEYWORDS = {
    "core_journalism": [
        "algorithmic journalism",
        "computational journalism",
        # Añade más aquí
    ],
    "communication_theory": [
        "AI agenda setting",
        # etc.
    ]
}
```

### Ajustar Límites Temporales

```python
TEMPORAL_STRATEGY = {
    "2020-2021": {
        "max_results": 50,      # Cambia esto
        "min_citations": 20,    # O esto
    }
}
```

### Cambiar Colores

Edita `docs/index.html`:

```css
:root {
    --accent-paper: #4ECDC4;    /* Cambiar color papers */
    --accent-seed: #FFD700;     /* Cambiar color seeds */
    --new-today: #00ff88;       /* Cambiar color "HOY" */
}
```

### Modificar Días de Novedad

```python
# En config.py
DAILY_UPDATE = {
    "new_paper_threshold_days": 7  # Cambiar de 7 a 5, 10, etc.
}
```

---

## 📊 Estadísticas Esperadas

### Después del Llenado Inicial
```
Papers: ~500
  ├─ Seeds: 10
  ├─ ArXiv: ~100-150
  └─ S2: ~350-400

Authors: ~1200-1500
Topics: ~200-300
Links: ~2500-3000
```

### Después de 1 Mes de Updates Diarios
```
Papers: ~520-550
  ├─ Seeds: 10
  ├─ Nuevos (últimos 7 días): 5-10
  └─ Normales: ~510-530

Graph Data Size: ~2-3 MB
```

---

## 🔧 Troubleshooting

### El llenado inicial falla

**Error**: `Rate limit exceeded`
**Solución**: Espera 1 hora y vuelve a ejecutar. Los scripts tienen rate limiting pero APIs pueden tener límites globales.

**Error**: `No module named 'semanticscholar'`
**Solución**: `pip install semanticscholar`

### No se ven papers nuevos en el frontend

1. Abre DevTools (F12) → Console
2. Busca errores JavaScript
3. Verifica que `graph_data.json` tenga campos `added_date` y `newness_level`
4. Haz hard refresh (Ctrl+Shift+R)

### El workflow diario no corre

1. Verifica que el workflow esté en `.github/workflows/daily-update.yml`
2. Comprueba que el cron esté activado (Actions → workflow → ···)
3. Revisa los logs de ejecución en Actions

---

## 📞 Contacto y Soporte

### Recursos
- **README.md**: Documentación completa
- **MIGRATION_GUIDE.md**: Guía de migración paso a paso
- **GitHub Actions Logs**: Para debugging

### Si algo no funciona
1. Lee el README.md completo
2. Revisa MIGRATION_GUIDE.md
3. Verifica los logs de GitHub Actions
4. Comprueba la consola del navegador (F12)

---

## ✅ Checklist de Verificación Final

Antes de considerar la migración completa:

- [ ] Todos los archivos copiados correctamente
- [ ] `pip install -r requirements.txt` sin errores
- [ ] Llenado inicial ejecutado exitosamente
- [ ] `docs/graph_data.json` creado con ~500 papers
- [ ] Frontend carga en localhost o GitHub Pages
- [ ] Grafo 3D renderiza correctamente
- [ ] Seeds visibles en color dorado
- [ ] Filtros funcionan
- [ ] Workflow diario configurado
- [ ] (Opcional) Telegram configurado

---

## 🎉 ¡Éxito!

Si llegaste hasta aquí y todos los checks están ✅, **¡felicidades!**

Ahora tienes:
- ✨ Un corpus limpio de ~500 papers bien seleccionados
- 🔄 Actualización automática diaria con novedad visual
- 🎨 Frontend profesional con filtros avanzados
- 📚 Seeds destacados como referencias fundacionales
- 🤖 Sistema completamente automatizado

**Siguiente paso**: Deja que el sistema trabaje para ti durante una semana. Cada día añadirá papers nuevos y los destacará con badges. Después de 7 días, el sistema se estabiliza y tienes un grafo vivo que crece orgánicamente.

**Disfruta tu nueva herramienta de investigación! 🚀📊**

---

## 📝 Changelog

### v2.0.0 - Reset Completo (2026-02-01)
- ✨ Sistema de dos fases (llenado inicial + updates diarios)
- ✨ Keywords estructuradas por categorías
- ✨ Sistema de novedad con degradación temporal
- ✨ Seeds destacados en dorado
- ✨ Frontend renovado (brutalist-academic design)
- ✨ Filtros avanzados (fuente, favoritos, novedad)
- ✨ Workflows separados (initial-build, daily-update)
- 📚 Documentación completa (README + MIGRATION_GUIDE)

### v1.x - Sistema Anterior
- Basic updater.py con límites no configurados
- KeyBERT para extracción de keywords
- Sin sistema de novedad
- Seeds no destacados
