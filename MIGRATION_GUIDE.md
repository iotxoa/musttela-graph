# 🔄 GUÍA DE MIGRACIÓN: Del Sistema Actual al Nuevo

Esta guía te ayuda a pasar del sistema actual (con 1591 papers mezclados) al nuevo sistema limpio y organizado.

---

## 📋 Estado Actual vs Estado Objetivo

### AHORA tienes:
- ✅ 1591 papers en GitHub (muchos irrelevantes)
- ✅ Mejoras UX/UI en local (badges, filtros, tooltips)
- ❌ Keywords extraídas con KeyBERT (raras)
- ❌ No hay sistema de "novedad"
- ❌ Seeds no destacados

### DESPUÉS tendrás:
- ✅ ~500 papers bien seleccionados
- ✅ Keywords estructuradas por categorías
- ✅ Sistema de novedad con degradación (HOY → NUEVO → RECIENTE)
- ✅ Seeds en dorado con badge especial
- ✅ Actualización automática diaria

---

## 🚀 PROCESO DE MIGRACIÓN (Paso a Paso)

### FASE PREPARACIÓN

#### 1. Backup del Estado Actual (Seguridad)

```bash
# En tu máquina local, dentro del repo:
mkdir backup_viejo_sistema
cp -r docs/ backup_viejo_sistema/
cp -r .github/ backup_viejo_sistema/
cp updater.py backup_viejo_sistema/

# También descarga una copia de GitHub por si acaso
wget https://raw.githubusercontent.com/iotxoa/musttela-graph/main/docs/graph_data.json -O backup_github_graph.json

echo "✅ Backup completado"
```

#### 2. Descargar el Nuevo Sistema

```bash
# (Asume que tienes los archivos del nuevo sistema en algún lugar)
# Si los tienes en una carpeta temporal llamada 'musttela-reset':

# Eliminar archivos antiguos (MANTÉN .git)
rm -rf docs/
rm -rf scripts/ 2>/dev/null || true
rm -rf config/ 2>/dev/null || true
rm updater.py
rm -rf .github/workflows/

# Copiar nuevo sistema
cp -r /ruta/a/musttela-reset/* .

# Verificar estructura
ls -la
# Deberías ver: docs/, scripts/, config/, .github/, requirements.txt, README.md
```

---

### FASE EJECUCIÓN

#### 3. Limpiar GitHub (Preparar para el Reset)

**Opción A: Force push (MÁS LIMPIA, pero DESTRUCTIVA)**

```bash
# CUIDADO: Esto BORRARÁ los 1591 papers de GitHub
# Solo hazlo si estás SEGURO de que quieres empezar de cero

# Asegúrate de tener backup antes

# Hacer commit del nuevo sistema
git add .
git commit -m "🔄 Reset completo: Nuevo sistema de dos fases"

# Force push (SOBRESCRIBE GitHub)
git push origin main --force

echo "⚠️  GitHub ahora tiene el nuevo sistema SIN papers"
echo "   Debes ejecutar el llenado inicial para poblar el grafo"
```

**Opción B: Merge suave (CONSERVADORA, pero menos limpia)**

```bash
# Esto mantiene el historial de GitHub pero puede haber conflictos

git add .
git commit -m "🔄 Migración a sistema de dos fases"

# Intenta push normal
git push origin main

# Si hay conflictos con graph_data.json:
git pull origin main
# Resolver conflicto: acepta la versión local (vacía)
git checkout --ours docs/graph_data.json
git add docs/graph_data.json
git commit -m "Resolver conflicto: usar grafo vacío para reset"
git push origin main
```

**Mi recomendación**: Opción A (force push) para empezar limpio

#### 4. Ejecutar Llenado Inicial

**4a. Localmente (para ver el progreso)**

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar llenado inicial
cd scripts
python initial_build.py

# Esto tomará 30-45 minutos
# Verás el progreso en consola:
# - Processing seeds
# - Searching ArXiv 2020-2021...
# - Searching S2 2022-2023...
# etc.

# Al final verás:
# ✅ LLENADO INICIAL COMPLETADO
# Papers: ~500
```

**4b. O en GitHub Actions (más fácil, pero no ves progreso en vivo)**

```bash
# Primero haz push de los scripts
git add .
git commit -m "Añadir scripts de nuevo sistema"
git push origin main

# Luego ve a GitHub:
# 1. Actions → 🚀 Llenado Inicial del Grafo
# 2. Click "Run workflow"
# 3. Espera ~40 minutos
# 4. Verifica que docs/graph_data.json se actualizó
```

#### 5. Verificar el Resultado

```bash
# Descargar el nuevo graph_data.json de GitHub
git pull origin main

# Ver estadísticas
cd scripts
python << EOF
import json
with open('../docs/graph_data.json') as f:
    data = json.load(f)
papers = [n for n in data['nodes'] if n['group'] == 'paper']
seeds = [p for p in papers if p.get('is_seed')]
print(f"Total papers: {len(papers)}")
print(f"Seeds: {len(seeds)}")
print(f"ArXiv: {sum(1 for p in papers if p.get('source') == 'arxiv')}")
print(f"S2: {sum(1 for p in papers if p.get('source') == 's2')}")
EOF
```

**Esperado:**
```
Total papers: ~500
Seeds: 10
ArXiv: ~100-150
S2: ~350-400
```

#### 6. Activar GitHub Pages

Si aún no lo tienes activado:

1. Ve a: Settings → Pages
2. Source: Deploy from a branch
3. Branch: main
4. Folder: /docs
5. Save

Espera 2-3 minutos y visita:
`https://iotxoa.github.io/musttela-graph/`

**Deberías ver:**
- Grafo 3D con ~500 papers
- Seeds en color dorado
- Estadísticas en header
- Filtros funcionando

---

### FASE ACTIVACIÓN

#### 7. Configurar Notificaciones Telegram (Opcional)

```bash
# 1. Crear un bot de Telegram:
#    - Habla con @BotFather en Telegram
#    - /newbot
#    - Sigue instrucciones
#    - Guarda el TOKEN

# 2. Obtener tu CHAT_ID:
#    - Habla con @userinfobot
#    - Te dará tu CHAT_ID

# 3. Añadir secretos en GitHub:
#    - Repo → Settings → Secrets → Actions
#    - New repository secret:
#      Name: TELEGRAM_TOKEN, Value: (tu token)
#    - New repository secret:
#      Name: TELEGRAM_CHAT_ID, Value: (tu chat id)
```

#### 8. Probar Actualización Diaria

```bash
# Opción A: Localmente
cd scripts
python daily_update.py

# Debería decir:
# ℹ️  No se encontraron papers nuevos hoy
# (porque acabas de crear el corpus)

# Opción B: En GitHub Actions
# Actions → 🔄 Actualización Diaria → Run workflow
```

#### 9. Dejar que el Sistema Corra Solo

✅ **¡Listo!** Ahora el sistema:
- Cada día a las 09:00 UTC busca papers nuevos
- Los marca con badge 🔥 HOY
- Limpia badges viejos (>7 días)
- Hace commit automático
- (Opcional) Te notifica por Telegram

---

## 🧪 TESTING POST-MIGRACIÓN

### Checklist de Verificación

- [ ] Grafo 3D carga correctamente
- [ ] Se ven ~500 papers
- [ ] Seeds tienen color dorado y badge 📚
- [ ] Badges de fuente (ArXiv/S2) funcionan
- [ ] Filtros funcionan (prueba cada uno)
- [ ] Favoritos funcionan (click en ★)
- [ ] Grid view funciona
- [ ] Click en paper abre URL
- [ ] Estadísticas en header son correctas
- [ ] Workflow diario configurado (cron)

### Pruebas Avanzadas

```bash
# 1. Simular paper "nuevo" manualmente
cd scripts
python << EOF
import json
from datetime import datetime

with open('../docs/graph_data.json', 'r') as f:
    data = json.load(f)

# Marcar el primer paper como "nuevo"
paper = [n for n in data['nodes'] if n['group'] == 'paper'][0]
paper['added_date'] = datetime.now().isoformat().split('T')[0]
paper['newness_level'] = 'today'
paper['newness_badge'] = '🔥 HOY'
paper['newness_color'] = '#00ff88'

with open('../docs/graph_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Paper marcado como nuevo")
EOF

# 2. Commit y push
git add docs/graph_data.json
git commit -m "Test: Marcar paper como nuevo"
git push

# 3. Espera 2 min y recarga la página
# Deberías ver el paper con badge 🔥 HOY en verde brillante
```

---

## 🚨 Rollback (Si algo sale mal)

Si el nuevo sistema no funciona y quieres volver al anterior:

```bash
# 1. Restaurar desde backup
rm -rf docs/ scripts/ config/ .github/
cp -r backup_viejo_sistema/* .

# 2. Restaurar graph_data.json de GitHub
cp backup_github_graph.json docs/graph_data.json

# 3. Commit y push
git add .
git commit -m "Rollback: Restaurar sistema anterior"
git push origin main --force

echo "✅ Sistema restaurado al estado previo"
```

---

## 💡 Tips Post-Migración

### Primeros días:
1. **No te preocupes** si la actualización diaria dice "No se encontraron papers nuevos" los primeros días. Es normal si no hay nada publicado que cumpla tus criterios.

2. **Monitorea los logs** de GitHub Actions durante la primera semana para detectar errores.

3. **Ajusta keywords** en `scripts/config.py` si ves que faltan papers importantes o entran demasiados irrelevantes.

### Optimizaciones:
1. Si el grafo 3D va lento, reduce `max_results` en config.py
2. Si quieres más papers, aumenta los límites temporales
3. Si los badges de novedad duran mucho, cambia `DAILY_UPDATE["new_paper_threshold_days"]` de 7 a 5

---

## ✅ Checklist Final

Antes de dar por completada la migración:

- [ ] Backup del sistema viejo guardado ✅
- [ ] Nuevo sistema en GitHub ✅
- [ ] Llenado inicial ejecutado ✅
- [ ] ~500 papers en el grafo ✅
- [ ] Frontend cargando correctamente ✅
- [ ] Seeds destacados en dorado ✅
- [ ] Workflow diario configurado ✅
- [ ] (Opcional) Telegram funcionando ✅
- [ ] README leído y entendido ✅

---

¡Felicidades! 🎉 Has migrado exitosamente al nuevo sistema de dos fases.

**Siguiente paso**: Siéntate, relájate y deja que el sistema trabaje para ti cada día 🚀
