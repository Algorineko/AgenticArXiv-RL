

<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Entorno de Entrenamiento para RL Agentic

> **Entorno de entrenamiento de RL Agentic basado en Agente ReAct + herramientas de arXiv**  
> Soporta ruta de entrenamiento progresivo SFT/DPO/GRPO/PPO, para investigar aprendizaje por refuerzo en Agentes de LLM

---

## 🎯 Posicionamiento del Proyecto

Transforma las tareas de búsqueda/descarga/traducción de papers de arXiv en un **entorno de aprendizaje por refuerzo entrenable**, enfocado en:

1. **Verifiable Reward (Recompensa Verificable)**: Basado en recompensas por reglas (precisión en llamadas a herramientas, completitud de la tarea, errores de parseo, etc.), sin necesidad de anotación humana.
2. **Entrenamiento progresivo**: SFT (Ajuste fino supervisado) → DPO (Optimización directa de preferencias) → GRPO (Optimización de política relativa por grupo) → PPO (Optimización de política proximal).
3. **Ingeniería ligera**: Puro Python + almacenamiento JSONL, sin necesidad de MySQL/FastAPI/frontend, enfocado en entrenamiento offline.

**No es objetivo**: Aplicación de arXiv de nivel producción, UI web, servicio de traducción en tiempo real (estas funcionalidades están archivadas en `archive/`).

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.9+
- LLM API (que soporte formato OpenAI API, como Claude, Gemini, Qwen, etc.)
- Usar entorno virtual `.venv`

### 1️⃣ Clonar el proyecto

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
```

Todos los comandos siguientes deben ejecutarse desde la raíz del repositorio
`AgenticArXiv-RL/`.

### 2️⃣ Configuración del Entorno

**Crear entorno virtual**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

**Instalar dependencias**:
```bash
pip install -r AgenticArxiv/requirements.txt
```

**Configurar LLM API**:
```bash
cat > AgenticArxiv/.env << 'EOF'
# Configuración LLM API
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo

# Opcional: Configuración de rutas PDF
PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
EOF
```

### 3️⃣ Prueba de Rollout

```bash
python -m AgenticArxiv.rl.rollout search_01 traces/train/
```

**Salida esperada**:
```
✅ Rollout de Task search_01 completado
   Reward: 1.50
   Metrics: task_completed=True, tool_call_accurate=True
   Trajectory guardada en: traces/train/rollout_20260621_150000.jsonl
```

---

## 📚 Conceptos Clave

### Diseño del MDP

| Dimensión | Definición |
|------|------|
| **State** | Descripción de la tarea + historial de diálogo + resultados de herramientas |
| **Action** | 4 herramientas (búsqueda/descarga/traducción/consulta de caché de arxiv) + FINISH |
| **Reward** | Recompensa verificable multigranular de cinco componentes (format / tool / argument / process / outcome, ver más abajo) |
| **Transition** | `execute_tool(action) → observation` (`MockArxivEnv` con replay de snapshot offline, determinista y reproducible) |

### Espacio de Acciones (4 herramientas)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Buscar papers en arXiv
2. `download_arxiv_pdf(ref, session_id)` — Descargar PDF
3. `translate_arxiv_pdf(ref, session_id)` — Traducir PDF
4. `get_paper_cache_status(ref, session_id)` — Consultar estado de la caché

### Componentes de Verifiable Reward

**Recompensa verificable multigranular de cinco componentes** (`rl/reward.py`, inspirada en la recompensa jerárquica de LLM-TIR). Cada componente se normaliza a `[-1, 1]` y se combina como suma ponderada dividida por la suma de pesos:

| Componente | Peso por defecto | Señal |
|------|:---:|------|
| `format` (formato) | 1 | Fracción de pasos cuya acción es una llamada JSON válida o un token de terminación |
| `tool` (secuencia de herramientas) | 3 | **LCS-F1** sensible al orden entre las secuencias predicha y esperada (`benchmark/metrics.py` coincidencia estricta) |
| `argument` (parámetros) | 2 | Recall de claves de parámetros × exactitud del valor; se omite automáticamente cuando la tarea no tiene `expected_tool_args` |
| `process` (proceso) | 1 | Crédito por pasos válidos menos penalizaciones por fallos de parseo/ejecución y llamadas innecesarias |
| `outcome` (resultado) | 3 | Completado correcto +1, completado con ruta de herramientas errónea +0.25, detención forzosa −0.5, error −1 |

**Aprendizaje curricular**: durante los primeros 30 pasos de entrenamiento los pesos de `tool` / `argument` / `outcome` se multiplican por 1/3 (primero el protocolo ReAct, después la semántica); desde el paso 30 todos los pesos están activos (`RewardCalculator.schedule`).

**Clave**: Todas las recompensas son **verificables** (basadas en reglas), sin necesidad de anotación humana → corresponde al marco RLVR (Reinforcement Learning with Verifiable Reward). Cada trayectoria guarda un desglose `reward_components` para auditar y detectar reward hacking.

---

## 🛠️ Ruta de Entrenamiento (SFT → DPO → GRPO)

### Fase 1: SFT (Ajuste Fino Supervisado)

**Objetivo**: Enseñar al modelo el formato básico de llamadas a herramientas.

**Pasos**:
1. Generar demostraciones de expertos:
   ```bash
   python scripts/generate_sft_data.py
   ```
2. Entrenar:
   ```bash
   python -m AgenticArxiv.rl.train_sft
   ```
3. Salida: Modelo en `./outputs/sft/final`

**Formato de datos** (`data/sft/sft_train.jsonl`):
```json
{
  "messages": [
    {"role": "system", "content": "Eres un Agente de búsqueda de papers de arXiv..."},
    {"role": "user", "content": "Busca papers de IA de los últimos 7 días"},
    {"role": "assistant", "content": "{\"name\":\"get_recently_submitted_cs_papers\",\"arguments\":{...}}"}
  ]
}
```

---

### Fase 2: DPO (Optimización Directa de Preferencias)

**Objetivo**: Que el modelo prefiera la selección correcta de herramientas y rechace rutas erróneas.

**Pasos**:
1. Realizar rollout con el modelo SFT para recolectar pares chosen/rejected:
   ```bash
   python scripts/generate_dpo_data.py
   ```
2. Entrenar:
   ```bash
   python -m AgenticArxiv.rl.train_dpo
   ```
3. Salida: Modelo en `./outputs/dpo/final`

**Formato de datos** (`data/dpo/dpo_train.jsonl`):
```json
{
  "prompt": "Busca papers de IA de los últimos 7 días",
  "chosen": "{\"name\":\"get_recently_submitted_cs_papers\",...}",
  "rejected": "{\"name\":\"download_arxiv_pdf\",...}"
}
```

---

### Fase 3: GRPO (Optimización de Política Relativa por Grupo)

**Objetivo**: Entrenamiento online usando verifiable reward, sin necesidad de value model.

**Pasos**:
```bash
python -m AgenticArxiv.rl.train_grpo
```

**Salida**: Modelo en `./outputs/grpo/final`

**Ventajas**:
- Sin necesidad de reward model (desventaja de DPO: no puede aprender online)
- Sin necesidad de value model (desventaja de PPO: alto consumo de VRAM)
- Adecuado para modelos pequeños (como Qwen2.5-1.5B)

**Puntuación de recompensa** (`rl/grpo_reward.py`): el `completion` de un solo paso se parsea como acción ReAct, se ejecuta con `MockArxivEnv` y se completa hasta formar una "trayectoria mínima completa" antes de que la puntúe el `RewardCalculator` de cinco componentes — el mismo estándar que usan rollout y benchmark, sin una segunda definición de recompensa.

### Garantías de calidad del entrenamiento (verificación automática)

El pipeline de entrenamiento incorpora varias capas de validación automática que convierten los "fallos de entrenamiento silenciosos" en errores sonoros:

- **Chequeo de longitud de generación**: antes de entrenar comprueba que `max_completion_length` pueda contener la acción canónica, evitando vueltas con gradientes nulos en las que el modelo nunca emite una acción completa
- **Guardia de varianza nula**: interrumpe el entrenamiento (con sugerencias de solución) cuando la varianza de recompensa dentro del grupo se mantiene en 0, es decir, todas las ventajas son cero (`RewardVarianceGuard`)
- **Evaluación canary**: muestrea sobre tareas fijas cada N pasos y detiene antes de tiempo si el desempeño degrada bajo el umbral repetidamente (`CanaryCallback`)
- **Verificación por etapa**: el modelo saliente de cada fase debe superar un umbral mínimo de calidad — tasa de parseo SFT ≥ 0.3, reward promedio DPO ≥ −0.3, reward promedio GRPO ≥ −0.2 (`StageVerifier`, se omite con `--no-verify`)
- **Precisión mixta adaptativa**: bf16 primero en CUDA, con respaldo a fp16, desactivada en CPU / MPS (`rl/precision.py`)
- **Validación del backend de registro**: un backend indicado en `--report_to` que no esté instalado falla antes de cargar el modelo, para no terminar un entrenamiento y descubrir que no hay ninguna curva (`rl/observability.py`)

---

## 📂 Estructura de Directorios

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # ⭐ Paquete Python (entorno de entrenamiento RL)
│  ├─ agents/                        # Núcleo del Agente
│  │  ├─ base_agent.py              # Bucle ReAct genérico
│  │  ├─ agent_engine.py            # ReActAgent (política RL)
│  │  ├─ context_manager.py
│  │  ├─ prompt_templates.py
│  │  └─ side_effects.py           # Interfaz desacoplada para efectos secundarios
│  ├─ tools/                         # Capa de herramientas (espacio de acciones)
│  │  ├─ tool_registry.py          # Registro de herramientas
│  │  ├─ arxiv_tool.py             # Búsqueda en arXiv
│  │  ├─ pdf_download_tool.py      # Descarga de PDF
│  │  ├─ pdf_translate_tool.py     # Traducción de PDF
│  │  └─ cache_status_tool.py      # Consulta de caché
│  ├─ benchmark/                     # ⭐ Fuente de Verifiable Reward
│  │  ├─ metrics.py               # TaskMetrics, coincidencia estricta de herramientas y parámetros
│  │  ├─ tasks.py                 # BENCHMARK_TASKS (8 semillas de tareas)
│  │  ├─ runner.py                 # Ejecutor de benchmarks
│  │  ├─ run_benchmark.py          # Entrada de benchmark por CLI
│  │  └─ report.py                 # Informe de métricas
│  ├─ rl/                            # ⭐ Núcleo de RL
│  │  ├─ train_sft.py              # ⭐ Entrenamiento SFT
│  │  ├─ train_dpo.py              # ⭐ Entrenamiento DPO
│  │  ├─ train_grpo.py             # ⭐ Entrenamiento GRPO (con guardias de entrenamiento)
│  │  ├─ env.py                    # RLEnv + MockArxivEnv (entorno de snapshot offline)
│  │  ├─ reward.py                 # RewardCalculator (recompensa de 5 componentes + currículum)
│  │  ├─ grpo_reward.py            # Adaptador de recompensa GRPO (completion de un paso → trayectoria)
│  │  ├─ rollout.py                # Recopilación de datos de rollout offline
│  │  ├─ trajectory.py             # Trajectory + lectura/escritura JSONL
│  │  ├─ build_snapshot.py         # Genera snapshot offline de arXiv (único paso con red)
│  │  ├─ canary.py                 # Evaluación periódica en entrenamiento (detención temprana)
│  │  ├─ stage_verifier.py         # Verificación de umbral de calidad por fase
│  │  ├─ precision.py              # Estrategia de precisión mixta (bf16/fp16/CPU)
│  │  └─ observability.py          # Backends de registro + curvas por componente
│  ├─ models/                        # Capa de almacenamiento (store_memory para RL, store_mysql para Web)
│  ├─ services/                      # Servicios de efectos secundarios (event_bus / log / runtime)
│  ├─ api/ · mcp_protocol/ · skill_cli/   # Capas de compatibilidad Web / MCP / Skill archivadas
│  ├─ utils/                         # llm_client, logger, utilidades PDF
│  ├─ tests/                         # 16 tests unitarios (unittest)
│  └─ requirements.txt
├─ scripts/                          # Generación de datos
│  ├─ generate_sft_data.py          # Trayectorias expertas con LLM API
│  └─ generate_dpo_data.py          # Pares de preferencia muestreando el modelo SFT local
├─ docs/
│  ├─ rl_building.md               # Plan de refactorización completo
│  ├─ multigranular_rl.md         # Diseño de recompensa multigranular (5 componentes + currículum)
│  └─ metric_stats.md            # Plan de estadísticas/métricas
├─ data/                             # Datasets (sft/ y dpo/ están en .gitignore — generarlos primero)
│  ├─ sft/                           # Dataset SFT (JSONL)
│  ├─ dpo/                           # Pares DPO (JSONL)
│  └─ mock_arxiv_snapshot.json       # Snapshot del MockEnv
├─ traces/                           # Almacenamiento de Trajectory (JSONL, gitignored)
├─ archive/                          # Archivado (app web original: PDFMathTranslate / arxiv-api / weather-agent)
├─ AgenticArxivWeb/                  # Frontend Vue3 original (archivado)
├─ bin/ · Makefile · Overview.md     # Scripts de arranque Web heredados y docs (a modernizar)
└─ README.md / README.en.md / README.es-ES.md   # 🇨🇳 🇬🇧 🇪🇸
```

---

## 🔬 Ejemplos de Uso

### 1. Rollout (recopilar trajectory)

```bash
# Tarea individual
python -m AgenticArxiv.rl.rollout search_01 traces/train/

# Rollout por lotes
python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/
```

### 2. Flujo de Entrenamiento (SFT → DPO → GRPO)

```bash
# Paso 1: Generar datos SFT
python scripts/generate_sft_data.py

# Paso 2: Entrenamiento SFT
python -m AgenticArxiv.rl.train_sft

# Paso 3: Generar datos DPO (requiere modelo SFT)
python scripts/generate_dpo_data.py

# Paso 4: Entrenamiento DPO
python -m AgenticArxiv.rl.train_dpo

# Paso 5: Entrenamiento GRPO
python -m AgenticArxiv.rl.train_grpo
```

### 3. Prueba de cálculo de Reward

```python
from rl.reward import RewardCalculator
from benchmark.tasks import get_task_by_id

task_def = get_task_by_id('search_01')
# Construir un resultado mock
result = {
    'history': [
        {'thought': '...', 'action': '...', 'observation': '...'},
        {'thought': '...', 'action': 'FINISH', 'observation': '...'},
    ],
    'timing': {...},
    'token_usage': {...},
    'iteration_count': 2,
}

reward_calc = RewardCalculator()
reward, metrics = reward_calc.compute_reward(task_def, result)
print(f'Reward: {reward:.2f}')  # Esperado: ~1.5
```

---

## 🧪 Conjunto de Tareas de Prueba

Provenientes de `benchmark/tasks.py`, contienen 8 tareas:

| ID | Tarea | Tipo | Herramienta Esperada |
|----|------|------|---------|
| `search_01` | Buscar papers de IA de los últimos 7 días | Búsqueda | `get_recently_submitted_cs_papers` |
| `search_02` | Obtener papers de ML de los últimos 3 días | Búsqueda | `get_recently_submitted_cs_papers` |
| `search_03` | Buscar papers de NLP de los últimos 7 días | Búsqueda | `get_recently_submitted_cs_papers` |
| `search_04` | Buscar papers de todas las categorías de ciencias de la computación de los últimos 7 días | Búsqueda | `get_recently_submitted_cs_papers` |
| `download_01` | Descargar PDF del 1er paper | Descarga | `download_arxiv_pdf` |
| `translate_01` | Traducir el 1er paper | Traducción | `translate_arxiv_pdf` |
| `cache_01` | Ver estado de caché del 1er paper | Caché | `get_paper_cache_status` |
| `composite_01` | Búsqueda + Descarga | Compuesta | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

---

## 📊 Monitoreo de Métricas

### Curva de Reward

Monitorear con TensorBoard o wandb:
```bash
tensorboard --logdir ./outputs/grpo/logs
```

### Métricas Clave

| Métrica | Descripción | Objetivo |
|------|------|------|
| `reward` | Recompensa promedio | ↑ Incremento |
| `kl_div` | Divergencia KL (vs modelo de referencia) | ↔ Estable (no excesiva) |
| `task_completed_rate` | Tasa de éxito de la tarea | ↑ Incremento |
| `tool_call_accurate_rate` | Tasa de precisión en llamadas a herramientas | ↑ Incremento |
| `parse_failures` | Número de fallos de parseo | ↓ Disminución |
| `tool_exec_failures` | Número de fallos en ejecución de herramientas | ↓ Disminución |

---

## 🛡️ Notas sobre Dependencias

**Dependencias principales** (`requirements.txt`):
```txt
torch>=2.0.0
transformers>=4.45.0
trl>=0.20.0               # TRL (SFT/DPO/GRPO), verificado en 0.29.1
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**Ya no necesario** (eliminado):
- `fastapi`, `uvicorn` (sin servicio web)
- `sqlalchemy`, `pymysql` (cambiado a JSONL)
- `pdf2zh` (usa mock durante el entrenamiento)

---

## 🔗 Recursos Relacionados

### Documentación Oficial
- [TRL 文档](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### Papers
- **InstructGPT** (OpenAI, 2022): Tres fases de RLHF (SFT → RM → PPO)
- **DPO** (Stanford, 2023): Optimización directa de preferencias
- **RLVR**: Reinforcement Learning with Verifiable Reward

### AgenticArXiv Original (versión Web App)
Este proyecto se basa en [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv), la versión original incluye:
- Backend FastAPI + frontend Vue3
- Tres arquitecturas de Agente (ReAct/MCP/Skill)
- Push SSE en tiempo real, almacenamiento MySQL, servicio de traducción de PDF

Estas funcionalidades están archivadas en `archive/`.

---

## 🤝 Contribuir

¡Son bienvenidos los Issues y Pull Requests!

### Recomendaciones de Desarrollo
1. Haz fork de este repositorio
2. Crea una rama feature: `git checkout -b feature/your-feature`
3. Commitea los cambios: `git commit -m "feat: add your feature"`
4. Empuja la rama: `git push origin feature/your-feature`
5. Envía el Pull Request

---

## 📄 Licencia

Licencia MIT

---

## 🙋 FAQ

### Q: ¿Diferencias con el AgenticArXiv original?

| Dimensión | AgenticArXiv Original | Este Proyecto (AgenticArXiv-RL) |
|------|------------------|-------------------------|
| **Enfoque** | Aplicación arXiv de nivel producción | Entorno de investigación para entrenamiento RL |
| **Arquitectura** | FastAPI + Vue3 + MySQL | Puro Python + JSONL |
| **Modo Agente** | 3 tipos (ReAct/MCP/Skill) | Solo ReAct (simplificado) |
| **Funcionalidades clave** | Traducción en tiempo real, SSE, UI web | Entrenamiento SFT/DPO/GRPO |
| **Dependencias** | Pesadas (14+ paquetes) | Livianas (8 paquetes principales) |

### Q: ¿Por qué se mantiene solo ReAct y se archiva MCP/Skill?

El entrenamiento RL se enfoca en una política única (parseo regex de ReAct). MCP/Skill añaden complejidad sin cambiar la lógica central.

### Q: ¿Por qué cambiar a JSONL en lugar de MySQL?

- **Portabilidad**: JSONL no requiere dependencias de base de datos
- **Ligero**: Más adecuado para escenarios offline de entrenamiento RL
- **Compatibilidad TRL**: Los datasets de TRL soportan JSONL directamente

### Q: ¿Por qué elegir GRPO y no PPO?

GRPO es más adecuado para proyectos de aprendizaje ligeros:
- ✅ Sin necesidad de value model adicional (menor consumo de VRAM/costo de entrenamiento)
- ✅ Adecuado para modelos pequeños (como Qwen2.5-1.5B)
- ✅ Implementación simple, fácil de depurar

PPO es más adecuado para entrenamiento de modelos grandes de nivel producción (7B+), este proyecto como demo de aprendizaje no lo cubre.

---

## 📝 TODO (Hoja de Ruta de Desarrollo)

Ordenado por prioridad. ¡Las contribuciones son bienvenidas (ver 🤝 Contribuir)!

### P0 — Corto plazo (cerrar brechas clave)

- [x] **Rollout Agentic Multiturno**: implementado el muestreo real «actuar → observar → actuar», con un `MockArxivEnv` independiente por generación; los tokens del entorno usan `env_mask=0`, mientras que la trayectoria completa del asistente participa en la optimización GRPO y en la recompensa de cinco componentes.
- [x] **Observabilidad del entrenamiento**: SFT / DPO / GRPO comparten un mismo `--report_to` (none / auto / tensorboard / wandb); si el backend no está instalado falla de inmediato en vez de no registrar nada en silencio. Además de reward / kl / grad_norm / `frac_reward_zero_std` que ya trae TRL, se registran por separado los cinco componentes de la recompensa (format/tool/argument/process/outcome) y los pesos actuales del currículo —el currículo cambia los pesos durante los primeros 30 pasos, así que la recompensa total por sí sola no distingue «la política mejoró» de «los pesos se movieron»—, más `rollout/` turns / finished / parse_error_rate / tool_error_rate para diagnosticar el propio muestreo multiturno.

### P1 — Medio plazo (datos y evaluación)

- [ ] **Ampliar el conjunto de tareas**: `benchmark/tasks.py` tiene solo 8 tareas; ampliarlo a 50+ y derivar automáticamente `expected_tools` / `expected_tool_args`.
- [ ] **eval/ badcase replay (reproducción de casos malos)**: el directorio `eval/` del árbol no existe aún; implementar `eval_cases.jsonl` + `badcase_replay.py` para cerrar el bucle.
- [ ] **Investigación de reward hacking**: ampliar `RewardVarianceGuard` / `CanaryCallback` con una biblioteca de casos de reward hacking y ajuste de pesos del currículum.

### P2 — Rendimiento y escala

- [ ] **Muestreo acelerado con vLLM**: sustituir HF generate para aumentar el throughput del rollout (la prioridad sube cuando aterrice el rollout multiturno).
- [ ] **Soporte multi-GPU**: configuración accelerate / FSDP (accelerate ya es dependencia, pero sin configurar).

### P3 — Largo plazo (evolución algorítmica)

- [ ] **Mejoras estilo DAPO**: clip-higher, dynamic sampling, filtro de overlong, loss a nivel de token (loss/clip viven dentro de TRL; requieren un fork o sobrescribir `compute_loss`).
- [ ] **Framework de entrenamiento asíncrono**: migrar a verl `fully_async_policy` / AReaL para alojar SAO (abajo).

### 🔭 SAO: el algoritmo RL agéntico asíncrono de próxima generación

> **SAO (Single-Rollout Asynchronous Optimization; Optimización Asíncrona de Rollout Único)** fue propuesto por el KEG Lab de la Universidad de Tsinghua (2026-07) como evolución de GRPO para el entrenamiento **agéntico asíncrono**. Motivación: el rollout es el cuello de botella en tareas agénticas de largo horizonte, y el muestreo por grupos de GRPO se vuelve off-policy e inestable bajo asincronía (normalmente colapsa en menos de 200 pasos).
>
> Componentes técnicos clave:
> 1. **Muestreo de rollout único**: una trayectoria por prompt, consumida en cuanto llega, en lugar de comparación por grupos;
> 2. **DIS (Direct Bilateral Importance Sampling)**: calcula `r_t = π_θ / π_rollout` a partir de los log-probs por token registrados durante el rollout y **enmascara los tokens fuera del intervalo de confianza `[1−ε_l, 1+ε_h]`** (no como el recorte unilateral de PPO);
> 3. **Actualizaciones desacopladas del value model**: el value model se actualiza dos veces por cada actualización de política (1:2), con las **capas de atención congeladas** durante su entrenamiento (solo se entrenan las proyecciones MoE);
> 4. **GAE con omisión de observaciones**: las ventajas se propagan solo entre los tokens generados por el modelo, omitiendo los tokens de observación del entorno para filtrar ruido.
>
> Resultados: entrenamiento estable durante ~1000 pasos; **97.3%** en AIME2025 (vs 84.2% en GRPO), 29.8% en SWE-Bench Verified; ya usado para entrenar GLM-5.2 (750B).
>
> Ruta de adopción: cerrar la brecha del rollout multiturno de P0 → introducir la omisión de observaciones y el recorte bilateral DIS → migrar a verl `fully_async_policy` (`gen_batch_size=1` / `staleness_threshold` / TIS a nivel de token, alineado con SAO) o AReaL v1.0 para entrenamiento totalmente asíncrono + value model.
>
> 📄 **Paper**: [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (arXiv:2607.07508)](https://arxiv.org/abs/2607.07508) (Tsinghua KEG; código oficial aún no liberado)

---

**¡Comienza tu viaje de entrenamiento en RL Agentic!** 🚀
