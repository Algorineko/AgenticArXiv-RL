

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
cd AgenticArxiv
python -m rl.rollout search_01 ../traces/train/
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
| **Reward** | Verificable (tarea exitosa +1.0, herramienta precisa +0.5, error de parseo -0.2, etc.) |
| **Transition** | `execute_tool(action) → observation` |

### Espacio de Acciones (4 herramientas)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Buscar papers en arXiv
2. `download_arxiv_pdf(ref, session_id)` — Descargar PDF
3. `translate_arxiv_pdf(ref, session_id)` — Traducir PDF
4. `get_paper_cache_status(ref, session_id)` — Consultar estado de la caché

### Componentes de Verifiable Reward

| Dimensión | Recompensa | Fuente |
|------|------|------|
| Tarea exitosa | +1.0 | `task_completed` |
| Llamada a herramienta precisa | +0.5 | `tool_call_accurate` (`_check_tool_sequence`) |
| Error de parseo | -0.2 | `parse_failures` |
| Fallo en ejecución de herramienta | -0.3 | `tool_exec_failures` |
| Tiempo de espera agotado | -0.5 | `termination_type == "FORCE_STOP"` |
| Terminación por error | -1.0 | `termination_type == "ERROR"` |

**Clave**: Todas las recompensas son **verificables** (basadas en reglas), sin necesidad de anotación humana → corresponde al marco RLVR (Reinforcement Learning with Verifiable Reward).

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
   python -m rl.train_sft
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
   python -m rl.train_dpo
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
python -m rl.train_grpo
```

**Salida**: Modelo en `./outputs/grpo/final`

**Ventajas**:
- Sin necesidad de reward model (desventaja de DPO: no puede aprender online)
- Sin necesidad de value model (desventaja de PPO: alto consumo de memoria VRAM)
- Adecuado para modelos pequeños (como Qwen2.5-1.5B)

---

## 📂 Estructura de Directorios

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # Paquete Python
│  ├─ agents/                        # Núcleo del Agente
│  │  ├─ base_agent.py              # Bucle ReAct genérico
│  │  ├─ agent_engine.py            # ReActAgent (política RL)
│  │  ├─ prompt_templates.py
│  │  └─ side_effects.py            # Interfaz desacoplada para efectos secundarios
│  ├─ tools/                         # Capa de herramientas (espacio de acciones)
│  │  ├─ tool_registry.py
│  │  ├─ arxiv_tool.py
│  │  ├─ pdf_download_tool.py
│  │  ├─ pdf_translate_tool.py
│  │  └─ cache_status_tool.py
│  ├─ benchmark/                     # ⭐ Fuente de Verifiable Reward
│  │  ├─ metrics.py                 # TaskMetrics, _check_tool_sequence
│  │  ├─ tasks.py                   # BENCHMARK_TASKS (semillas del conjunto de tareas)
│  │  └─ runner.py
│  ├─ rl/                            # ⭐ Núcleo de RL
│  │  ├─ env.py                     # RLEnv + MockArxivEnv
│  │  ├─ policy.py
│  │  ├─ reward.py                  # RewardCalculator
│  │  ├─ trajectory.py              # Trajectory + lectura/escritura JSONL
│  │  ├─ rollout.py
│  │  ├─ tasks.py
│  │  ├─ train_sft.py               # ⭐ Entrenamiento SFT
│  │  ├─ train_dpo.py               # ⭐ Entrenamiento DPO
│  │  └─ train_grpo.py              # ⭐ Entrenamiento GRPO
│  ├─ utils/
│  │  ├─ llm_client.py
│  │  └─ logger.py
│  └─ requirements.txt
├─ traces/                           # Almacenamiento de Trajectory (JSONL)
│  ├─ train/
│  └─ eval/
├─ data/
│  ├─ sft/                           # Dataset SFT
│  ├─ dpo/                           # Dataset DPO
│  └─ mock_arxiv_snapshot.json       # Snapshot del MockEnv
├─ eval/
│  ├─ eval_cases.jsonl
│  └─ badcase_replay.py
├─ scripts/
│  ├─ generate_sft_data.py
│  └─ generate_dpo_data.py
├─ archive/                          # Archivado (aplicación web original)
│  ├─ api/
│  ├─ AgenticArxivWeb/
│  ├─ mcp_protocol/
│  └─ skill_cli/
├─ docs/
│  └─ rl_building.md                # Plan de refactorización completo
├─ .venv/                            # Entorno virtual Python
└─ README.md                         # Este documento
```

---

## 🔬 Ejemplos de Uso

### 1. Rollout (recopilar trajectory)

```bash
cd AgenticArxiv

# Tarea individual
python -m rl.rollout search_01 ../traces/train/

# Rollout por lotes
python -m rl.rollout --all ../traces/train/
```

### 2. Flujo de Entrenamiento (SFT → DPO → GRPO)

```bash
# Paso 1: Generar datos SFT
python scripts/generate_sft_data.py

# Paso 2: Entrenamiento SFT
python -m rl.train_sft

# Paso 3: Generar datos DPO (requiere modelo SFT)
python scripts/generate_dpo_data.py

# Paso 4: Entrenamiento DPO
python -m rl.train_dpo

# Paso 5: Entrenamiento GRPO
python -m rl.train_grpo
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

Provenientes de `benchmark/tasks.py`, contienen 7 tareas:

| ID | Tarea | Tipo | Herramienta Esperada |
|----|------|------|---------|
| `search_01` | Buscar papers de IA de los últimos 7 días | Búsqueda | `get_recently_submitted_cs_papers` |
| `search_02` | Obtener papers de ML de los últimos 3 días | Búsqueda | `get_recently_submitted_cs_papers` |
| `search_03` | Buscar papers de NLP de los últimos 7 días | Búsqueda | `get_recently_submitted_cs_papers` |
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
transformers>=4.35.0
trl>=0.8.0                # TRL (SFT/DPO/GRPO/PPO)
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

**¡Comienza tu viaje de entrenamiento en RL Agentic!** 🚀
