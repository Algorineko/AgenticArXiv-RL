

<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Entorno de Entrenamiento para RL Agentic

> **Entorno de entrenamiento de RL Agentic basado en Agente ReAct + herramientas de arXiv**  
> Soporta ruta de entrenamiento progresivo SFT/DPO/GRPO/PPO, además de una ruta opcional de OPD (destilación on-policy), para investigar aprendizaje por refuerzo en Agentes de LLM

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="Descripción general del proyecto AgenticArXiv-RL" width="800"/>
</p>

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

**Ejemplo de salida (la recompensa varía según la trayectoria y está en `[-1, 1]`)**:
```
✅ Rollout de Task search_01 completado
   Reward: 1.00
   Metrics: task_completed=True, tool_call_accurate=True
   Trajectory guardada en: traces/train/rollout_20260621_150000.jsonl
```

---

## 📚 Conceptos Clave

### Diseño del MDP

| Dimensión | Definición |
|------|------|
| **State** | Descripción de la tarea + historial de diálogo + resultados de herramientas |
| **Action** | 8 herramientas (navegación/búsqueda por palabras clave/descarga/traducción/consulta de caché/lectura de papers/resumen de papers/extracción de figuras) + FINISH |
| **Reward** | Recompensa verificable multigranular de cinco componentes (format / tool / argument / process / outcome, ver más abajo) |
| **Transition** | `execute_tool(action) → observation` (`MockArxivEnv` con replay de snapshot offline, determinista y reproducible) |

### Espacio de Acciones (8 herramientas)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Buscar papers en arXiv
2. `download_arxiv_pdf(ref, session_id)` — Descargar PDF
3. `translate_arxiv_pdf(ref, session_id)` — Traducir PDF
4. `get_paper_cache_status(ref, session_id)` — Consultar estado de la caché
5. `search_arxiv_papers(query, max_results, days=None)` — Buscar por palabra clave, título o autor
6. `get_paper_content(ref, session_id, section=None)` — Leer el resumen o una sección concreta de un paper descargado
7. `summarize_paper(ref, style, max_words)` — Resumen del lado del entorno sobre un paper descargado (tldr / structured / bullet)
8. `extract_paper_figures(ref)` — Extraer los archivos de figuras y sus captions de un paper descargado

> El bucle de interpretación «buscar → descargar → leer → resumir → extraer figuras» ya está conectado. El **análisis** de figuras (T5, requiere un VLM del lado del entorno) sigue siendo una propuesta de diseño; ver «🧰 Diseño de Evolución del Conjunto de Herramientas» más abajo.

### Componentes de Verifiable Reward

**Recompensa verificable multigranular** (`rl/reward.py`, inspirada en la recompensa jerárquica de LLM-TIR). Los cinco componentes principales se normalizan a `[-1, 1]` y conservan los pesos del currículo; además se registran dos componentes de diagnóstico, calidad del resultado y eficiencia, y los fallos graves reciben un techo de recompensa no compensable:

| Componente | Peso por defecto | Señal |
|------|:---:|------|
| `format` (formato) | 1 | Fracción de pasos cuya acción es una llamada JSON válida o un token de terminación |
| `tool` (secuencia de herramientas) | 3 | **LCS-F1** sensible al orden entre las secuencias predicha y esperada (`benchmark/metrics.py` coincidencia estricta) |
| `argument` (parámetros) | 2 | Recall de claves de parámetros × exactitud del valor; se omite automáticamente cuando la tarea no tiene `expected_tool_args` |
| `process` (proceso) | 1 | Crédito por pasos válidos menos penalizaciones por fallos de parseo/ejecución y llamadas innecesarias |
| `outcome` (resultado) | 3 | Completado correcto +1, completado con ruta de herramientas errónea +0.25, detención forzosa −0.5, error −1 |

| `result_quality` (calidad del resultado) | diagnóstico / gate | Comprueba que la observation de la herramienta represente de verdad un resultado exitoso; los resultados vacíos, los fallbacks y los errores de ejecución disparan una puerta negativa |
| `efficiency` (eficiencia) | diagnóstico / gate | Normaliza llamadas redundantes y reintentos fallidos contra los pasos estándar de la tarea; una redundancia grave dispara un techo de recompensa |

**Aprendizaje curricular**: durante los primeros 30 pasos de entrenamiento los pesos de `tool` / `argument` / `outcome` se multiplican por 1/3 (primero el protocolo ReAct, después la semántica); desde el paso 30 todos los pesos están activos (`RewardCalculator.schedule`).

**Fallos no compensables**: los fallos de parseo, los fallos de ejecución de herramientas y los resultados desconocidos obtienen como mucho una recompensa negativa; un FINISH falso no puede compensarse con puntos de formato. El `FINISH` que el GRPO de un solo paso añade automáticamente para construir la trayectoria completa se marca como `forced_finish` y no recibe la bonificación terminal completa.

**Comprobación del resultado de T5**: la observación de `analyze_figure` debe analizarse por completo como JSON o literal de Python, con `paper_id` y `answer` como cadenas no vacías tras quitar espacios. Una respuesta vacía, ausente, de otro tipo o truncada recibe `-1` en ese paso; sigue siendo válida una explicación no vacía de que el caption carece de la información solicitada. Cualquier paso de análisis de figuras sin tal respuesta válida activa la puerta de fallo grave y limita la recompensa total a `-0.75`, incluso en tareas de varias herramientas donde el promedio podría ocultar ese fallo. Solo se comprueba la presencia de la respuesta, no la exactitud del texto del VLM; las demás herramientas conservan sus reglas.

**Clave**: Todas las recompensas son **verificables** (basadas en reglas), sin necesidad de anotación humana → corresponde al marco RLVR (Reinforcement Learning with Verifiable Reward). Cada trayectoria guarda un desglose `reward_components` para auditar y detectar reward hacking.

### Aislamiento de Rollout

`RolloutSandbox`, en `rl/sandbox.py`, registra antes de empezar una trayectoria el estado base del entorno, del Store en memoria y de los directorios de artefactos indicados, y restaura ese estado eliminando los archivos nuevos antes de la siguiente trayectoria. El GRPO multiturno, el rollout normal y el benchmark offline usan el mismo contrato de reset, de modo que los resultados de búsqueda, la caché de descargas, el estado de traducción y los contadores no se encadenan entre trayectorias.

---

## 🛠️ Ruta de Entrenamiento (SFT → DPO → GRPO / OPD)

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

> **Pesos publicados** (SFT a parámetros completos sobre Qwen2.5-1.5B, 2 épocas con 2628 trayectorias expertas parametrizadas, loss 0.079):
> [🤗 ModelScope · AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT)
> Este checkpoint cubre solo las 8 primeras herramientas. `analyze_figure` (T5) se añadió después; ahora se pueden generar datos expertos de T5 de forma opcional, pero este modelo publicado no se entrenó con ellos.

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

Este comando carga directamente el modelo local de Hugging Face en `outputs/sft/final`
y muestrea varias veces, así que no necesita `LLM_API_KEY`. Parámetros opcionales
habituales:

```bash
python scripts/generate_dpo_data.py \
  --model outputs/sft/final \
  --num_rollouts_per_task 8 \
  --temperature 0.8 \
  --seed 42
```

Si ya existe `data/mock_arxiv_snapshot.json`, las llamadas a herramientas usan
automáticamente el replay offline, lo que hace reproducible la generación de datos;
en caso contrario se recurre a la red en vivo. Solo forman un par de preferencia las
trayectorias cuya diferencia de recompensa supera `--min_reward_gap` (0.05 por defecto)
y cuya primera acción de herramienta es distinta.
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

> **Pesos publicados** (GRPO online sobre el modelo SFT de la fase 1; el conjunto de entrenamiento son 33 tareas con señal, seleccionadas por un sondeo de política congelada entre las 51 tareas de entrenamiento, con repetición offline del snapshot):
> [🤗 ModelScope · AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO)
> Evaluación offline (seed 45 / repeat 3), tasa de éxito estricta SFT → GRPO: rl_train 0.081 → 0.636, dev 0.000 → 0.375, iid_test 0.056 → 0.444, ood_test 0.000 → 0.500.

**Ventajas**:
- Sin necesidad de reward model (desventaja de DPO: no puede aprender online)
- Sin necesidad de value model (desventaja de PPO: alto consumo de VRAM)
- Adecuado para modelos pequeños (como Qwen2.5-1.5B)

**Rollout multiturno y puntuación de recompensa** (`rl/grpo_reward.py`): en cada turno la política actual genera una acción ReAct, un `MockArxivEnv` independiente la ejecuta y la observación se reinserta en el contexto, hasta FINISH, un fallo de parseo o `--max_turns`. Los tokens del asistente entran en la pérdida GRPO y los tokens del entorno solo hacen de contexto vía `env_mask=0`; la trayectoria completa la puntúa el `RewardCalculator` de cinco componentes, el mismo estándar que usan rollout y benchmark.

```bash
python -m AgenticArxiv.rl.build_snapshot
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --max_turns 4

# Registrar curvas de entrenamiento (el mismo parámetro en todas las fases)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --report_to tensorboard
tensorboard --logdir outputs/grpo/logs

# Opciones de la familia DAPO (los valores por defecto no cambian nada, así que los experimentos históricos siguen comparables)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --dapo
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --epsilon_high 0.2   # solo clip-higher

# Multi-GPU (DDP; lanzar accelerate con el intérprete que tiene las dependencias de entrenamiento)
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --no-qlora
```

**Multi-GPU**: `configs/accelerate/` incluye una configuración DDP y otra FSDP. Los scripts de entrenamiento detectan el `LOCAL_RANK` que exporta `accelerate launch` y entonces dejan de anclar una sola GPU (anclarla ocultaría los dispositivos de los demás ranks); en ejecución mono-proceso siguen anclando, lo que evita que el Trainer caiga a `DataParallel` y provoque un segfault. El `device_map={"": 0}` de QLoRA es incompatible con el arranque multiproceso, así que esa combinación falla de forma explícita en lugar de entrenar en silencio con una sola tarjeta. FSDP necesita `torch>=2.6`.

**Curvas de entrenamiento** (`rl/observability.py`): `--report_to` acepta `none` / `auto` / `tensorboard` / `wandb` (separados por comas), compartido por las cinco fases (SFT / DPO / GRPO / OPD / PPO). Además de las métricas que ya trae TRL, se registran:

| Grupo de métricas | Contenido | Por qué se registra aparte |
|---|---|---|
| `reward_components/*` | format / tool / argument / process / outcome | cada uno acotado a `[-1,1]` e independiente de los pesos |
| `reward_weights/*` | pesos actuales del currículo | el currículo reduce tool/argument/outcome los primeros 30 pasos, así que mirar solo la recompensa total confunde «se abrieron los pesos» con «la política retrocedió» |
| `rollout/*` | turns / finished / parse_error_rate / tool_error_rate | cuando la recompensa cae, separa «la política retrocedió» de «nunca aprendió a terminar y agota max_turns» |

### Garantías de calidad del entrenamiento (verificación automática)

El pipeline de entrenamiento incorpora varias capas de validación automática que convierten los "fallos de entrenamiento silenciosos" en errores sonoros:

- **Chequeo de longitud de generación**: antes de entrenar comprueba que `max_completion_length` pueda contener la acción canónica, evitando vueltas con gradientes nulos en las que el modelo nunca emite una acción completa
- **Guardia de varianza nula**: interrumpe el entrenamiento (con sugerencias de solución) cuando la varianza de recompensa dentro del grupo se mantiene en 0, es decir, todas las ventajas son cero (`RewardVarianceGuard`)
- **Evaluación canary**: muestrea sobre tareas fijas cada N pasos y detiene antes de tiempo si el desempeño degrada bajo el umbral repetidamente (`CanaryCallback`)
- **Verificación por etapa**: el modelo saliente de cada fase debe superar un umbral mínimo de calidad — tasa de parseo SFT ≥ 0.3, reward promedio DPO ≥ −0.3, reward promedio GRPO ≥ −0.2 (`StageVerifier`, se omite con `--no-verify`)
- **Precisión mixta adaptativa**: bf16 primero en CUDA, con respaldo a fp16, desactivada en CPU / MPS (`rl/precision.py`)
- **Validación del backend de registro**: un backend indicado en `--report_to` que no esté instalado falla antes de cargar el modelo, para no terminar un entrenamiento y descubrir que no hay ninguna curva (`rl/observability.py`)

### Fase 3': OPD (On-Policy Distillation, opcional, intercambiable con GRPO)

**Objetivo**: destilar la capacidad de acción ReAct del estudiante a partir de la señal por token de un profesor fuerte, sin depender de ninguna recompensa durante el entrenamiento.

**Pasos**:
```bash
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final --teacher Qwen/Qwen2.5-7B-Instruct

# OPD agéntico multi-turno: ejecutar Action → Observation → Action
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final \
  --teacher Qwen/Qwen2.5-7B-Instruct --max_turns 4 \
  --snapshot data/mock_arxiv_snapshot.json
```

**Salida**: modelo `./outputs/opd/final`

**Posicionamiento**: OPD es un **paradigma de entrenamiento completo**, no un simple truco — el estudiante muestrea on-policy sobre los prompts de las tareas, el profesor puntúa cada token con sus logprobs, y la pérdida es la KL reversa `D_KL(π_estudiante ‖ π_profesor)` (mode-seeking). Frente a GRPO es la "ruta del profesor" frente a la "ruta de la recompensa":

| Dimensión | GRPO | OPD |
|---|---|---|
| Señal de aprendizaje | verifiable reward de cinco componentes (dispersa, a nivel de trayectoria) | logprobs por token del profesor (densa) |
| Modelo extra | ninguno | modelo profesor (necesita pesos locales para los logprobs; las APIs externas no exponen distribuciones por token) |
| Techo de rendimiento | puede explorar más allá del profesor | converge al comportamiento del profesor |
| Mejor cuando | hay recompensa verificable y no hay profesor | hay un profesor fuerte y se quiere ahorrar el costo de exploración de RL |

OPD también puede usarse como truco: warm start antes de RL, regularización teacher-KL dentro de RL, o el PG-OPD de verl (la KL reversa tratada como recompensa para el gradiente de política).

**Modos single-turn y multi-turn**: `--max_turns 1` (predeterminado) conserva el comportamiento original prompt/completion de GKDTrainer. `--max_turns > 1` activa OPD agéntico: cada trayectoria recibe un `MockArxivEnv` independiente, se ejecutan las herramientas y la Observation real se inserta en el turno siguiente. Las etiquetas del prompt y de Observation son `-100`; reverse-KL se calcula solo sobre tokens assistant generados por el estudiante. La primera versión exige vocabularios token-to-id idénticos y falla antes de entrenar si difieren. Estudiante y profesor comparten GPU (1.5B + 7B necesitan unos 17GB solo para pesos bf16; gradientes, optimizador y logits requieren memoria adicional, por lo que se recomienda un modelo menor en una GPU de 32GB). Verificado con trl 1.5.1 (`trl.experimental.gkd`); `beta=1.0` selecciona reverse-KL. Canary y la verificación por etapa siguen compartiéndose con GRPO.

**Comparación offline**: ejecuta SFT, `--max_turns 1`, `--max_turns > 1` y GRPO con el mismo snapshot y task set, y compara los resultados comunes de `StageVerifier` en `final/verification_report.json` de cada directorio de salida. Esas recompensas solo se usan para evaluación y nunca entran en la pérdida OPD.

### Fase 4: PPO (Optimización de Política Proximal) — ⚠️ no disponible con las dependencias actuales

**Objetivo**: Ajustar en línea la política y la red de valor con una arquitectura Actor-Critic estándar.

**Estado**: TRL deprecó y después **eliminó** el trainer PPO clásico — `PPOTrainer` / `PPOConfig` / `AutoModelForCausalLMWithValueHead` no existen en el `trl>=0.28.0` que permite `requirements.txt`, así que `train_ppo.py` ni siquiera se puede importar. Ahora lo dice explícitamente al importarse, en lugar de lanzar un ImportError sin causa visible.

Hacer que PPO funcione implica reescribir el script contra la API `trl.experimental.ppo` (campos de configuración y formato de dataset distintos) o fijar trl por debajo de 0.9. En este proyecto PPO es una comparación didáctica: con recompensa verificable la vía es GRPO, y con un buen profesor, OPD — ambas consumen menos VRAM que PPO.

```bash
# sale de inmediato con la explicación anterior
python -m AgenticArxiv.rl.train_ppo --model outputs/grpo/final
```

**Salida** (tras reescribirlo): Modelo en `./outputs/ppo/final`

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
│  │  ├─ arxiv_tool.py             # Búsqueda por categoría/palabra clave en arXiv
│  │  ├─ pdf_download_tool.py      # Descarga de PDF
│  │  ├─ pdf_translate_tool.py     # Traducción de PDF
│  │  ├─ cache_status_tool.py      # Consulta de caché
│  │  ├─ paper_content_tool.py     # Lectura determinista del contenido (T2)
│  │  ├─ paper_summary_tool.py     # Resumen de papers del lado del entorno (T3)
│  │  └─ paper_figures_tool.py     # Extracción determinista de figuras (T4)
│  ├─ benchmark/                     # ⭐ Fuente de Verifiable Reward
│  │  ├─ metrics.py               # TaskMetrics, coincidencia estricta de herramientas y parámetros
│  │  ├─ tasks.py                 # BENCHMARK_TASKS (8 tareas de humo)
│  │  ├─ tasks_expanded.py        # Conjunto de tareas ampliado (77 tareas, 12 familias de plantillas)
│  │  ├─ task_spec.py             # TaskSpec: expected_tools / expected_tool_args derivados de steps
│  │  ├─ badcases.py              # Veredictos y replay de casos malos
│  │  ├─ splits.py                # Partición train/iid/ood a nivel de plantilla
│  │  ├─ baselines.py · run_baselines.py  # Baselines deterministas de política degradada (puertas por categoría)
│  │  ├─ runner.py · run_benchmark.py     # Ejecutor de benchmarks y entrada CLI
│  │  └─ report.py                # Informe de métricas
│  ├─ rl/                            # ⭐ Núcleo RL
│  │  ├─ train_sft.py              # ⭐ Entrenamiento SFT
│  │  ├─ train_dpo.py              # ⭐ Entrenamiento DPO
│  │  ├─ train_grpo.py             # ⭐ Entrenamiento GRPO (con guardias de entrenamiento)
│  │  ├─ train_ppo.py              # ⭐ Entrenamiento PPO (Actor-Critic)
│  │  ├─ train_opd.py              # ⭐ Entrenamiento OPD (destilación on-policy, intercambiable con GRPO)
│  │  ├─ opd_multiturn.py          # Rollout multi-turno, máscaras Observation y pérdida GKD propia
│  │  ├─ env.py                    # RLEnv + MockArxivEnv (entorno de snapshot offline)
│  │  ├─ multiturn_env.py          # Adaptador multiturno para TRL (environment_factory, una instancia por generación)
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
│  ├─ tests/                         # Tests unitarios (unittest / pytest)
│  └─ requirements.txt
├─ configs/accelerate/                # Configuración de arranque multi-GPU (ddp_2gpu / fsdp_2gpu)
├─ scripts/                          # Generación de datos
│  ├─ generate_sft_data.py          # Trayectorias expertas con LLM API
│  ├─ generate_parametric_sft_data.py  # Trayectorias expertas derivadas paramétricamente de las plantillas (sin API)
│  ├─ augment_sft_data.py           # Aumento de redacción que preserva la semántica
│  ├─ build_sft_train_mix.py        # Mezcla de entrenamiento auditable + manifest
│  └─ generate_dpo_data.py          # Pares de preferencia muestreando el modelo SFT local
├─ docs/
│  ├─ rl_building.md               # Plan de refactorización completo
│  ├─ multigranular_rl.md         # Diseño de recompensa multigranular (5 componentes + currículum)
│  └─ metric_stats.md            # Plan de estadísticas/métricas
├─ data/                             # Datasets (sft/ y dpo/ están en .gitignore — generarlos primero)
│  ├─ sft/                           # Dataset SFT (JSONL)
│  ├─ dpo/                           # Pares DPO (JSONL)
│  ├─ splits/                        # Particiones train/iid/ood por plantilla (v1 → v3_81)
│  └─ mock_arxiv_snapshot.json       # Snapshot del MockEnv
├─ eval/                             # Bucle de replay de casos malos
│  ├─ badcase_replay.py             # CLI de replay / captura (sin LLM)
│  ├─ eval_cases.jsonl              # Biblioteca de casos, doble uso como biblioteca de reward hacking
│  └─ readme.md
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
print(f'Reward: {reward:.2f}')  # Rango de recompensa: [-1, 1]; el valor depende de la trayectoria
```

---

## 🧪 Conjunto de Tareas y Evaluación

### Conjunto de humo (`benchmark/tasks.py`, 8 tareas)

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

### Conjunto ampliado (`benchmark/tasks_expanded.py`, 77 tareas)

Se activa con `run_benchmark.py --task-set expanded` y cubre doce familias de plantillas: search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction. Ambos conjuntos pasan por el `TaskSpec` de `benchmark/task_spec.py`: `expected_tools` y `expected_tool_args` se derivan de la misma fuente `steps`, así que dos listas mantenidas a mano nunca pueden divergir.

La familia de interpretación (`paper_reading` / `paper_summary`) fija en el oráculo de argumentos la exigencia de «esta clave debe omitirse» con un `{clave: None}` explícito: `argument_match_score` solo cuenta el acierto de las claves esperadas y no penaliza las extra, así que sin declarar la clave `section`, una tarea que pide «solo título + resumen» también aceptaría `section="method"` con la puntuación de argumentos completa.

Los `ref` de `figure_extraction` están **elegidos contra el snapshot**: en el snapshot offline esas posiciones sí llevan mapas de bits incrustados (las 12 primeras del pool de CV, todas). Que un paper tenga figuras es una propiedad del snapshot y no de la tarea, así que al cambiar de snapshot hay que repasarlo — `extract_paper_figures` devuelve `count: 0` en lugar de fallar cuando un paper no tiene mapas de bits, de modo que un desajuste se manifiesta como ruido en la recompensa y no como un fallo sonoro.

### Métricas de evaluación y particiones

Más allá de «tasa de éxito + tokens/iteraciones medias», el informe incluye:

- **Fiabilidad `pass^k`** (convención tau-bench, estimada por tarea y luego promediada; las tareas con pocas muestras se marcan como omitidas, no como 0)
- **`false_finish`**: termina con FINISH pero no se llamaron todas las herramientas esperadas — políticas degradadas miden `always_finish` 91.5% vs `reference` 0%
- **`ref_score`**: compara el `paper_id` resuelto en vez del literal `ref`, eliminando a la vez falsos positivos y falsos negativos
- **Costo normalizado por aciertos** (`skill_cli` corregido de 43% a 99% más caro) y desglose de modos de fallo

Las tareas se particionan por **plantilla** en train/iid_test/ood_test (`benchmark/splits.py`, fijado en `data/splits/v1.json`); `--split` está conectado tanto en `run_benchmark.py` como en `train_grpo.py`; `rl_train` toma solo la banda intermedia de tasa de éxito — las tareas de los extremos tienen varianza cero dentro del grupo y no producen gradiente.

### Puertas de discriminación y replay de casos malos

- **Puerta de discriminación por categoría** (`benchmark/run_baselines.py`): cuantifica la discriminación de la recompensa con políticas degradadas deterministas y fija umbrales por categoría (`tests/test_reward_discrimination.py`) — tras corregir las cuatro fugas de puntuación en la partida de argumentos, «buscar siempre cs.AI sin mirar la tarea» bajó de 0.833 a 0.446 en las tareas de búsqueda, y «llamar a una herramienta cuando lo correcto es no hacer nada» pasó de +0.165 a −0.235.
- **Replay de casos malos** (`eval/badcase_replay.py` + `eval/eval_cases.jsonl`): congela una trayectoria fallida junto con el veredicto original en un caso de regresión permanente; el replay solo re-ejecuta el evaluador, sin LLM / red / herramientas, así que `pytest` es en sí mismo la puerta (`tests/test_badcases.py::ShippedCasesTest`). Los casos son `open` (el fallo sigue ahí) o `fixed` (ya corregido — reproducirlo de nuevo es una regresión, código de salida 1); `hack/*` registra trayectorias tramposas de políticas degradadas con aserciones de umbral («este comportamiento no debe sacar X», custodiando los dos agujeros corregidos en #40 y #47), con doble uso como biblioteca de reward hacking — complementaria a las puertas de `run_baselines.py`: **las puertas miran medias, los casos clavan fallos individuales**. `--save-traces` + `capture` eligen casos malos de trayectorias reales por `false_finish` / `ref_score` / secuencia de herramientas, no solo los que cascan.

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

**Dependencias principales** (`requirements.txt`, cubre rollout / benchmark / las cinco fases de entrenamiento):
```txt
torch>=2.0.0
transformers>=4.45.0
trl>=0.28.0               # el mínimo lo fija el GRPO multiturno: solo desde 0.28.0 se llama
                          # a rollout_func en la ruta sin vLLM — versiones anteriores degradan
                          # en silencio los rollouts multiturno; verificado en 0.29.1 (OPD en 1.5.1)
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**Dependencias opcionales** (`requirements-extra.txt`, instalar según necesidad; el pipeline de entrenamiento principal no las requiere):
- `pdf2zh` — traducción real de PDF (el entrenamiento/benchmark usan el mock; solo para eval/demos de traducción)
- `fastapi` / `uvicorn` / `sqlalchemy` / `pymysql` — solo para ejecutar la versión Web archivada
- `tensorboard` (recomendado, cero configuración y offline) / `wandb` — backends de curvas de entrenamiento para `--report_to`

---

## 🔗 Recursos Relacionados

### Pesos del modelo
- [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) — checkpoint de la fase 1 (SFT), ajuste a parámetros completos sobre Qwen2.5-1.5B (ModelScope)
- [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) — checkpoint de la fase 3 (GRPO), entrenamiento online con verifiable reward sobre los pesos SFT (ModelScope)

### Documentación Oficial
- [Documentación de TRL](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### Papers
- **InstructGPT** (OpenAI, 2022): Tres fases de RLHF (SFT → RM → PPO)
- **DPO** (Stanford, 2023): Optimización directa de preferencias
- **RLVR**: Reinforcement Learning with Verifiable Reward
- **On-Policy Distillation** ([Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/)): origen de la receta OPD (muestreo on-policy del estudiante + KL reversa por token del profesor)
- **GKD / On-Policy Distillation of Language Models** (Agarwal et al., ICLR 2024, [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)): destilación JSD generalizada, base del GKDTrainer de TRL
- **Rethinking On-Policy Distillation of Large Language Models** ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016)): replicación y análisis independientes de la receta OPD

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
| **Dependencias** | Pesadas (14+ paquetes) | Livianas (11 paquetes principales + extras opcionales) |

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

### Q: ¿Cómo elegir entre OPD, SFT, DPO y GRPO?

| Escenario | Recomendación |
|------|------|
| Hay demostraciones de experto, aprender primero el formato de acción | SFT |
| Hay pares de preferencia (trayectorias buenas/malas), sin recompensa en línea | DPO |
| Hay recompensa verificable y se quiere superar la línea base en línea | GRPO |
| Hay un modelo profesor fuerte y se quiere ahorrar el costo de exploración de RL | OPD |

Se complementan entre sí: SFT es el punto de partida de todas las rutas; OPD y GRPO van encima de SFT — la primera destila desde un profesor (techo = profesor), el segundo optimiza contra recompensa verificable (puede explorar más allá), y la salida de cualquiera puede servir de inicialización o de línea base para el otro.

---

## 🧰 Diseño de Evolución del Conjunto de Herramientas (hoja de ruta incremental)

> El objetivo final de este proyecto es un **LLM ligero desplegado localmente que resuelva de forma autónoma la búsqueda, descarga e interpretación de papers de arXiv**. T1–T4 están implementados; T5 sigue siendo una propuesta de diseño.

### Estado actual y brechas

| Herramienta | Capacidad | Límite |
|------|------|------|
| `get_recently_submitted_cs_papers` | Búsqueda por categoría + ventana temporal | Solo entiende `cat:cs.*` + fecha de envío; sin paginación; resúmenes truncados a 200 caracteres |
| `search_arxiv_papers` | Búsqueda por palabra clave / título / autor | Sin paginación; las consultas ausentes del snapshot reciben un fallback determinista marcado explícitamente |
| `download_arxiv_pdf` | Descargar PDF | — |
| `translate_arxiv_pdf` | Traducción del paper completo con pdf2zh | Produce un archivo PDF traducido; el cuerpo traducido no entra en el contexto del modelo; depende del extra opcional |
| `get_paper_cache_status` | Consulta de caché | — |
| `get_paper_content` | Leer el resumen o una sección method/result/conclusion | Requiere un PDF descargado; extracción determinista sin llamar a un LLM |
| `summarize_paper` | Generar un resumen según style/presupuesto de palabras | Requiere un PDF descargado; por defecto usa un backend extractivo determinista, con un backend `local_model` opcional |
| `extract_paper_figures` | Extraer las imágenes de figuras incrustadas + captions y devolver las rutas de archivo | Requiere un PDF descargado; un paper solo vectorial (sin mapas de bits incrustados) devuelve `count: 0` |

Tres conclusiones:

1. **La mitad de recuperación del bucle está conectada**: ya funcionan la navegación por ventana temporal y la búsqueda por palabra clave/título/autor; la paginación sigue sin implementar.
2. **El bucle de interpretación ya está conectado**: leer contenido → resumir → extraer figuras son tres herramientas deterministas, así que el modelo puede completar por sí solo una cadena de interpretación de un paper. El **análisis semántico** de figuras (T5) sigue pendiente: requiere un VLM residente en el lado del entorno.
3. **Un espacio de acciones más grande no es automáticamente mejor**: la política es un modelo de ~1.5B, y cada herramienta nueva amplía la carga de aprendizaje de selección de herramientas y formato JSON. El criterio de admisión de una herramienta nueva es «habilita una nueva categoría de tareas», no «puede que sea útil» — de los 5 candidatos de la tabla, T1/T2 son el camino crítico, T3 es el incremento principal, T4 ya está implementado, T5 es opcional.

### Herramientas nuevas propuestas (en orden de dependencia)

| Prioridad | Herramienta | Diseño | Por qué sigue siendo amiga de RLVR |
|--------|------|----------|----------------------|
| **T1** ✅ | `search_arxiv_papers(query, max_results, days=None)` | Búsqueda por palabras clave mapeada a los campos `all:` / `ti:` / `au:` de la API de arXiv; coexiste con la herramienta actual (navegar por ventana temporal y búsqueda puntual son tipos de tarea distintos) | Las herramientas/parámetros esperados siguen derivándose de `task_spec.steps`; `MockArxivEnv` repite offline indexado por un hash del query, y los query no recogidos degradan de forma **determinista** (devuelve un subconjunto fijo, marcado explícitamente en la observation) — reproducible, y evita que el modelo confunda un resultado vacío con una búsqueda exitosa |
| **T2** ✅ | `get_paper_content(ref, section=None)` | PDF → texto plano (PyMuPDF); por defecto devuelve title/abstract, y por secciones (method / result / conclusion) a petición | Extracción de texto determinista, sin LLM; los resultados de extracción van pre-guardados en el snapshot. **Es el prerrequisito de todas las tareas de interpretación** |
| **T3** ✅ | `summarize_paper(ref, style, max_words)` | Resumir un paper: el resumen se genera **en el lado del entorno** (con el texto de T2 como entrada) y devuelve el texto | Lo entrenable es «cuándo llamarlo, sobre qué ref, si style/longitud son correctos» — todo verificable por reglas; la calidad del resumen en sí **no entra en la recompensa** (ver abajo) |
| **T4** ✅ | `extract_paper_figures(ref)` | Preparación de figuras/tablas: extrae imágenes de figuras + captions, devuelve rutas de archivos | Determinista; se verifica «ref correcto + cantidad ≥ 1» |
| **T5** (opcional, multimodal) | `analyze_figure(ref, figure_no, question=None)` | Análisis de figuras: un VLM local del lado del entorno (p. ej. Qwen2.5-VL) lee la figura y responde | Las reglas solo juzgan «si se llamó bien y si los parámetros son correctos»; la calidad de la respuesta del VLM no entra en la recompensa, manteniendo el ruido de un modelo tercero fuera del gradiente de política |

**El backend de resumen de T3**: este README decía originalmente «un modelo resumidor local del lado del entorno». Al implementarlo se adoptó por defecto un **backend extractivo determinista** (frases enteras por sección, recortadas al presupuesto de palabras, sin muestreo y sin modelo), por tres razones: hace que una misma trayectoria se repita byte a byte en cualquier momento; evita añadir, fuera de `build_snapshot`, otro prerrequisito que necesite pesos; y como la recompensa solo mira la decisión de llamada a herramientas y no el texto del resumen, el backend con modelo no aporta nada a la señal de entrenamiento. Para resúmenes de lenguaje más natural se puede cambiar al backend con modelo con `SUMMARY_BACKEND=local_model SUMMARY_MODEL_PATH=<directorio del modelo local>` (decodificación greedy, también determinista), a cambio de cargar pesos durante la construcción del snapshot.

Plantillas de tareas asociadas (siguiendo las familias de `tasks_expanded.py`, todas derivadas declarativamente de `task_spec.steps`):

- `search_kw_*`: tareas de búsqueda por palabras clave (T1)
- `paper_reading`: buscar → descargar → leer contenido (T2), 5 tareas
- `paper_summary`: buscar → descargar → resumir (T3), 5 tareas
- `figure_extraction`: buscar → descargar → extraer figuras (T4), 4 tareas
- `long_chain`, la tarea `chain_ai5_read_then_summary`: leer → resumir encadenado en 4 pasos (T2+T3)
- `analyze_figure(ref, figure_no)` (T5, opcional): buscar → descargar → extraer figuras → análisis de figuras, activo solo en entornos multimodales

### Diseño derivado para entrenamiento y evaluación

1. **Extensión del snapshot**: `build_snapshot.py` lo hace todo de una pasada — además de los resultados de búsqueda, **pre-extrae el texto completo y los archivos de figuras** de los papers del snapshot; todas las herramientas nuevas se repiten offline, conservando el contrato de «build_snapshot es el único paso con red».
2. **Cero cambios en la recompensa**: el esquema de cinco componentes se reutiliza tal cual; `expected_tools` / `expected_tool_args` se derivan de `steps`, así que `reward.py` y el currículo no se tocan.
3. **Prevención de reward hacking**: las herramientas de interpretación abren una superficie nueva para «llamar herramientas al azar para farmear puntos de process» — reutilizar las puertas por categoría de `run_baselines.py` + clavar casos individuales en `eval/eval_cases.jsonl` (p. ej. llamar a `summarize_paper` con un ref que apunta a un paper inexistente debe restar puntos).
4. **El problema de la recompensa por calidad del resumen (deliberadamente no hecho)**: convertir «si el resumen es bueno» en recompensa requiere LLM-as-judge o rúbricas, lo que introduce recompensas no deterministas y una nueva superficie de hacking. El diseño reduce primero el resumen a un **problema de decisión de llamada a herramientas** (cuándo llamar, a quién); evaluar calidad queda como un proyecto aparte a largo plazo.
5. **La frontera multimodal (aislada deliberadamente)**: el VLM de T5 vive solo en el lado del entorno; la política sigue siendo un modelo pequeño de solo texto — en el espacio de acciones solo está «llamar o no, cómo preguntar», y la comprensión de figuras se externaliza al entorno. Solo si la política en sí se vuelve multimodal se consideraría meter imágenes en la observation.
6. **Requisito de hardware**: T5 añade un modelo del lado del entorno (~6GB el VLM, menos cuantizado); no afecta a la VRAM de entrenamiento (sin gradientes). El backend por defecto de T3 es extractivo y no necesita pesos extra, así que el bucle de interpretación funciona sin más hardware que la máquina de entrenamiento.

### Orden de implantación

```
T1 búsqueda por palabras clave ──→ T2 leer contenido ──→ T3 resumir   (bucle de interpretación, completado)
                                        └────→ T4 extraer → T5 analizar figuras (T4 completado; T5 opcional, entorno multimodal)
```

Cada vez que aterriza una herramienta: ampliar las plantillas de tareas → re-ejecutar `run_baselines.py` para recalcular los umbrales de discriminación por categoría → regenerar los datos SFT/DPO → añadir los casos correspondientes a `eval/eval_cases.jsonl`. Al aterrizar T3/T4 se hizo además:

- Plantillas de tareas: nuevas familias `paper_reading` (5 tareas), `paper_summary` (5 tareas) y `figure_extraction` (4 tareas) más 1 cadena de interpretación de 4 pasos; el conjunto ampliado pasa de 62 a 77 tareas
- Particiones: nuevo `data/splits/v3_81.json` (v1/v2 quedan intactos, así que las tasas de éxito de los experimentos históricos siguen siendo comparables)
- Discriminación: las puertas por categoría de `run_baselines.py` cubren las categorías nuevas (`tests/test_reward_discrimination.py`)
- Casos malos: `eval/eval_cases.jsonl` suma 8 casos `hack/summary-*` / `hack/read-*` / `hack/figure-*`, que clavan formas de trampa como «presupuesto mal pasado», «resumir o extraer figuras sin descargar», «pasar una section de más» o «extraer figuras del paper equivocado» (la biblioteca tiene ya 14 casos)

---

## 📝 TODO (Hoja de Ruta de Desarrollo)

Ordenado por prioridad. ¡Las contribuciones son bienvenidas (ver 🤝 Contribuir)!

### P0 — Expansión del conjunto de herramientas (bucle de interpretación)

T1–T4 están implementados (ver «🧰 Diseño de Evolución del Conjunto de Herramientas»):

- [x] **T1 Búsqueda por palabras clave** `search_arxiv_papers`: añade la búsqueda puntual de «encontrar un paper concreto»
- [x] **T2 Lectura de papers** `get_paper_content`: PDF → texto determinista con replay offline del snapshot, el prerrequisito de todas las tareas de interpretación (camino crítico)
- [x] **T3 Resumen de papers** `summarize_paper`: resumen del lado del entorno, convirtiendo «interpretar» en una decisión de llamada a herramientas entrenable (backend extractivo determinista por defecto; `SUMMARY_BACKEND=local_model` cambia a un modelo local)
- [x] **T4 Extracción de figuras** `extract_paper_figures`: extracción determinista de figuras incrustadas y captions, con replay offline del snapshot
- [ ] **T5 Análisis de figuras** `analyze_figure` (opcional, entorno multimodal): ya están escritos la herramienta, la integración con el entorno, las plantillas, los tests y las reglas opcionales para derivar datos SFT parametrizados. Aún faltan un snapshot real y los datos generados. El VLM permanece en el entorno y la política sigue siendo de solo texto.
  - ⏳ **Falta el snapshot**: los snapshots locales no se incluyen en el repositorio y aquí no hay entradas T5 para verificar. Durante la construcción del 2026-09-22 arXiv limitó este host a ~5KB/s (un paper de 34MB entregó 492KB en 90s), por lo que no se completó la reconstrucción. Con acceso de red o PDFs en caché, ejecuta `python -m AgenticArxiv.rl.build_snapshot --skip-prefetch`; el replay falla si faltan entradas T5.
  - Si ya existe un snapshot con registros T4 de figuras, ejecuta `python -m AgenticArxiv.rl.backfill_figure_analysis --snapshot data/mock_arxiv_snapshot.json` para completar los resultados T5 del backend `extractive` a partir de captions, sin volver a descargar los PDFs. Los resultados VLM siguen requiriendo su propia grabación.
  - Backends: `extractive` por defecto (reutiliza el caption que T4 ya extrajo — determinista y sin pesos); `FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH=<dir del VLM local>` cambia a un VLM local (decodificación greedy, respuestas registradas al construir el snapshot).
  - La carga y el redimensionamiento de imágenes para VLM usan `Pillow`, declarado en `AgenticArxiv/requirements.txt`; la reproducción extractiva no requiere pesos del VLM.
  - **Generar datos expertos T5** (requiere un snapshot con entradas T5):
    ```bash
    python scripts/generate_parametric_sft_data.py \
      --split-file data/splits/v3_81.json \
      --snapshot data/mock_arxiv_snapshot.json \
      --include-t5
    ```
    El comando predeterminado conserva los datos históricos v2; la salida T5 va a `data/sft/sft_v3_t5_parametric_seed.jsonl`.
  - Límite conocido: en este repositorio no se han generado datos T5 ni se ha entrenado un modelo con ellos; el modelo publicado aún no aprendió esta herramienta.

### P1 — Ajuste del currículo de recompensa

- [x] **Calibración del currículo multigranular (medida; veredicto: la rampa de 30 pasos no se sostiene partiendo de SFT)**

  `scripts/analyze_curriculum.py` lee las curvas de entrenamiento reales. Hicimos una comparación de dos brazos: mismo punto de partida SFT (Qwen2.5-1.5B, 2 épocas, loss ≈ 0.08), las mismas 9 tareas que realmente producen gradiente intragrupo, 60 pasos cada uno, con la única diferencia de `--reward_curriculum_steps`.

  | Componente | currículo 30: suprimido → completo | currículo 0 |
  |---|---|---|
  | `format` | 0.983 → 1.000 | 0.983 → 1.000 |
  | `process` | 0.937 → 1.000 | 0.934 → 1.000 |
  | `tool` | −0.074 → +0.215 | −0.074 → +0.215 |
  | `argument` | −0.545 → −0.335 | −0.530 → −0.319 |
  | `outcome` | −0.127 → +0.028 | −0.127 → +0.030 |

  Dos conclusiones:

  1. **La premisa del currículo ya se cumple en el primer paso**: `format` arranca en 0.983 y `process` en 0.937. La rampa existe para «aprender primero el protocolo ReAct», y SFT ya lo ha enseñado — la puerta protege una etapa que el modelo ya ha superado.
  2. **Los dos brazos son indistinguibles**: las curvas de componentes se solapan casi punto por punto. Suprimir `tool`/`argument`/`outcome` durante treinta pasos no mejoró ni empeoró el aprendizaje semántico; solo gastó treinta pasos sobre una señal atenuada.

  Recomendación: **usa `--reward_curriculum_steps 0` para GRPO partiendo de SFT**; reserva la rampa para arranques en frío / entrenar directamente desde un modelo base. Advertencia honesta: 9 tareas × 60 pasos es una ejecución pequeña y la diferencia cae dentro del ruido — la conclusión no es «el currículo perjudica», sino «no se ha ganado esos 30 pasos».

- [x] **Defecto en la selección de tareas corregido de paso**: `rl_train` toma la banda media de tasa de éxito, pero las `rates` se midieron con el modelo **base**. Tras SFT la banda de dificultad se desplaza: de 48 tareas de train, solo **9 (19 %)** siguen dando gradiente intragrupo, y **24 de las 39 tareas de varianza cero están en el caso «banda media pero constante»** — «tasa media ⇒ varianza de grupo» es falso para una política saturada, y esas tareas queman pasos con `frac_reward_zero_std=1` hasta que `RewardVarianceGuard` aborta. Hay que volver a medir las rates cada vez que cambia el modelo en vez de arrastrarlas.

### P2 — Rendimiento y escala

- [x] **Soporte multi-GPU**: `configs/accelerate/ddp_2gpu.yaml` (DDP) y `fsdp_2gpu.yaml` (FSDP) ya están listos; los scripts de entrenamiento omiten automáticamente el anclaje a una sola GPU bajo `accelerate launch`, y arrancar QLoRA en multiproceso falla de forma explícita. DDP se verificó en una máquina de dos GPU (ambos ranks levantan la comunicación NCCL, sincronizan gradientes y guardan el modelo con normalidad).
  - La configuración FSDP necesita `torch>=2.6`: el `rl/trl_compat.py` de este repositorio solo ofrece un marcador de posición mono-GPU por debajo de esa versión, y esa ruta no se verificó en esta máquina.
  - Uso: `accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml -m AgenticArxiv.rl.train_sft --no-qlora ...` (el lanzador `accelerate` debe usar el mismo intérprete de Python que tiene las dependencias de entrenamiento)
- [ ] **Muestreo acelerado con vLLM**: sustituir HF generate para aumentar el throughput de muestreo del rollout multiturno.
  - **Estado**: TRL 0.29 exige vLLM 0.10.2–0.12.0, mientras que la versión disponible en esta máquina es una compilación de plataforma 0.6.2; instalar ambas hace que `trl.trainer.grpo_trainer` falle ya en la fase de import. Para avanzar haría falta primero una compilación de vLLM para la plataforma que case con el TRL actual: es una dependencia del entorno, no un cambio de código, así que queda en espera.

### P3 — Largo plazo (evolución algorítmica)

- [x] **Mejoras estilo DAPO (la parte que TRL ya soporta de forma nativa)**: los tres interruptores `--loss_type` (loss a nivel de token), `--epsilon_high` (clip-higher) y `--mask_truncated_completions` (filtro de overlong) ya están conectados en `train_grpo.py`, y hay un preset `--dapo` que los rellena de una vez (`loss_type=dapo`, `epsilon_high=0.28`, `mask_truncated_completions=True`, `beta=0`). Los valores por defecto no cambian nada, así que los experimentos históricos siguen siendo comparables.
  - **dynamic sampling (remuestreo de grupos con varianza cero) no está implementado**: TRL 0.29 no expone ningún gancho de generación interceptable, y sobrescribir `_generate_and_score_completions` se desviaría entre versiones. La red actual es `RewardVarianceGuard` (aborta el entrenamiento si la varianza es cero de forma consecutiva) junto con la curva `frac_reward_zero_std`, que convierte el «giro en vacío silencioso» en una señal visible.
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
> Ruta de adopción: el rollout multiturno ya está disponible → introducir la omisión de observaciones y el recorte bilateral DIS → migrar a verl `fully_async_policy` (`gen_batch_size=1` / `staleness_threshold` / TIS a nivel de token, alineado con SAO) o AReaL v1.0 para entrenamiento totalmente asíncrono + value model.
>
> 📄 **Paper**: [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (arXiv:2607.07508)](https://arxiv.org/abs/2607.07508) (Tsinghua KEG; código oficial aún no liberado)

---

**¡Comienza tu viaje de entrenamiento en RL Agentic!** 🚀
