<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a> | <a href="README.ja.md">🇯🇵 日本語</a>
</p>

# AgenticArXiv-RL — Entorno de Entrenamiento para RL Agentic

> **Entorno de entrenamiento de RL agentic basado en un agente ReAct + herramientas de arXiv**
> Soporta rutas de entrenamiento SFT/DPO/GRPO/OPD y recompensa verificable (RLVR), para investigar el aprendizaje por refuerzo en agentes LLM.
> Objetivo final: **un asistente de papers multimodal y ligero, desplegado en el dispositivo** — un modelo de lectura de papers de extremo a extremo, desde la búsqueda, la descarga y la traducción hasta el resumen y el análisis de figuras.

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="Introducción al proyecto AgenticArXiv-RL" width="800"/>
</p>

---

## 🎯 Posicionamiento del Proyecto

Convertir las tareas de búsqueda/descarga/traducción/interpretación de papers de arXiv en un **entorno de aprendizaje por refuerzo entrenable**, centrado en:

1. **Verifiable Reward**: recompensas basadas en reglas (precisión de las llamadas a herramientas, completitud de la tarea, errores de parseo, etc.), sin anotación humana
2. **Entrenamiento progresivo**: SFT → DPO → GRPO (se ofrece además una ruta de destilación on-policy con OPD; PPO no está disponible con las dependencias actuales)
3. **Ingeniería ligera**: Python puro + almacenamiento en JSONL, sin base de datos ni frontend, enfocado en el entrenamiento offline

**No es objetivo**: una aplicación de arXiv de nivel de producción o un servicio de traducción en tiempo real. `AgenticArxivWeb/` se conserva en este repositorio y se convertirá en el sandbox de la coevolución del entorno para RSI (véase el Roadmap P3); el original histórico está en [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv). El documento de diseño completo de la transformación del entorno está en [docs/rl_building.md](docs/rl_building.md).

### Experimento Opcional de Enrutamiento de Herramientas con Jev

Un enrutador discriminativo de la siguiente herramienta, desactivado por defecto (`TOOL_ROUTER=jev`): en una comprobación estructural con muestra pequeña, el éxito estricto subió del 40% al 60% y los tokens locales de Qwen bajaron un 74%; no se integra en el entrenamiento GRPO.

<details>
<summary>Desplegar: mecanismo de enrutamiento de Jev, política de repliegue y detalles del experimento</summary>

La inferencia ReAct estándar admite un enrutador Jev opcional que elige la siguiente herramienta.
En modo guided, un analizador determinista procesa los argumentos explícitos, los ambiguos quedan
a cargo del modelo de política y el entorno ejecuta las herramientas. `TOOL_ROUTER=policy` conserva
el comportamiento por defecto; al seleccionar `jev`, una confianza baja, un fallo de API o un error
de validación de argumentos hacen que se vuelva a la ruta original de la política. Esta función no
se integra en el entrenamiento GRPO. La configuración, las referencias, el piloto de 81 tareas que
solo evalúa el enrutamiento, las limitaciones y la reproducción están en [JEV_ROUTER.md](JEV_ROUTER.md);
los ajustes de activación y de API listos para copiar están en [jev_config.example.env](jev_config.example.env).
Una prueba de humo de las interfaces del proyecto, sin GPU ni red, comprueba que las rutas de la
política y de Jev son intercambiables mediante las mismas interfaces de ReAct, entorno y benchmark.
En una evaluación pequeña de 10 tareas mixtas con un Qwen entrenado con GRPO y 3 repeticiones,
guided Jev elevó el éxito estricto del 40% al 60%, redujo los tokens locales de Qwen un 74.01% y
aumentó la latencia total un 36.96%. Es una comprobación de integración con una muestra pequeña;
no establece la tasa de éxito global en las 81 tareas.

</details>

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar e instalar (Python 3.9+)
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
python3 -m venv .venv && source .venv/bin/activate
pip install -r AgenticArxiv/requirements.txt

# 2. Configurar la API del LLM (necesaria para los rollouts con API; el entrenamiento/evaluación con modelos locales no la necesita)
cat > AgenticArxiv/.env << 'EOF'
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo
EOF

# 3. Probar un rollout (recompensa en el rango [-1, 1], varía según la trayectoria)
python -m AgenticArxiv.rl.rollout search_01 traces/train/
# ✅ Rollout de search_01 completado  Recompensa: 1.00
```

Rollout por lotes: `python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/`. Los comandos siguientes se ejecutan por defecto desde la raíz del repositorio.

---

## 📚 Conceptos Clave

Las tareas sobre papers se convierten en un MDP: espacio de acciones con 10 herramientas + recompensa verificable de cinco componentes (RLVR) + aislamiento de rollouts — todo evaluable por reglas y reproducible offline.

<details>
<summary>Desplegar: diseño del MDP · espacio de acciones (10 herramientas) · recompensa verificable de cinco componentes · aislamiento de rollouts</summary>

### Diseño del MDP

| Dimensión | Definición |
|------|------|
| **State** | Descripción de la tarea + historial de diálogo + resultados de las herramientas |
| **Action** | 10 herramientas (véase abajo) + FINISH |
| **Reward** | Recompensa verificable multigranular de cinco componentes (format / tool / argument / process / outcome) |
| **Transition** | `execute_tool(action) → observation` (replay offline de snapshots con `MockArxivEnv`, determinista y reproducible) |

### Espacio de Acciones (10 herramientas)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Navegación por subárea + ventana temporal
2. `search_arxiv_papers(query, max_results, days=None)` — Búsqueda por palabra clave/título/autor
3. `download_arxiv_pdf(ref, session_id)` — Descarga del PDF
4. `translate_arxiv_pdf(ref, session_id)` — Traducción del PDF (pdf2zh)
5. `get_paper_cache_status(ref, session_id)` — Consulta del estado de la caché
6. `get_paper_content(ref, session_id, section=None)` — Lectura del abstract o de una sección concreta (extracción determinista)
7. `summarize_paper(ref, style, max_words)` — Resumen del lado del entorno (tldr / structured / bullet)
8. `extract_paper_figures(ref)` — Extracción de los archivos de figuras y sus pies de figura
9. `analyze_figure(ref, figure_no, question=None)` — Análisis de figuras: el entorno llama a un VLM local para leer la imagen (requiere el entorno multimodal)
10. `get_translated_content(ref, session_id, page=1)` — Lectura de la traducción página a página (el PDF traducido al chino, extracción determinista; la página 1 suele contener el título y el abstract)

> El bucle de interpretación «búsqueda → descarga → lectura → resumen → extracción de figuras → análisis de figuras» está completamente cerrado: el análisis de figuras lo realiza del lado del entorno el [VLM FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) post-entrenado en este proyecto, y los resultados se graban en snapshots. Del lado de la política, los pesos [SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) han aprendido la cadena de cuatro pasos para el análisis de figuras.
> **Criterio de admisión**: un espacio de acciones más grande no es automáticamente mejor — el estándar para añadir una herramienta es «habilita una nueva categoría de tareas», no «puede que sea útil». La cadena de decisiones de diseño está en [Diseño de Evolución del Conjunto de Herramientas](docs/toolset_evolution.md).

### Verifiable Reward (cinco componentes)

**Recompensa verificable multigranular** (`rl/reward.py`). Los cinco componentes se normalizan a `[-1, 1]`; además hay componentes de diagnóstico y un techo de fallo no compensable:

| Componente | Peso por defecto | Señal |
|------|:---:|------|
| `format` | 1 | Que la acción de cada paso sea una llamada JSON válida a una herramienta o un terminador |
| `tool` | 3 | **LCS-F1 sensible al orden** entre la secuencia de herramientas predicha y la esperada |
| `argument` | 2 | Recall de las claves de argumentos × precisión de los valores |
| `process` | 1 | Puntos por pasos válidos − penalizaciones por fallos de parseo/ejecución y llamadas de más |
| `outcome` | 3 | Completado correcto +1, completado con ruta errónea +0.25, parada forzada −0.5, error −1 |
| `result_quality` / `efficiency` | gate de diagnóstico | Si la observation representa un éxito real; la redundancia grave activa un techo de reward |

- **Currículo**: durante los primeros 30 pasos los pesos de `tool`/`argument`/`outcome` se multiplican por 1/3 (`RewardCalculator.schedule`). Conclusión medida: un punto de partida SFT ya domina la estructura ReAct (format arranca en 0.983) y los dos brazos del currículo son indistinguibles — **partiendo de SFT usa directamente `--reward_curriculum_steps 0`**; esa rampa queda reservada para escenarios de arranque en frío.
- **Fallos no compensables**: un fallo de parseo, un fallo de ejecución o un FINISH falso reciben como mucho recompensa negativa y no pueden compensarse con puntos de formato; si `analyze_figure` no aporta una respuesta válida, la recompensa total queda limitada a −0.75.
- Toda recompensa es **verificable por reglas** (RLVR) y cada trayectoria registra el desglose de `reward_components` para facilitar la auditoría. Los criterios detallados están en la [documentación de la recompensa multigranular](docs/multigranular_rl.md).

### Aislamiento de Rollout

El `RolloutSandbox` de `rl/sandbox.py` registra antes de cada trayectoria la línea base del entorno, del Store y de los directorios de artefactos, y la restaura y limpia al terminar — el GRPO multiturno, el rollout y el benchmark comparten este contrato de reset, eliminando cualquier encadenamiento de estado entre trayectorias.

</details>

---

## 🛠️ Ruta de Entrenamiento (SFT → DPO → GRPO / OPD)

Entrenamiento progresivo SFT → DPO → GRPO; la destilación on-policy con OPD es intercambiable con GRPO; PPO no está disponible con las dependencias actuales. Las variantes de objetivo GSPO / Dr.GRPO / DAPO ya están conectadas y completaron el entrenamiento comparativo con pesos publicados.

<details>
<summary>Desplegar: comandos y notas de cada fase (SFT · DPO · GRPO y variantes · OPD · VLM multimodal)</summary>

### Fase 1: SFT

```bash
python scripts/generate_parametric_sft_data.py   # deriva trayectorias expertas de forma paramétrica (sin API)
python -m AgenticArxiv.rl.train_sft              # produce outputs/sft/final
```

> Publicado: [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) (Qwen2.5-1.5B a parámetros completos, 2 epochs / 2628 trayectorias expertas, loss 0.079; cubre solo las primeras 8 herramientas).

### Fase 2: DPO

```bash
python scripts/generate_dpo_data.py --model outputs/sft/final --num_rollouts_per_task 8
python -m AgenticArxiv.rl.train_dpo              # produce outputs/dpo/final
```

Los pares de preferencia proceden del muestreo local del modelo SFT (sin necesidad de `LLM_API_KEY`); cuando hay snapshot, el replay offline es automático, y solo forman par las trayectorias cuya diferencia de recompensa supera `--min_reward_gap` y cuya primera herramienta es distinta.

### Fase 3: GRPO

```bash
python -m AgenticArxiv.rl.build_snapshot         # genera el snapshot offline (único paso con red)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --max_turns 4

# Preset DAPO (loss_type=dapo + clip-higher + overlong filtering + dynamic sampling)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --dapo

# Importancia a nivel de secuencia (GSPO) y variante sin sesgo Dr.GRPO
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --importance_sampling_level sequence
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --loss_type dr_grpo

# Multi-GPU (DDP verificado en dos tarjetas; FSDP necesita torch>=2.6)
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --no-qlora

# Curvas de entrenamiento
python -m AgenticArxiv.rl.train_grpo --report_to tensorboard
tensorboard --logdir outputs/grpo/logs
```

> Publicado: [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) (33 tareas con señal, replay offline de snapshots de principio a fin). Evaluación offline (seed 45 / repeat 3, tasa de éxito estricta pass³) SFT → GRPO: **rl_train 0.081→0.636, dev 0.000→0.375, iid_test 0.056→0.444, ood_test 0.000→0.500**.

> **Comparación de variantes de objetivo** (seed 42 / 60 pasos, pass³, pesos publicados — véase la tabla de abajo): GSPO empató con la referencia en todas las particiones (trayectorias divergentes, mismo comportamiento convergido); Dr.GRPO quedó ligeramente por delante en rl_train (0.657) con la mejor precisión de herramientas en ood; **DAPO fue el mejor en tres particiones (0.667 / 0.500 / 0.481 / 0.500)**. Notas de implementación en [docs/rl_paradigm_comparison.md](docs/rl_paradigm_comparison.md). **Marco de ablación de pesos de recompensa**: `--reward_config` admite nueve diseños de pesos de una sola variable ([docs/ablation_reward_weights.md](docs/ablation_reward_weights.md), 🟡 diseñado, ejecución pendiente).

**Rollout y puntuación multiturno** (`rl/grpo_reward.py`): en cada turno la política actual genera una acción ReAct y un `MockArxivEnv` independiente la ejecuta, reinsertando la observation en el contexto; los tokens del assistant entran en la loss y los del entorno actúan solo como contexto vía `env_mask=0`; la trayectoria completa la puntúa el `RewardCalculator` de cinco componentes, con el mismo estándar que usan el rollout y el benchmark.

**Garantías de calidad del entrenamiento** (de fallos silenciosos a errores explícitos): comprobación de la longitud de generación, protección frente a varianza cero (`RewardVarianceGuard`), evaluación canary intermedia con parada temprana, umbrales de verificación por fase (tasa de parseo SFT ≥ 0.3, reward DPO ≥ −0.3, reward GRPO ≥ −0.2), precisión mixta adaptativa y validación previa del backend de logging. Además de las métricas que trae TRL, las curvas registran `reward_components/*` (desglose por componente), `reward_weights/*` (pesos del currículo) y `rollout/*` (turns / finished / parse_error_rate).

### Fase 3': OPD (destilación on-policy, opcional, intercambiable con GRPO)

```bash
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final \
  --teacher Qwen/Qwen2.5-7B-Instruct --max_turns 4 --snapshot data/mock_arxiv_snapshot.json
```

El estudiante muestrea on-policy sobre los prompts de las tareas, el profesor aporta los logprobs por token y la pérdida es una reverse-KL (mode-seeking). Frente a GRPO:

| Dimensión | GRPO | OPD |
|---|---|---|
| Señal de aprendizaje | recompensa verificable (dispersa, a nivel de trayectoria) | logprobs por token del profesor (densa) |
| Modelo adicional | ninguno | modelo profesor (necesita pesos locales) |
| Techo | puede explorar y superar al profesor | converge al comportamiento del profesor |
| Cuándo usarlo | hay recompensa verificable | hay un profesor fuerte y se quiere ahorrar el coste de exploración |

### Fase 4: PPO — ⚠️ No disponible con las dependencias actuales

TRL eliminó el PPOTrainer clásico (no existe en `trl>=0.28.0`); `train_ppo.py` muestra una explicación clara en la fase de import y después termina. Con recompensa verificable la vía es GRPO y con un profesor fuerte, OPD — ambas consumen menos VRAM que PPO.

### Post-entrenamiento multimodal (VLM del lado del entorno)

```bash
python scripts/build_figure_qa_dataset.py                          # datos semilla de QA de figuras
python -m AgenticArxiv.rl.train_vlm_figure_qa                      # post-entrenamiento LoRA de Qwen3-VL-4B
# Sustituye /path/to/figureqa por el directorio local del modelo FigureQA
FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH="/path/to/figureqa" \
  python -m AgenticArxiv.rl.backfill_figure_analysis --snapshot data/mock_arxiv_snapshot.json
```

> Publicado: [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) (451 figuras / 73 papers, partición por paper para evitar fugas; holdout de 94 figuras, ROUGE-1 0.105→0.126 y ROUGE-L 0.081→0.102). Los resultados de la cadena completa de los VLM de política de 8B / 27B están en «🏆 Resultados del Post-entrenamiento de VLM para el Dispositivo», más abajo.

</details>

---

## 📦 Modelos Publicados

| Modelo | Base | Descripción |
|------|------|------|
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) | Qwen2.5-1.5B | Fase 1: SFT a parámetros completos (primeras 8 herramientas) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) | Qwen2.5-1.5B | Fase 3: GRPO (evaluación en las cuatro particiones, más arriba) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO) | Qwen2.5-1.5B | GRPO + muestreo de importancia a nivel de secuencia (empatado con la referencia en las cuatro particiones; ver el doc de comparación) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO) | Qwen2.5-1.5B | Variante sin sesgo de GRPO (rl_train 0.657 / dev 0.375 / iid 0.444 / ood 0.500, pass³) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO) | Qwen2.5-1.5B | Objetivo DAPO sin muestreo dinámico (rl_train 0.667 / dev 0.500 / iid 0.481 / ood 0.500, pass³; la mejor de las tres variantes) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) | Qwen2.5-1.5B | SFT + cadena de cuatro pasos con `analyze_figure` (rl_train 0.485 / dev 0.250 / iid 0.278 / ood 0.250, pass³) |
| [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) | Qwen3-VL-4B | VLM de análisis de figuras del lado del entorno |
| [AgenticArXiv-RL-Qwen3-VL-8B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-8B-GRPO) | Qwen3-VL-8B | **Modelo principal del asistente de artículos en el dispositivo**: política multimodal y cadena completa SFT→DPO→GRPO (éxito estricto del 72.8% en 81 tareas, 2.9× la base; ROUGE-1 de figuras +32%) |
| [AgenticArXiv-RL-Qwen3.8-27B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3.8-27B-SFT) | Qwen3.8-27B | Edición de investigación especializada en comprensión de figuras (ROUGE en clean-val +60%/+87%; las decisiones de herramientas no alcanzaron los criterios de aceptación, véase el documento de experimentos) |

Todos se alojan en ModelScope (`Algorineko/AgenticArXiv-RL-*`); las model cards incluyen la tabla de evaluación offline y una descripción honesta de sus limitaciones.

---

## 🏆 Resultados del Post-entrenamiento de VLM para el Dispositivo (Qwen3-VL-8B · Qwen3.8-27B)

En 4×64GB de DCU Hygon se completó la **cadena SFT → DPO → GRPO** de Qwen3-VL-8B y el SFT
especializado en comprensión de figuras de Qwen3.8-27B, un modelo multimodal nativo. Los datos
pasaron por dos rondas de depuración: filtrado de clipart y normalización del manual de herramientas.
La evaluación comparte un **protocolo de decodificación greedy**, un benchmark offline de 81 tareas
y ROUGE de figuras en clean-val.

| Fase | Éxito estricto en 81 tareas | Tasa de finalización | Precisión de herramientas | Finalización falsa | ROUGE-1 limpio |
|---|---|---|---|---|---|
| 8B base | 24.7% | 80.2% | 32.1% | 22.2% | 0.1139 |
| 8B SFT v3 | 66.7% | 95.1% | 72.8% | 8.6% | 0.1446 (+27%) |
| 8B DPO v3 | 65.4% | 97.5% | 74.1% | 11.1% | 0.1435 |
| **8B GRPO v3 (modelo principal)** | **72.8%** | **97.5%** | **82.7%** | **6.2%** | **0.1509 (+32%)** |
| 27B base (control) | 24.7% | 80.2% | 32.1% | 22.2% | 0.1349 |
| 27B cap SFT (edición de investigación) | 21.0% | 66.7% | 25.9% | **4.9%** | **0.2164 (+60%)** |

**Modelo principal: 8B GRPO (éxito estricto del 72.8%) = 2.9× 27B base**, con el mayor éxito estricto
y la mayor precisión de herramientas de la tabla. Dos ejecuciones independientes alcanzaron un
71.6% y un 72.8% de éxito estricto (4.9% de tareas con cambio de resultado); el ROUGE-1 de figuras
en clean-val mejoró un 32% respecto a 8B base. El adaptador 27B de mayor capacidad (r128) también
se publica como edición de investigación, con mejoras de ROUGE de +60%/+87%, las mayores del
proyecto. Los límites de la adaptación LoRA con pocos datos al fuerte conocimiento previo de 27B y
las tres etapas de análisis de causas (contaminación por clipart / deriva del registro / protocolo
de muestreo) se documentan en **[docs/edge-vlm-posttrain.md](docs/edge-vlm-posttrain.md)**
(selección del modelo / evolución de datos / cuatro rondas de experimentos / análisis de causas).

---

## 🧪 Conjunto de Tareas y Evaluación

- **Conjunto de pruebas de humo** (`benchmark/tasks.py`): 8 tareas (search / download / translate / cache / composite)
- **Conjunto ampliado** (`benchmark/tasks_expanded.py`): 86 tareas y catorce familias de plantillas (search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction / figure_analysis / **translation_reading**); se activa con `run_benchmark.py --task-set expanded`. Ambos conjuntos pasan por el mismo `TaskSpec`: `expected_tools` / `expected_tool_args` se derivan de una única fuente `steps`, de modo que nunca hay dos listas escritas a mano que diverjan.
- **Particiones**: train / iid_test / ood_test por plantilla (`benchmark/splits.py`, fijadas en `data/splits/`); `rl_train` toma solo la banda intermedia de la tasa de éxito (en los extremos la varianza intragrupo es cero y no se produce gradiente). Tras cambiar de modelo hay que volver a medir las tasas de éxito: los umbrales antiguos no se pueden reutilizar.
- **Criterios de evaluación**: fiabilidad `pass^k` (convención tau-bench), `false_finish` (la política degradada `always_finish` alcanza un 91.5% frente al 0% de `reference`), `ref_score` (compara el `paper_id` y no la forma literal del `ref`), coste normalizado por número de aciertos.
- **Puertas de discriminación** (`run_baselines.py`): políticas degradadas deterministas con umbral por categoría — «buscar siempre cs.AI» baja de 0.833 a 0.446 en la categoría de búsqueda, y «llamar a herramientas cuando no tocaba» pasa de +0.165 a −0.235.
- **Replay de casos malos** (`eval/badcase_replay.py` + `eval/eval_cases.jsonl`, 17 casos): cada trayectoria fallida se congela como caso de regresión permanente; el replay ejecuta solo el evaluador y el propio `pytest` hace de puerta. `hack/*` documenta formas de engañar la puntuación con aserciones de umbral y sirve además de biblioteca de casos de reward hacking.

```bash
python -m AgenticArxiv.benchmark.run_benchmark --task-set expanded --split iid_test --offline
```

---

## 🛡️ Notas sobre Dependencias

**Dependencias principales** (`AgenticArxiv/requirements.txt`): `torch>=2.0`, `transformers>=4.45`, `trl>=0.28.0` (verificado en 0.29.1; el mínimo lo fija la ruta `rollout_func` del GRPO multiturno), `peft`, `bitsandbytes` (QLoRA), `datasets`, `accelerate`, `arxiv`, `requests`, `python-dotenv`, `loguru`, `pydantic>=2`, `fire`, `PyMuPDF>=1.24.0`, `Pillow`.

**Dependencias opcionales** (`AgenticArxiv/requirements-extra.txt`): `pdf2zh` (traducción real de PDF), `fastapi`/`uvicorn`/`sqlalchemy`/`pymysql` (capa de compatibilidad web archivada), `mcp` (capa de compatibilidad MCP archivada), `tensorboard`/`wandb` (backends de curvas de entrenamiento), `matplotlib`/`numpy`/`pandas` (scripts de gráficas).

---

## 🔗 Recursos Relacionados

- **Documentación**: [Diseño de Evolución del Conjunto de Herramientas](docs/toolset_evolution.md) · [Recompensa Multigranular](docs/multigranular_rl.md) · [Estadísticas de Métricas](docs/metric_stats.md) · [Notas de Investigación del Roadmap](docs/roadmap_notes.md) · [Documentación de TRL](https://huggingface.co/docs/trl/)
- **Papers de métodos**: RLVR; DPO (Stanford, 2023); [On-Policy Distillation (Thinking Machines, 2025)](https://thinkingmachines.ai/blog/on-policy-distillation/); [GKD (arXiv:2306.13649)](https://arxiv.org/abs/2306.13649)
- **Proyecto original**: [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) (versión app web, FastAPI + Vue3 + MySQL)

---

## 🤝 Contribuir

¡Issues y PRs son bienvenidos! El flujo es: Fork → rama feature → `pytest AgenticArxiv/tests/` → commits con prefijo `feat:` → PR. Más detalles en [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 Licencia

Licencia MIT

---

## 🙋 FAQ

<details>
<summary>Desplegar: preguntas frecuentes (diferencias con el original · por qué GRPO · por qué la calidad no entra en la recompensa)</summary>

**P: ¿En qué se diferencia del AgenticArXiv original?** El original es una aplicación web de nivel de producción (FastAPI+Vue3+MySQL, tres arquitecturas de agente); este proyecto es un entorno de entrenamiento RL en Python puro + JSONL que conserva únicamente la política única ReAct: su núcleo son el entrenamiento SFT/DPO/GRPO y la recompensa verificable.

**P: ¿Por qué GRPO y no PPO?** GRPO no necesita value model (menor consumo de VRAM), encaja con modelos pequeños de la escala de 1.5B y su implementación es simple y fácil de depurar; PPO es más adecuado para entrenar modelos grandes de nivel de producción.

**P: ¿Por qué la calidad del resumen/análisis de figuras no entra en la recompensa?** Puntuar la calidad exigiría LLM-as-judge, lo que introduce no determinismo y una superficie nueva de hacking. El diseño reduce la interpretación a un **problema de decisión de llamada a herramientas** (cuándo llamar, a quién, con qué argumentos) — todo evaluable por reglas y reproducible; es la premisa de RLVR.

</details>

---

## 📝 Roadmap (Hoja de Ruta)

> Los argumentos completos y la ruta de adopción están en [docs/roadmap_notes.md](docs/roadmap_notes.md). Las tareas completadas se han integrado en las secciones correspondientes del cuerpo del documento. Anímate a reclamar una tarea (véase 🤝 Contribuir).

<details>
<summary>Desplegar: hoja de ruta (P0 bucle en el dispositivo · P1 nuevos algoritmos · P2 Jev · P3 coevolución · ⛔ bloqueos)</summary>

### P0 — Asistente de Papers Multimodal en el Dispositivo (objetivo final)

- [ ] **Cadena de lectura de extremo a extremo**: búsqueda → descarga → traducción → resumen → análisis de figuras, cerrada dentro de un único modelo y optimizada para la inferencia en el dispositivo (cuantización + escala 2–4B). La traducción ya está lista (`get_translated_content` lee la traducción de pdf2zh página a página de forma determinista, con replay desde snapshots grabados con `build_snapshot --translate-max-ref N`); la multimodalidad del lado de la política completó la cadena de 8B (véase «🏆 Resultados del Post-entrenamiento de VLM para el Dispositivo» y el [registro de post-entrenamiento de VLM](docs/edge-vlm-posttrain.md)) — el núcleo pendiente es meter toda la cadena en el presupuesto del dispositivo

### P1 — Algoritmos de RL Agentic de Nueva Generación

- [ ] **Entrenamiento asíncrono estilo SAO**: migrar a verl `fully_async_policy` / AReaL, introduciendo primero la máscara skip-observation y el recorte bilateral de DIS ([arXiv:2607.07508](https://arxiv.org/abs/2607.07508), código oficial sin publicar)
- [ ] **Asignación de crédito entre pasos**: ventajas intragrupo que crucen pasos, al estilo GiGPO / ARPO, para aliviar la señal dispersa a nivel de trayectoria de las cadenas largas (línea de investigación propia)
- [ ] **TTT agéntico (entrenamiento/adaptación en tiempo de prueba)**: el TTT está muy extendido en la IA encarnada; que funcione a nivel de agente LLM es una pregunta abierta. Forma en este repositorio: dentro del episodio de un solo paper, autogenerar intentos de QA/resumen → pseudorrecompensas por voto mayoritario/verificador (TTRL, [arXiv:2504.16084](https://arxiv.org/abs/2504.16084)) → unos pocos pasos de LoRA → responder de nuevo, comparado con una referencia de ICL de contexto largo para comprobar si la «especialización instantánea» compensa. Restricciones: presupuesto de latencia en el dispositivo, olvido catastrófico, el conflicto entre el no determinismo del TTT y la evaluación reproducible de RLVR, y el criterio de fuga de «entrenar sobre la distribución de prueba» con snapshots offline

### P2 — Componente de Decisión Discriminativo estilo Jev

- [ ] **Cabeza discriminativa estilo Jev**: decisión estructurada no autorregresiva (choice/score, ~1/400 del coste de un LLM), posibles puntos de integración: ① una cabeza discriminativa para `result_quality` en lugar de reglas ② un enrutador de componentes de recompensa (en lugar del currículo fijo) ③ decodificación estructurada de la selección de herramientas (los enums cerrados encajan de forma natural en lo discriminativo). Restricción: inferencia determinista o uso solo como señal de diagnóstico, para mantener la reproducibilidad de RLVR
- [ ] **Jev entrenado con RLCD**: señal contrastiva de trayectorias de decisión exitosas vs fallidas (recompensa verificable, sin etiquetas humanas) entrena la cabeza discriminativa, sustituyendo la API externa de TypeSafe. **Comparación de dos brazos validada** ([docs/jev_rlcd.md](docs/jev_rlcd.md)): el término contrastivo supera a la base de solo CE en las cuatro suites (992 puntos congelados, McNemar p<0.0001, Δ +33.0pp); pasos intermedios de cadena 91.0% con alta precisión dentro de la puerta; primeros pasos de cadenas compuestas 53.1% como debilidad conocida (repliegue seguro bajo la puerta de 0.80). Pendiente: aumento dirigido de primeros pasos, conexión de `TOOL_ROUTER=local`

### P3 — Autoevolución RSI Acotada (bounded RSI)

- [ ] **Bucle de datos autoevolutivo**: la evaluación holdout expone debilidades → la biblioteca de casos malos se amplía automáticamente → derivación paramétrica de datos dirigidos → reentrenamiento → congelación de un nuevo holdout (reutiliza las líneas existentes de evaluación/datos/entrenamiento; los prerrequisitos ya están listos). Criterio de aceptación: el pass³ en las cuatro particiones no retrocede en ninguna ronda y la biblioteca de casos malos solo crece (ciclo de cuatro pasos de [arXiv:2609.11873](https://arxiv.org/abs/2609.11873)). **Orquestador implementado** (`AgenticArxiv/rl/self_evolve.py`: diagnose → mine → select → synthesize → freeze → gate, solo JSON, sin modelo; `generate_parametric_sft_data.py --only-parents` para datos dirigidos); queda pendiente una ronda real de entrenamiento con la comparación en cuatro particiones (ver [docs/roadmap_notes.md](docs/roadmap_notes.md))
- [ ] **Coevolución de harness y entorno**: la autoevolución va más allá de los pesos (véase el bucle de datos anterior) — lado harness: automodificación y archivo al estilo DGM de prompts / enrutamiento de herramientas / flujos de trabajo ([Darwin Gödel Machine, arXiv:2505.22954](https://arxiv.org/abs/2505.22954)); lado entorno: distribuciones de tareas y funciones de recompensa autogeneradas (self-play al estilo Absolute-Zero, [arXiv:2505.03335](https://arxiv.org/abs/2505.03335)), con `AgenticArxivWeb/` como portador que se convertirá en un sandbox evolvable de tareas/recompensas. La disciplina acotada se mantiene: auditorías manifest SHA256, puertas holdout de cuatro particiones, promoción humana

### ⛔ Bloqueos del Entorno

- [ ] **Muestreo acelerado con vLLM**: TRL 0.29 exige vLLM 0.10.2–0.12.0, y la compilación personalizada 0.6.2 de la plataforma local, instalada en conjunto, hace que la importación de GRPOTrainer falle. Es una dependencia del entorno, no un cambio de código: queda a la espera de que la plataforma ofrezca una compilación compatible.

</details>

---

**¡Comienza tu viaje de entrenamiento en RL Agentic!** 🚀

---

## 🌟 Colaboradores

Gracias a todas las personas que han hecho mejor este proyecto:

<a href="https://github.com/Algorineko/AgenticArXiv-RL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Algorineko/AgenticArXiv-RL" alt="Colaboradores"/>
</a>
