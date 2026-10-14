<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Entorno de Entrenamiento para RL Agentic

> **Entorno de entrenamiento de RL agentic basado en un agente ReAct + herramientas de arXiv**
> La línea de entrenamiento admitida es SFT → GRPO; la inferencia y la evaluación usan ReAct textual con recompensas verificables (RLVR).
> Objetivo final: **un asistente de papers multimodal y ligero, desplegado en el dispositivo** — un modelo de lectura de papers de extremo a extremo, desde la búsqueda, la descarga y la traducción hasta el resumen y el análisis de figuras.

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="Introducción al proyecto AgenticArXiv-RL" width="800"/>
</p>

---

## 🎯 Posicionamiento del Proyecto

Convertir las tareas de búsqueda/descarga/traducción/interpretación de papers de arXiv en un **entorno de aprendizaje por refuerzo entrenable**, centrado en:

1. **Verifiable Reward**: recompensas basadas en reglas (precisión de las llamadas a herramientas, completitud de la tarea, errores de parseo, etc.), sin anotación humana
2. **Línea directa de entrenamiento**: SFT → GRPO, con QLoRA por defecto y entrenamiento de todos los parámetros explícito
3. **Ingeniería ligera**: Python puro + almacenamiento en JSONL, sin base de datos ni frontend, enfocado en el entrenamiento offline

**No es objetivo**: una aplicación de arXiv de nivel de producción, una UI web o un servicio de traducción en tiempo real (la app web original se retiró de este repositorio; véase el [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) original).

---

## Inicio Rápido

Ejecuta los comandos desde la raíz del repositorio y usa `.venv`. La instalación no requiere frontend ni base de datos.

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
python3 -m venv .venv
source .venv/bin/activate
pip install -r AgenticArxiv/requirements.txt

python -m AgenticArxiv.rl.rollout --help
python -m AgenticArxiv.benchmark.run_benchmark --help
python -m AgenticArxiv.main --help
```

Para la inferencia mediante API, configura explícitamente `LLM_BASE_URL`, `LLM_API_KEY` y `MODEL`; no publiques credenciales ni un `.env` real. El entrenamiento y la evaluación locales no necesitan API key. Instala `AgenticArxiv/requirements-extra.txt` solo para traducción PDF real o registro de curvas; no hay servicio asíncrono de la aplicación.

```bash
# Ejecución futura con modelo local y snapshot existentes; no se ejecutó aquí
python -m AgenticArxiv.rl.rollout \
  --task_id=search_01 --output_dir=traces/react_messages \
  --model=outputs/sft_qlora/final --offline=True \
  --snapshot=data/mock_arxiv_snapshot.json --max_iterations=6
```

main usa API/live por defecto. Rollout usa offline=True; False explícito activa herramientas live. El modo de herramientas es independiente del backend del modelo; véase el [contrato de ejecución](docs/react_runtime.md).

Ejecutar tareas requiere un modelo existente o una API disponible. Las herramientas offline necesitan un snapshot existente y no descargan entradas ausentes. Verificar la instalación o la ayuda no equivale a ejecutar tareas; este cambio no llamó a APIs externas, entrenó modelos ni evaluó su rendimiento.

---

## 📚 Conceptos Clave

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
3. `download_arxiv_pdf(ref)` — Descarga del PDF
4. `translate_arxiv_pdf(ref)` — Traducción del PDF (pdf2zh)
5. `get_paper_cache_status(ref)` — Consulta del estado de la caché
6. `get_paper_content(ref, section=None)` — Lectura del abstract o de una sección concreta (extracción determinista)
7. `summarize_paper(ref, style, max_words)` — Resumen del lado del entorno (tldr / structured / bullet)
8. `extract_paper_figures(ref)` — Extracción de los archivos de figuras y sus captions
9. `analyze_figure(ref, figure_no, question=None)` — Análisis de figuras: el entorno llama a un VLM local para leer la imagen (requiere el entorno multimodal)
10. `get_translated_content(ref, page=1)` — Lectura de la traducción página a página (el PDF traducido al chino, extracción determinista; la página 1 suele contener el título y el abstract)

> Se muestran parámetros de negocio habituales, no el schema completo. `session_id` se inyecta al ejecutar y nunca se incluye en Action ni en las etiquetas SFT.

> **Capacidad histórica del protocolo original, sin repetir la evaluación con el protocolo nuevo.** El bucle de interpretación «búsqueda → descarga → lectura → resumen → extracción de figuras → análisis de figuras» está completamente cerrado: el análisis de figuras lo realiza del lado del entorno el [VLM FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) post-entrenado en este proyecto, que graba las respuestas en el snapshot, y del lado de la política existen los pesos [SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5), que aprendieron esa cadena de cuatro pasos.
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

- **Currículo**: el código conserva el escalado por 1/3 de `tool`/`argument`/`outcome` durante los primeros 30 pasos (`RewardCalculator.schedule`); la limpieza no lo cambió a 0. El registro histórico indica format 0.983 al inicio SFT y brazos indistinguibles. Una futura opción explícita `--reward_curriculum_steps 0` debe registrarse en el manifest y no describirse como valor predeterminado conservado.
- **Fallos no compensables**: un fallo de parseo, un fallo de ejecución o un FINISH falso reciben como mucho recompensa negativa y no pueden compensarse con puntos de formato; si `analyze_figure` no aporta una respuesta válida, la recompensa total queda limitada a −0.75.
- Toda recompensa es **verificable por reglas** (RLVR) y cada trayectoria registra el desglose de `reward_components` para facilitar la auditoría. Los criterios detallados están en la [documentación de la recompensa multigranular](docs/multigranular_rl.md).

### Aislamiento por Episodio

Cada episodio posee su estado de sesión mutable y su directorio de artefactos. Los snapshots y los recursos precargados se comparten en modo de solo lectura; reset/close limpia únicamente rutas propias, nunca archivos del usuario ni de otros episodios. El presupuesto predeterminado es de 6 turnos, incluido FINISH. Solicitar terminar no demuestra que la tarea haya tenido éxito.

---

## Protocolo de Mensajes Compartido

El único protocolo activo es `react-runtime-v2` / `react-messages-v3`: un mensaje user inicial establece tarea/herramientas/setup real, seguido de turnos assistant y Observation del entorno como mensajes user marcados explícitamente como datos. SFT supervisa solo la completion actual. GRPO conserva los IDs muestreados, máscara de política=1 y de entorno/límites insertados=0, verifica estrictamente el prefijo de tokens, conserva EOS existente y no vuelve a multiplicar por G en rollout.

No se declaran compatibles automáticamente datos antiguos/desconocidos ni artefactos entrenados; omitir la comprobación de manifest no evita la validación de protocolo. Los informes separan el protocolo de ejecución y los metadatos reales de entrenamiento del modelo de origen, con relaciones same_protocol/cross_protocol/unknown; unknown no es una reproducción comparable. Aquí no se verificaron prefijos con tokenizer real, gradientes ni capacidad del modelo. Véase el [contrato de ejecución](docs/react_runtime.md).

## Línea de Entrenamiento: SFT → GRPO

Estos comandos corresponden a una ejecución futura separada, no a la aceptación de ingeniería. Requieren datos SFT del protocolo nuevo y su manifest, modelo base/recursos de entrenamiento y un snapshot offline existente. Este cambio no generó esas entradas ni ejecutó entrenamiento.

```bash
# QLoRA por defecto: GRPO carga el adaptador SFT y sus metadatos de modelo base
python -m AgenticArxiv.rl.train_sft \
  --config configs/training/mainline.json \
  --data data/sft/sft_react_messages_v3_train_mix.jsonl \
  --output_dir outputs/sft_qlora
python -m AgenticArxiv.rl.train_grpo \
  --config configs/training/mainline.json \
  --model outputs/sft_qlora/final \
  --task_set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max_turns 6

# Todos los parámetros explícitamente; no usar un directorio de adaptador como modelo completo
python -m AgenticArxiv.rl.train_sft \
  --config configs/training/mainline.json \
  --data data/sft/sft_react_messages_v3_train_mix.jsonl \
  --no-qlora --output_dir outputs/sft_full
python -m AgenticArxiv.rl.train_grpo \
  --config configs/training/mainline.json --model outputs/sft_full/final \
  --no-qlora --task_set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max_turns 6
```

GRPO formal vincula explícitamente un conjunto de tareas y una partición de entrenamiento `FILE:NAME`. `--smoke` selecciona las 8 tareas smoke; las 86 tareas nunca se usan implícitamente todas para entrenar. Las 33 tareas históricas de `v6_grpo_train.json:rl_train` no son las 55 de `v7_86.json:train`.

La configuración mínima conserva valores explícitos del código, valores predeterminados y transmisión condicional de argumentos. Los campos no establecidos de pérdida, muestreo de importancia o escalado de recompensa se delegan en la biblioteca; no se inventan valores efectivos ni equivalencia con recetas históricas. La ausencia de entradas/metadatos produce un error, sin cambiar de modelo o API. La [guía principal](docs/rl_building.md) detalla generación, ampliación y mezcla futuras, recursos, mensajes/máscaras y compatibilidad.

Se conservan controles de longitud, diagnóstico de varianza, Canary, umbral SFT de parseo ≥ 0.3, umbral GRPO de recompensa media ≥ −0.2, precisión y registro. La validación usa episodios completos, no solo la primera acción correcta; aquí no se verificó que modelos reales superen los umbrales.

Se retiraron las entradas activas de DPO, OPD, PPO, variantes GRPO, autoevolución, post-entrenamiento VLM independiente, Jev, MCP/Skill CLI, Web/HTTP, MySQL y SSE. Se conservan inferencia de figuras, construcción de snapshots, backfill T5, puntuación de casos malos y APIs LLM externas.

### Resultados Históricos Publicados (protocolo original, sin repetir)

Los datos publicados y las cifras siguientes conservan su alcance experimental original. No demuestran la capacidad actual bajo el protocolo nuevo. El entrenamiento nuevo rechaza entradas incompatibles; la evaluación entre protocolos debe indicarlo explícitamente.

> Publicado: [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) (Qwen2.5-1.5B a parámetros completos, 2 epochs / 2628 trayectorias expertas, loss 0.079; cubre solo las primeras 8 herramientas).

> Publicado: [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) (33 tareas con señal, replay offline de snapshots de principio a fin). Evaluación offline (seed 45 / repeat 3, tasa de éxito estricta pass³) SFT → GRPO: **rl_train 0.081→0.636, dev 0.000→0.375, iid_test 0.056→0.444, ood_test 0.000→0.500**.

> Publicado: [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) (451 figuras / 73 papers, partición por paper para evitar fugas; holdout de 94 figuras, ROUGE-1 0.105→0.126 y ROUGE-L 0.081→0.102).

---

## 📦 Modelos Publicados

> **Lista de publicaciones históricas.** Las variantes retiradas no son entradas de entrenamiento y no se garantiza el comportamiento de checkpoints antiguos con el contexto nuevo.

| Modelo | Base | Descripción |
|------|------|------|
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) | Qwen2.5-1.5B | Fase 1: SFT a parámetros completos (primeras 8 herramientas) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) | Qwen2.5-1.5B | Fase 3: GRPO (evaluación en las cuatro particiones, más arriba) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO) | Qwen2.5-1.5B | GRPO + muestreo de importancia a nivel de secuencia (empatado con la referencia en las cuatro particiones; ver el doc de comparación) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO) | Qwen2.5-1.5B | Variante sin sesgo de GRPO (rl_train 0.657 / dev 0.375 / iid 0.444 / ood 0.500, pass³) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO) | Qwen2.5-1.5B | Objetivo DAPO sin muestreo dinámico (rl_train 0.667 / dev 0.500 / iid 0.481 / ood 0.500, pass³; la mejor de las tres variantes) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) | Qwen2.5-1.5B | SFT + cadena de cuatro pasos con `analyze_figure` (rl_train 0.485 / dev 0.250 / iid 0.278 / ood 0.250, pass³) |
| [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) | Qwen3-VL-4B | VLM de análisis de figuras del lado del entorno |

Todos se alojan en ModelScope (`Algorineko/AgenticArXiv-RL-*`); las model cards incluyen la tabla de evaluación offline y una descripción honesta de sus limitaciones.

---

## 🧪 Conjunto de Tareas y Evaluación

- **Conjunto de humo** (`benchmark/tasks.py`): 8 tareas (search / download / translate / cache / composite)
- **Conjunto ampliado** (`benchmark/tasks_expanded.py`): 86 tareas y catorce familias de plantillas (search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction / figure_analysis / **translation_reading**); se activa con `run_benchmark.py --task-set expanded`. Ambos conjuntos pasan por el mismo `TaskSpec`: `expected_tools` / `expected_tool_args` se derivan de una única fuente `steps`, de modo que nunca hay dos listas escritas a mano que diverjan.
- **Particiones**: `data/splits/v7_86.json` contiene train 55 / dev 8 / iid_test 19 / ood_test 4. Las 33 tareas históricas de `v6_grpo_train.json:rl_train` proceden de otra partición. Una futura selección de señal usa solo train declarado y escribe un archivo nuevo; dev/iid/ood no entran en entrenamiento y aquí no se volvieron a medir las rates.
- **Criterios de evaluación (cifras históricas)**: fiabilidad `pass^k` (convención tau-bench), `false_finish` (la política degradada `always_finish` alcanza un 91.5% frente al 0% de `reference`), `ref_score` (compara el `paper_id` y no la forma literal del `ref`), coste normalizado por número de aciertos.
- **Puertas de discriminación (estadísticas históricas)** (`run_baselines.py`): políticas degradadas deterministas con umbral por categoría — «buscar siempre cs.AI» baja de 0.833 a 0.446 en la categoría de búsqueda, y «llamar a herramientas cuando no tocaba» pasa de +0.165 a −0.235.
- **Replay de casos malos** (`eval/badcase_replay.py` + `eval/eval_cases.jsonl`, 17 casos): cada trayectoria fallida se congela como caso de regresión permanente; el replay ejecuta solo el evaluador y el propio `pytest` hace de puerta. `hack/*` documenta formas de engañar la puntuación con aserciones de umbral y sirve además de biblioteca de casos de reward hacking.

```bash
python -m AgenticArxiv.benchmark.run_benchmark \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:iid_test \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --output artifacts/eval_react_messages_iid
```

---

## Dependencias

[requirements.txt](AgenticArxiv/requirements.txt) declara las bibliotecas de entrenamiento y `arxiv`, `requests`, `python-dotenv`, `loguru`, `pydantic`, `fire`, PyMuPDF y Pillow. Se conservan las versiones declaradas y los controles de interfaz, sin actualizar ni degradar bibliotecas. Las notas históricas de validación no prueban una integración real actual.

[requirements-extra.txt](AgenticArxiv/requirements-extra.txt) conserva `pdf2zh` para traducción real y `tensorboard` para curvas; `wandb` puede instalarse por separado. Los backends de figuras/resumen necesitan modelos locales explícitos. Una dependencia opcional ausente falla al usar su función, sin reducir las 10 herramientas. Web, bases de datos y otros agentes ya no son opciones de instalación.

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

**P: ¿En qué se diferencia del AgenticArXiv original?** El original es una aplicación web de nivel de producción (FastAPI+Vue3+MySQL, tres arquitecturas de agente); este proyecto es un entorno de entrenamiento RL en Python puro + JSONL que conserva únicamente la política única ReAct: su núcleo son SFT → GRPO y ReAct textual con recompensa verificable.

**P: ¿Por qué GRPO y no PPO?** GRPO no necesita value model (menor consumo de VRAM), encaja con modelos pequeños de la escala de 1.5B y su implementación es simple y fácil de depurar; PPO es más adecuado para entrenar modelos grandes de nivel de producción.

**P: ¿Por qué la calidad del resumen/análisis de figuras no entra en la recompensa?** Puntuar la calidad exigiría LLM-as-judge, lo que introduce no determinismo y una superficie nueva de hacking. El diseño reduce la interpretación a un **problema de decisión de llamada a herramientas** (cuándo llamar, a quién, con qué argumentos) — todo evaluable por reglas y reproducible; es la premisa de RLVR.

---

## Roadmap Histórico (no son tareas actuales)

> **Lo siguiente conserva las direcciones originales, no un plan actual de ejecución o implementación.** Cifras, flags antiguos y rutas mantienen su contexto histórico. Las entradas retiradas no funcionan; los experimentos antiguamente pendientes no forman parte de la aceptación de este cambio. La [guía principal](docs/rl_building.md) recoge comandos futuros y el alcance no ejecutado.


> Los argumentos completos y la ruta de adopción están en [docs/roadmap_notes.md](docs/roadmap_notes.md). Anímate a reclamar una tarea (véase 🤝 Contribuir).

### P0 — Asistente de Papers Multimodal en el Dispositivo (objetivo final)

- [ ] **Multimodalidad del lado de la política**: pasar el VLM del entorno a la política (la observation transporta las figuras), reutilizando la ruta visual-lingüística de TRL ya validada en `train_vlm_figure_qa.py`; bases candidatas Qwen3-VL-4B / Qwen2.5-VL-2B (gama on-device)
- [ ] **Cadena de lectura de extremo a extremo**: búsqueda → descarga → traducción → resumen → análisis de figuras, cerrada dentro de un único modelo y optimizada para la inferencia en el dispositivo (cuantización + escala 2-4B)
- [x] **El texto traducido, en el contexto**: nueva herramienta `get_translated_content`, que lee la traducción de pdf2zh página a página de forma determinista; se reproduce desde snapshots offline (grabados con `build_snapshot --translate-max-ref N`) y se evalúa con la regla de calidad de resultado de las herramientas de lectura

### P1 — Algoritmos de RL Agentic de Nueva Generación

- [x] **Integración de GSPO / Dr.GRPO**: `--importance_sampling_level sequence` (muestreo de importancia a nivel de secuencia, GSPO) y `--loss_type dr_grpo` (normalización por longitud sin sesgo) ya están conectados en `train_grpo.py`, y son ortogonales y combinables con el preset `--dapo`
- [x] **Entrenamiento y comparación con métodos nuevos**: las variantes GSPO / Dr.GRPO / DAPO fueron entrenadas sobre las particiones congeladas y publicadas (receta común alineada con la referencia, seed 42 / 60 pasos; pass³ fuera de línea, GRPO de referencia: rl_train 0.636 / dev 0.375 / iid 0.444 / ood 0.500):
  - **GSPO** (IS a nivel de secuencia): 0.636 / 0.375 / 0.444 / 0.500 — idéntico a la referencia en todas las particiones (pesos distintos, curvas de recompensa divergentes, mismo comportamiento convergido)
  - **Dr.GRPO** (objetivo sin sesgo): 0.657 / 0.375 / 0.444 / 0.500 — ligeramente superior en rl_train; mejor precisión de herramientas en ood (0.67) y menor tasa de finalización falsa (0.33)
  - **DAPO** (clip-higher 0.28 + máscara de truncamiento + β=0; el muestreo dinámico no es viable en el grupo de 33 tareas, por lo que se desactiva): **0.667 / 0.500 / 0.481 / 0.500** — el mejor en rl_train / dev / iid
  - Detalles y notas de implementación en `docs/rl_paradigm_comparison.md`
- [ ] **Entrenamiento asíncrono estilo SAO**: migrar a verl `fully_async_policy` / AReaL, introduciendo primero la máscara skip-observation y el recorte bilateral de DIS ([arXiv:2607.07508](https://arxiv.org/abs/2607.07508), código oficial sin publicar)
- [ ] **Asignación de crédito entre pasos**: ventajas intragrupo que crucen pasos, al estilo GiGPO / ARPO, para aliviar la señal dispersa a nivel de trayectoria de las cadenas largas (línea de investigación propia)

### P2 — Componente de Decisión Discriminativo estilo Jev

- [ ] **Cabeza discriminativa estilo Jev**: decisión estructurada no autorregresiva (choice/score, ~1/400 del coste de un LLM), posibles puntos de integración: ① una cabeza discriminativa para `result_quality` en lugar de reglas ② un enrutador de componentes de recompensa (en lugar del currículo fijo) ③ decodificación estructurada de la selección de herramientas (los enums cerrados encajan de forma natural en lo discriminativo). Restricción: inferencia determinista o uso solo como señal de diagnóstico, para mantener la reproducibilidad de RLVR

### P3 — Autoevolución RSI Acotada (bounded RSI)

- [ ] **Bucle de datos autoevolutivo**: la evaluación holdout expone debilidades → la biblioteca de casos malos se amplía automáticamente → derivación paramétrica de datos dirigidos → reentrenamiento → congelación de un nuevo holdout (reutiliza las líneas existentes de evaluación/datos/entrenamiento; los prerrequisitos ya están listos). Criterio de aceptación: el pass³ en las cuatro particiones no retrocede en ninguna ronda y la biblioteca de casos malos solo crece (ciclo de cuatro pasos de [arXiv:2609.11873](https://arxiv.org/abs/2609.11873)). **Orquestador implementado** (`AgenticArxiv/rl/self_evolve.py`: diagnose → mine → select → synthesize → freeze → gate, solo JSON, sin modelo; `generate_parametric_sft_data.py --only-parents` para datos dirigidos); queda pendiente una ronda real de entrenamiento con la comparación en cuatro particiones (ver [docs/roadmap_notes.md](docs/roadmap_notes.md))

### ⛔ Bloqueos del Entorno

- [ ] **Muestreo acelerado con vLLM**: TRL 0.29 exige vLLM 0.10.2–0.12.0, y la compilación personalizada 0.6.2 de la plataforma local, instalada en conjunto, hace que la importación de GRPOTrainer falle. Es una dependencia del entorno, no un cambio de código: queda a la espera de que la plataforma ofrezca una compilación compatible.

---

**¡Comienza tu viaje de entrenamiento en RL Agentic!** 🚀
