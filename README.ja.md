<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a> | <a href="README.ja.md">🇯🇵 日本語</a>
</p>

# AgenticArXiv-RL — エージェント強化学習の訓練環境

> **ReAct エージェントと arXiv ツールを基盤とする Agentic RL 訓練環境**
> SFT/DPO/GRPO/OPD の訓練経路と検証可能な報酬（RLVR）に対応し、LLM エージェントの強化学習を研究します。
> 最終目標は、**端末上で動作する軽量マルチモーダル論文アシスタント**。検索・ダウンロード・翻訳から要約・図表理解までを担う、エンドツーエンドの論文読解モデルです。

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="AgenticArXiv-RL プロジェクトの概要" width="800"/>
</p>

---

## 🎯 プロジェクトの位置づけ

arXiv 論文の検索・ダウンロード・翻訳・読解タスクを、**訓練可能な強化学習環境**として構成します。主な特徴は次のとおりです。

1. **検証可能な報酬（Verifiable Reward）**：ツール呼び出しの正確さ、タスク達成度、解析エラーなどをルールで採点し、人手によるアノテーションを不要にします。
2. **段階的な訓練**：SFT → DPO → GRPO。OPD によるオンポリシー蒸留も利用できます。PPO は現在の依存関係では利用できません。
3. **軽量な構成**：Python と JSONL ストレージのみを使い、データベースやフロントエンドを必要とせず、オフライン訓練に集中します。

**対象外**：本番向け arXiv アプリ、リアルタイム翻訳サービス。`AgenticArxivWeb/` はこのリポジトリに残し、RSI における環境との共進化のためのサンドボックスとして改造していく予定です（ロードマップ P3 参照）。元プロジェクトの [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) を参照してください。環境を RL 訓練環境へ改造した設計の全体像は [docs/rl_building.md](docs/rl_building.md) にあります。

### 任意で利用できる Jev ツールルーティング実験

既定で無効化されている識別型の次ツールルーター（`TOOL_ROUTER=jev`）：小規模な構造検証では、厳密成功率が 40% から 60% に上がり、ローカル Qwen のトークン使用量は 74% 減少しました。GRPO 訓練には組み込まれていません。

<details>
<summary>展開：Jev ルーティングの仕組み、フォールバック方針、実験の詳細</summary>

通常の ReAct 推論では、次に使うツールを Jev が選ぶルーターを任意で有効にできます。
guided モードでは、明示的な引数を決定論的なパーサーで処理し、曖昧な引数は方策モデルに任せ、
ツールの実行は環境が担当します。既定の `TOOL_ROUTER=policy` は従来の動作を維持します。
`jev` を指定すると有効になり、信頼度の不足、API 障害、引数検証の失敗時には元の方策経路に
フォールバックします。この機能は GRPO 訓練には組み込まれていません。設定、ベースライン、
81 タスクのルーティングのみの予備実験、制約、再現手順は [JEV_ROUTER.md](JEV_ROUTER.md)、
コピーして使える切り替え・API 設定は [jev_config.example.env](jev_config.example.env) を参照してください。
GPU やネットワークを使わないインターフェースのスモークテストもあり、方策経路と Jev 経路を
同じ ReAct・環境・ベンチマークのインターフェースで切り替えられることを確認します。
GRPO で訓練した Qwen を用いた混合 10 タスク・3 回反復の小規模評価では、guided Jev により
厳密成功率が 40% から 60% に上がり、ローカル Qwen のトークン使用量が 74.01% 減少した一方、
総レイテンシは 36.96% 増加しました。これは少数サンプルによる統合の検証であり、
全 81 タスクでの総合的な成功率を示すものではありません。

</details>

---

## 🚀 クイックスタート

```bash
# 1. クローンとインストール（Python 3.9+）
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
python3 -m venv .venv && source .venv/bin/activate
pip install -r AgenticArxiv/requirements.txt

# 2. LLM API を設定（API による rollout に必要。ローカルモデルの訓練・評価には不要）
cat > AgenticArxiv/.env << 'EOF'
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo
EOF

# 3. rollout を試す（報酬は [-1, 1] の範囲で軌跡に応じて変化）
python -m AgenticArxiv.rl.rollout search_01 traces/train/
# ✅ タスク search_01 の rollout が完了  報酬: 1.00
```

一括 rollout：`python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/`。以降のコマンドは、特記がない限りリポジトリのルートで実行します。

---

## 📚 基本概念

論文タスクを MDP として構成します。10 個のツールからなる行動空間 + 5 成分の検証可能報酬（RLVR）+ rollout の分離。すべてルールで判定でき、オフラインで再現可能です。

<details>
<summary>展開：MDP の設計 · 行動空間（10 個のツール）· 5 成分の検証可能報酬 · rollout の分離</summary>

### MDP の設計

| 要素 | 定義 |
|------|------|
| **State** | タスクの説明 + 対話履歴 + ツールの実行結果 |
| **Action** | 10 個のツール（下記参照）+ FINISH |
| **Reward** | 5 成分からなる多粒度の検証可能報酬（format / tool / argument / process / outcome） |
| **Transition** | `execute_tool(action) → observation`（`MockArxivEnv` がオフラインスナップショットを再生し、決定論的に再現可能） |

### 行動空間（10 個のツール）

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — 研究分野と期間を指定して論文を閲覧
2. `search_arxiv_papers(query, max_results, days=None)` — キーワード・タイトル・著者による検索
3. `download_arxiv_pdf(ref, session_id)` — PDF をダウンロード
4. `translate_arxiv_pdf(ref, session_id)` — PDF を翻訳（pdf2zh）
5. `get_paper_cache_status(ref, session_id)` — キャッシュの状態を確認
6. `get_paper_content(ref, session_id, section=None)` — 要旨または指定した節を読み取り（決定論的に抽出）
7. `summarize_paper(ref, style, max_words)` — 環境側で要約を生成（tldr / structured / bullet）
8. `extract_paper_figures(ref)` — 図表ファイルとキャプションを抽出
9. `analyze_figure(ref, figure_no, question=None)` — 環境側でローカル VLM を呼び出して図表を分析（マルチモーダル環境で有効）
10. `get_translated_content(ref, session_id, page=1)` — 翻訳後の中国語 PDF から訳文をページ単位で読み取り（決定論的に抽出。1 ページ目は通常、タイトルと要旨）

> 「検索 → ダウンロード → 読む → 要約 → 図表抽出 → 図表分析」の読解ループは接続済みです。図表分析は、本プロジェクトで事後訓練した [FigureQA VLM](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) が環境側で担当し、結果をスナップショットに記録します。方策側には、図表分析の 4 ステップの連鎖を学習した [SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) の重みがあります。
> **ツールの採用基準**：行動空間は大きければよいわけではありません。新しいツールは「新たな種類のタスクを可能にする」場合に採用します。設計判断の詳細は [ツールセットの進化設計](docs/toolset_evolution.md) を参照してください。

### 検証可能な報酬（5 成分）

**多粒度の検証可能報酬**（`rl/reward.py`）を使用します。5 成分を `[-1, 1]` に正規化し、診断用の成分と、他の得点で相殺できない失敗時の報酬上限を設けています。

| 成分 | 既定の重み | 評価する信号 |
|------|:---:|------|
| `format` | 1 | 各ステップの行動が有効な JSON ツール呼び出しまたは終了記号かどうか |
| `tool` | 3 | 予測したツール列と期待するツール列の**順序を考慮した LCS-F1** |
| `argument` | 2 | 引数キーの再現率 × 値の完全一致精度 |
| `process` | 1 | 有効なステップへの加点 − 解析失敗・実行失敗・余分な呼び出しへの減点 |
| `outcome` | 3 | 正しい完了 +1、誤った経路での完了 +0.25、強制停止 −0.5、エラー −1 |
| `result_quality` / `efficiency` | 診断ゲート | 観測結果が実際に成功を示すかを確認。著しい冗長性があれば報酬を制限 |

- **カリキュラム学習**：最初の 30 ステップは `tool` / `argument` / `outcome` の重みを 1/3 にします（`RewardCalculator.schedule`）。実測では、SFT 済みモデルはすでに ReAct の構造を学習しており（初期 format は 0.983）、カリキュラムの有無による差は確認できませんでした。**SFT から開始する場合は `--reward_curriculum_steps 0` を指定**し、このスケジュールはコールドスタート向けに残しています。
- **相殺できない失敗**：解析失敗、実行失敗、虚偽の FINISH は負の報酬に制限され、形式の得点で相殺できません。`analyze_figure` に有効な回答がない場合、総報酬の上限は −0.75 です。
- すべての報酬は**ルールに基づいて検証可能**（RLVR）です。各軌跡に `reward_components` の内訳を記録し、監査できるようにします。判定基準は [多粒度報酬のドキュメント](docs/multigranular_rl.md) を参照してください。

### rollout の分離

`rl/sandbox.py` の `RolloutSandbox` は、各軌跡の開始前に環境・Store・生成物ディレクトリの基準状態を記録し、終了後に復元と後片付けを行います。複数ターンの GRPO、rollout、ベンチマークが同じリセット規約を使い、軌跡間で状態が引き継がれることを防ぎます。

</details>

---

## 🛠️ 訓練経路（SFT → DPO → GRPO / OPD）

SFT → DPO → GRPO の段階的訓練。OPD によるオンポリシー蒸留は GRPO の代替となり、PPO は現在の依存関係では利用できません。GSPO / Dr.GRPO / DAPO の目的関数変種は実装済みで、対照訓練を完了し重みも公開しています。

<details>
<summary>展開：各ステージのコマンドと説明（SFT · DPO · GRPO と変種 · OPD · マルチモーダル VLM）</summary>

### ステージ 1：SFT

```bash
python scripts/generate_parametric_sft_data.py   # パラメトリックに専門家軌跡を生成（API 不要）
python -m AgenticArxiv.rl.train_sft              # outputs/sft/final を生成
```

> 公開済み：[AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT)（Qwen2.5-1.5B の全パラメータを訓練。2 エポック / 専門家軌跡 2628 件、loss 0.079。最初の 8 ツールのみを対象）。

### ステージ 2：DPO

```bash
python scripts/generate_dpo_data.py --model outputs/sft/final --num_rollouts_per_task 8
python -m AgenticArxiv.rl.train_dpo              # outputs/dpo/final を生成
```

選好ペアは SFT モデルによるローカルサンプリングから作成します（`LLM_API_KEY` は不要）。スナップショットがあれば自動的にオフライン再生し、報酬差が `--min_reward_gap` を超え、最初のツールが異なる軌跡だけをペアにします。

### ステージ 3：GRPO

```bash
python -m AgenticArxiv.rl.build_snapshot         # オフラインスナップショットを作成（唯一のネットワーク接続工程）
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --max_turns 4

# DAPO プリセット（loss_type=dapo + clip-higher + 過長出力フィルタリング + 動的サンプリング）
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --dapo

# 系列単位の重要度サンプリング（GSPO）と、バイアスを補正する Dr.GRPO
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --importance_sampling_level sequence
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --loss_type dr_grpo

# 複数 GPU（DDP は 2 GPU で動作確認済み。FSDP は torch>=2.6 が必要）
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --no-qlora

# 訓練曲線
python -m AgenticArxiv.rl.train_grpo --report_to tensorboard
tensorboard --logdir outputs/grpo/logs
```

> 公開済み：[AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO)（学習信号のある 33 タスクを使用し、全工程でオフラインスナップショットを再生）。オフライン評価（seed 45 / 3 回反復、厳密成功率 pass³）での SFT → GRPO：**rl_train 0.081→0.636、dev 0.000→0.375、iid_test 0.056→0.444、ood_test 0.000→0.500**。

> **目的関数変種の対照結果**（seed 42 / 60 ステップ、pass³、重みはいずれも公開済み — 下表参照）：GSPO はすべての分割でベースラインと同等（軌跡は分岐するが収束後の振る舞いは同じ）。Dr.GRPO は rl_train で 0.657 とわずかに改善し、ood のツール精度が最良。**DAPO は 3 分割で最良（0.667 / 0.500 / 0.481 / 0.500）**。実装の注記は [docs/rl_paradigm_comparison.md](docs/rl_paradigm_comparison.md)。**報酬重みのアブレーション框架**：`--reward_config` により 9 種類の単一変数の重み設計をサポート（[docs/ablation_reward_weights.md](docs/ablation_reward_weights.md)、🟡 設計完了・実行待ち）。

**複数ターンの rollout と採点**（`rl/grpo_reward.py`）：各ターンで現在の方策が ReAct の行動を生成し、独立した `MockArxivEnv` が実行して観測結果をコンテキストに戻します。assistant のトークンは損失計算に含め、環境のトークンは `env_mask=0` によってコンテキストとしてのみ使います。軌跡全体は 5 成分の `RewardCalculator` で採点し、rollout・ベンチマークと同じ基準を適用します。

**訓練品質の保護機構**：生成長の検査、ゼロ分散の監視（`RewardVarianceGuard`）、訓練途中の Canary 評価による早期停止、ステージごとの検証しきい値（SFT 解析可能率 ≥ 0.3、DPO 報酬 ≥ −0.3、GRPO 報酬 ≥ −0.2）、混合精度の自動調整、ログバックエンドの事前検証により、失敗を明示的なエラーとして検出します。TRL 標準の指標に加えて、`reward_components/*`（成分内訳）、`reward_weights/*`（カリキュラムの重み）、`rollout/*`（turns / finished / parse_error_rate）を記録します。

### ステージ 3'：OPD（オンポリシー蒸留、任意。GRPO の代替）

```bash
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final \
  --teacher Qwen/Qwen2.5-7B-Instruct --max_turns 4 --snapshot data/mock_arxiv_snapshot.json
```

生徒モデルがタスクのプロンプトに対してオンポリシーでサンプリングし、教師モデルがトークンごとの対数確率を与えます。損失は逆 KL（mode-seeking）です。GRPO との比較は次のとおりです。

| 観点 | GRPO | OPD |
|---|---|---|
| 学習信号 | 検証可能な報酬（疎、軌跡単位） | 教師のトークンごとの対数確率（密） |
| 追加モデル | 不要 | 教師モデル（ローカルの重みが必要） |
| 到達範囲 | 探索によって教師を超えられる | 教師の振る舞いに収束 |
| 適した条件 | 検証可能な報酬がある | 強い教師があり、探索コストを抑えたい |

### ステージ 4：PPO — ⚠️ 現在の依存関係では利用不可

TRL では従来の PPOTrainer が削除されており、`trl>=0.28.0` には存在しません。`train_ppo.py` はインポート時に理由を表示して終了します。検証可能な報酬があれば GRPO、強い教師があれば OPD を使います。どちらも PPO より GPU メモリ消費を抑えられます。

### マルチモーダルの事後訓練（環境側 VLM）

```bash
python scripts/build_figure_qa_dataset.py                          # 図表 QA のシードデータ
python -m AgenticArxiv.rl.train_vlm_figure_qa                      # Qwen3-VL-4B の LoRA 事後訓練
# /path/to/figureqa をローカルの FigureQA モデルディレクトリに置き換える
FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH="/path/to/figureqa" \
  python -m AgenticArxiv.rl.backfill_figure_analysis --snapshot data/mock_arxiv_snapshot.json
```

> 公開済み：[AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA)（73 論文 / 451 図表。情報漏洩を防ぐため論文単位で分割。ホールドアウト 94 図表で ROUGE-1 は 0.105→0.126、ROUGE-L は 0.081→0.102）。8B / 27B の方策側 VLM の全経路の成果は、下記「🏆 端末向け VLM の事後訓練成果」を参照してください。

</details>

---

## 📦 公開モデル

| モデル | ベースモデル | 説明 |
|------|------|------|
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) | Qwen2.5-1.5B | ステージ 1：全パラメータ SFT（最初の 8 ツール） |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) | Qwen2.5-1.5B | ステージ 3：GRPO（4 分割での評価は上記参照） |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO) | Qwen2.5-1.5B | GRPO + 系列単位の重要度サンプリング（4 分割すべてでベースラインと同等。方式比較のドキュメントを参照） |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO) | Qwen2.5-1.5B | バイアスを補正する GRPO（rl_train 0.657 / dev 0.375 / iid 0.444 / ood 0.500、pass³） |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO) | Qwen2.5-1.5B | 動的サンプリングなしの DAPO 目的関数（rl_train 0.667 / dev 0.500 / iid 0.481 / ood 0.500、pass³。3 変種中で最高） |
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) | Qwen2.5-1.5B | SFT + `analyze_figure` の 4 ステップ連鎖（rl_train 0.485 / dev 0.250 / iid 0.278 / ood 0.250、pass³） |
| [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) | Qwen3-VL-4B | 環境側の図表分析 VLM |
| [AgenticArXiv-RL-Qwen3-VL-8B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-8B-GRPO) | Qwen3-VL-8B | **端末向け論文アシスタントの主モデル**：方策側マルチモーダル、SFT→DPO→GRPO の全経路（81 タスクの厳密成功率 72.8%、base の 2.9 倍。図表 ROUGE-1 +32%） |
| [AgenticArXiv-RL-Qwen3.8-27B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3.8-27B-SFT) | Qwen3.8-27B | 図表理解に特化した研究版（clean-val ROUGE +60%/+87%。ツール選択は受け入れ基準に未達。実験記録を参照） |

すべて ModelScope（`Algorineko/AgenticArXiv-RL-*`）で公開しています。各モデルカードにはオフライン評価表と制約を記載しています。

---

## 🏆 端末向け VLM の事後訓練成果（Qwen3-VL-8B · Qwen3.8-27B）

4×64GB の Hygon DCU 上で、Qwen3-VL-8B の **SFT → DPO → GRPO 全経路**と、
ネイティブマルチモーダルモデル Qwen3.8-27B の図表理解に特化した SFT を完了しました。
データは clipart の除去とツールマニュアルの正規化という 2 段階で整備しました。
評価は共通の**貪欲デコード（greedy）プロトコル**を使い、81 タスクのオフラインベンチマークと
clean-val の図表 ROUGE を測定しています。

| 段階 | 81 タスクの厳密成功率 | 完了率 | ツール精度 | 虚偽終了率 | clean ROUGE-1 |
|---|---|---|---|---|---|
| 8B base | 24.7% | 80.2% | 32.1% | 22.2% | 0.1139 |
| 8B SFT v3 | 66.7% | 95.1% | 72.8% | 8.6% | 0.1446（+27%） |
| 8B DPO v3 | 65.4% | 97.5% | 74.1% | 11.1% | 0.1435 |
| **8B GRPO v3（主モデル）** | **72.8%** | **97.5%** | **82.7%** | **6.2%** | **0.1509（+32%）** |
| 27B base（対照） | 24.7% | 80.2% | 32.1% | 22.2% | 0.1349 |
| 27B cap SFT（研究版） | 21.0% | 66.7% | 25.9% | **4.9%** | **0.2164（+60%）** |

**主モデル：8B GRPO（厳密成功率 72.8%）は 27B base の 2.9 倍**で、
厳密成功率とツール精度が上表で最高となりました。独立した 2 回の実行では厳密成功率が
71.6% と 72.8% となり（タスク結果の反転率 4.9%）、図表理解の clean-val ROUGE-1 は
8B base に対して 32% 向上しました。容量を増やした 27B アダプター（r128）も研究版として公開し、
図表理解の ROUGE は +60%/+87% とプロジェクト内で最高の改善を示しました。
強い事前知識を持つ 27B に少量データの LoRA を適用する際の限界と、3 段階の原因分析
（clipart 汚染 / レジストリのずれ / サンプリングプロトコル）は、
**[docs/edge-vlm-posttrain.md](docs/edge-vlm-posttrain.md)** に記録しています
（モデル選定・データの変遷・4 ラウンドの実験・原因分析）。

---

## 🧪 タスクセットと評価

- **スモークテスト用セット**（`benchmark/tasks.py`）：8 タスク（search / download / translate / cache / composite）。
- **拡張セット**（`benchmark/tasks_expanded.py`）：86 タスク、14 種類のテンプレート（search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction / figure_analysis / **translation_reading**）。`run_benchmark.py --task-set expanded` で有効にします。両セットで共通の `TaskSpec` を使い、`expected_tools` / `expected_tool_args` を同じ `steps` から導出して、手書きの二重管理による不整合を防ぎます。
- **データ分割**：テンプレート単位で train / iid_test / ood_test に分割します（`benchmark/splits.py`、`data/splits/` に固定）。`rl_train` は成功率が中間帯のタスクのみを使います。両端ではグループ内分散がゼロとなり、勾配が生じないためです。モデルを変更したら成功率を再測定し、以前の区分を流用しないようにします。
- **評価指標**：`pass^k` による信頼性（tau-bench の定義）、`false_finish`（退化方策 `always_finish` は 91.5%、`reference` は 0%）、`ref_score`（`ref` の表記ではなく `paper_id` を比較）、成功回数で正規化したコスト。
- **識別力の検証ゲート**（`run_baselines.py`）：決定論的な退化方策をカテゴリ別のしきい値で検査します。「常に cs.AI を検索」は検索カテゴリで 0.833→0.446、「不要な場面でツールを呼ぶ」は +0.165→−0.235 となります。
- **失敗例の再生**（`eval/badcase_replay.py` + `eval/eval_cases.jsonl`、17 件）：失敗軌跡を恒久的な回帰テストとして固定し、再生時は採点器のみを実行します。`pytest` が検証ゲートとなり、`hack/*` には報酬の抜け道としきい値のアサーションを記録して、reward hacking の事例集としても使います。

```bash
python -m AgenticArxiv.benchmark.run_benchmark --task-set expanded --split iid_test --offline
```

---

## 🛡️ 依存関係

**主要な依存関係**（`AgenticArxiv/requirements.txt`）：`torch>=2.0`、`transformers>=4.45`、`trl>=0.28.0`（0.29.1 で検証済み。下限は複数ターン GRPO の `rollout_func` 経路による）、`peft`、`bitsandbytes`（QLoRA）、`datasets`、`accelerate`、`arxiv`、`requests`、`python-dotenv`、`loguru`、`pydantic>=2`、`fire`、`PyMuPDF>=1.24.0`、`Pillow`。

**任意の依存関係**（`AgenticArxiv/requirements-extra.txt`）：`pdf2zh`（実際の PDF 翻訳）、`fastapi` / `uvicorn` / `sqlalchemy` / `pymysql`（アーカイブされた Web 互換レイヤー）、`mcp`（アーカイブされた MCP 互換レイヤー）、`tensorboard` / `wandb`（訓練曲線の記録）、`matplotlib` / `numpy` / `pandas`（描画スクリプト）。

---

## 🔗 関連リソース

- **ドキュメント**：[ツールセットの進化設計](docs/toolset_evolution.md) · [多粒度報酬](docs/multigranular_rl.md) · [指標の統計](docs/metric_stats.md) · [ロードマップの調査ノート](docs/roadmap_notes.md) · [TRL ドキュメント](https://huggingface.co/docs/trl/)
- **手法に関する論文**：RLVR、DPO（Stanford、2023）、[On-Policy Distillation（Thinking Machines、2025）](https://thinkingmachines.ai/blog/on-policy-distillation/)、[GKD（arXiv:2306.13649）](https://arxiv.org/abs/2306.13649)
- **元プロジェクト**：[AgenticArXiv](https://github.com/Algorineko/AgenticArXiv)（Web アプリ版、FastAPI + Vue3 + MySQL）

---

## 🤝 コントリビューション

Issue や PR を歓迎します。手順は Fork → feature ブランチ → `pytest AgenticArxiv/tests/` → `feat:` プレフィックスのコミット → PR です。詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## 📄 ライセンス

MIT License

---

## 🙋 よくある質問

<details>
<summary>展開：よくある質問（元プロジェクトとの違い · GRPO を選ぶ理由 · 品質を報酬に含めない理由）</summary>

**Q：元の AgenticArXiv との違いは？** 元プロジェクトは本番向けの Web アプリ（FastAPI + Vue3 + MySQL、3 種類のエージェント構成）です。本プロジェクトは Python + JSONL の RL 訓練環境で、ReAct の単一方策だけを残し、SFT/DPO/GRPO 訓練と検証可能な報酬を中心にしています。

**Q：PPO ではなく GRPO を選ぶ理由は？** GRPO は価値モデルが不要で GPU メモリの消費が少なく、1.5B 規模の小型モデルに適し、実装とデバッグも容易です。PPO は本番向けの大規模モデル訓練により適しています。

**Q：要約や図表分析の品質を報酬に含めない理由は？** 品質の採点に LLM-as-judge を使うと、非決定性と報酬の抜け道が増えます。本設計では読解を**ツール呼び出しの意思決定問題**（いつ、どのツールを、どの引数で呼ぶか）として扱います。すべてをルールで判定でき、再現可能であることが RLVR の前提です。

</details>

---

## 📝 ロードマップ

> 詳細な根拠と導入経路は [docs/roadmap_notes.md](docs/roadmap_notes.md) を参照してください。完了した項目は本文の該当セクションに統合済みです。各項目への参加を歓迎します（🤝 コントリビューション参照）。

<details>
<summary>展開：ロードマップ（P0 端末でのループ完結 · P1 新アルゴリズム · P2 Jev · P3 共進化 · ⛔ 阻害要因）</summary>

### P0 — 端末向けマルチモーダル論文アシスタント（最終目標）

- [ ] **エンドツーエンドの読解連鎖**：検索 → ダウンロード → 翻訳 → 要約 → 図表分析を単一モデルで完結させ、端末上の推論に向けて最適化します（量子化 + 2–4B 規模）。翻訳はすでに利用可能（`get_translated_content` による pdf2zh 訳文のページ単位の決定論的読み取り + `build_snapshot --translate-max-ref N` で記録したスナップショットの再生）。方策側のマルチモーダル化は 8B で全経路を完了済み（「🏆 端末向け VLM の事後訓練成果」と [VLM 事後訓練の記録](docs/edge-vlm-posttrain.md) 参照）。残る核心は、チェーン全体を端末の予算に収めることです。

### P1 — 次世代の Agentic RL アルゴリズム

- [ ] **SAO 型の非同期訓練**：verl の `fully_async_policy` / AReaL に移行し、まず skip-observation マスクと DIS の両側クリッピングを導入します（[arXiv:2607.07508](https://arxiv.org/abs/2607.07508)、公式コードは未公開）。
- [ ] **ステップ間のクレジット割り当て**：GiGPO / ARPO 型のグループ内・ステップ間のアドバンテージを導入し、長い行動連鎖における軌跡単位の疎な学習信号を改善します（独自の研究課題）。
- [ ] **Agentic TTT（テスト時訓練・適応）**：TTT は具身 AI で広く使われていますが、LLM エージェントのレイヤーで成立するかは開かれた問題です。本リポジトリでの形：単一論文のエピソード内で QA・要約の試行を自己生成 → 多数決/検証器による擬似報酬（TTRL、[arXiv:2504.16084](https://arxiv.org/abs/2504.16084)）→ 数ステップの LoRA 更新 → 再回答し、長コンテキスト ICL のベースラインと比較して「即時の特化」が割に合うかを検証します。制約：端末のレイテンシ予算、壊滅的な忘却、TTT の非決定性と RLVR の再現可能な評価との衝突、オフラインスナップショット下で「テスト分布で訓練する」ことの位置づけ。

### P2 — Jev 型の識別的意思決定コンポーネント

- [ ] **Jev 型の識別ヘッド**：非自己回帰の構造化された意思決定（choice/score、コストは LLM の約 1/400）。候補は ① ルールを置き換える `result_quality` 識別ヘッド ② 固定カリキュラムを置き換える報酬成分ルーター ③ ツール選択の構造化デコードです。閉じた選択肢の集合は識別モデルに適しています。RLVR の再現性を保つため、推論を決定論的にするか、診断信号としてのみ使います。
- [ ] **RLCD による Jev の自己訓練**：成功/失敗のツール決定軌跡からの対照信号（検証可能な報酬で人手ラベル不要）で識別ヘッドを訓練し、外部 TypeSafe API を置き換えます。**2 アーム比較で検証済み**（[docs/jev_rlcd.md](docs/jev_rlcd.md)）：対照項は CE 単独のベースラインに 4 スイートすべてで勝利（992 凍結ポイント、McNemar p<0.0001、Δ +33.0pp）。チェーン中間ステップ 91.0%、0.80 ゲート内で高精度。複合チェーンの最初のステップ 53.1% が既知の弱点（ゲート下で安全にフォールバック）。残タスク：最初のステップの targeted 強化、`TOOL_ROUTER=local` の配線

### P3 — 制約付き自己進化（bounded RSI）

- [ ] **自己進化するデータループ**：ホールドアウト評価で弱点を特定 → 失敗例を自動追加 → 対象を絞ったデータをパラメトリックに生成 → 再訓練 → 新しいホールドアウトを固定します。既存の評価・データ・訓練パイプラインを再利用でき、前提機能は整っています。受け入れ基準は、各ラウンドで 4 分割の pass³ が低下せず、失敗例集が増え続けることです（[arXiv:2609.11873](https://arxiv.org/abs/2609.11873) の 4 ステップループ）。**オーケストレーターは実装済み**（`AgenticArxiv/rl/self_evolve.py`：diagnose → mine → select → synthesize → freeze → gate。JSON のみでモデル不要。`generate_parametric_sft_data.py --only-parents` で対象を絞ったデータを補充）。実際の訓練 1 ラウンドと 4 分割での比較は未実施です（[docs/roadmap_notes.md](docs/roadmap_notes.md) 参照）。
- [ ] **ハーネスと環境の共進化**：自己進化は重みにとどまらず（データループは上記）— ハーネス側：プロンプト / ツールルーティング / ワークフローの DGM 式自己改造とアーカイブ（[Darwin Gödel Machine、arXiv:2505.22954](https://arxiv.org/abs/2505.22954)）。環境側：タスク分布 / 報酬関数の自己生成（Absolute-Zero 式 self-play、[arXiv:2505.03335](https://arxiv.org/abs/2505.03335)）。`AgenticArxivWeb/` を改造した進化可能なタスク・報酬サンドボックスを環境の担い手とします。制約付きの規律は変わりません：manifest の SHA256 監査、4 分割のホールドアウトゲート、人手による昇格。

### ⛔ 環境上の阻害要因

- [ ] **vLLM によるサンプリング高速化**：TRL 0.29 は vLLM 0.10.2–0.12.0 を必要とします。実験環境にインストールされているプラットフォーム独自版 0.6.2 と併用すると、GRPOTrainer のインポートが失敗します。環境の依存関係による問題であり、対応するビルドの提供待ちです。

</details>

---

**Agentic RL の訓練を始めましょう！** 🚀

---

## 🌟 コントリビューター

このプロジェクトをより良いものにしてくれたすべての貢献者に感謝します：

<a href="https://github.com/Algorineko/AgenticArXiv-RL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Algorineko/AgenticArXiv-RL" alt="コントリビューター"/>
</a>
