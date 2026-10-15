FROM --platform=linux/amd64 python:3.10.19-bookworm@sha256:c8e004af3b763d720985b2bbb75825e28639b7620de4ff45e9e6e98c6e129809

ARG CODE_IDENTITY=unverified
LABEL org.opencontainers.image.title="agentic-arxiv-training" \
      org.opencontainers.image.source="https://github.com/Arkalin/AgenticArXiv-RL" \
      io.agentic-arxiv.code-identity="${CODE_IDENTITY}" \
      io.agentic-arxiv.recovery-protocol="1"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

COPY containers/requirements-linux-py310-cu126.lock /opt/agentic-build/requirements.lock
COPY containers/base-image.json /opt/agentic-build/base-image.json
COPY containers/lock-resolution.json /opt/agentic-build/lock-resolution.json
COPY build-identity.json /opt/agentic-build/build-identity.json
RUN CODE_IDENTITY="${CODE_IDENTITY}" python -c 'import json, os; m=json.load(open("/opt/agentic-build/build-identity.json")); assert m["tree_sha256"] == os.environ["CODE_IDENTITY"], "构建身份必须显式匹配清单"'
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r /opt/agentic-build/requirements.lock

COPY pyproject.toml README.md LICENSE /tmp/project/
COPY src/agentic_arxiv /tmp/project/src/agentic_arxiv
RUN python -c 'import hashlib,json,pathlib; r=pathlib.Path("/tmp/project"); m=json.load(open("/opt/agentic-build/build-identity.json")); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in m["files"].items() if p.startswith("src/") or p in ("pyproject.toml","README.md","LICENSE")), "代码与构建身份不一致"'
RUN cd /tmp/project && python -m build --no-isolation --wheel --outdir /opt/agentic-build/wheels \
    && python -m pip install --no-cache-dir --no-deps /opt/agentic-build/wheels/*.whl \
    && python -m pip check \
    && python -m pip inspect > /opt/agentic-build/installed-dependencies.json \
    && cd / && python -c 'import agentic_arxiv; assert "/site-packages/" in agentic_arxiv.__file__' \
    && agentic-arxiv --help > /opt/agentic-build/cli-help.txt \
    && rm -r /tmp/project

WORKDIR /tmp
USER 65532:65532
ENTRYPOINT ["agentic-arxiv"]
CMD ["--help"]
