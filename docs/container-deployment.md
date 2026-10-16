# 固定训练镜像与服务器部署前提

本页描述人工准备操作，不授予执行权限。实验管理器不会执行系统安装、用户授权、存储迁移、镜像构建/拉取或输入下载。人工部署与启动实验是独立授权步骤；服务器实际结果以各次部署目录中的检查报告为准。

## 固定环境与现有证据

- 目标平台为 `linux/amd64`、Python 3.10.19。官方 Python `bookworm` 基础镜像按平台 manifest 摘要固定，来源和查询日期见 `containers/base-image.json`。选用包含系统运行库的完整基础镜像，不在 Dockerfile 中使用未固定的 apt 仓库安装。
- PyTorch 2.10.0+cu126 来自 [PyTorch 官方 cu126 wheel 源](https://pytorch.org/get-started/previous-versions/)，CUDA 用户态依赖由其 wheel 闭包提供，而非依赖宿主机安装 CUDA toolkit。
- Transformers 4.57.6、TRL 0.29.1、PEFT 0.17.1、bitsandbytes 0.45.5、Accelerate 1.11.0 保持已有基线。`containers/requirements-linux-py310-cu126.lock` 是训练、日志及构建工具的完整解析闭包，94 个包均记录精确版本、分发文件 SHA256 和解析源，包含 NVIDIA runtime、NumPy、datasets 等传递依赖。
- 该锁文件面向 Linux x86_64 / Python 3.10.19，锁本身不是训练库接口、镜像安装或 GPU 兼容性通过证明。不同版本的分发文件有多个哈希，安装只允许对应平台 wheel，禁止退回源码构建。
- 客户端本地 Docker daemon 为 ARM64，服务器为 AMD64。应在目标平台核验镜像；不能把基础镜像摘要填成训练镜像身份，实际 profile 使用构建并推送后获得的训练镜像 RepoDigest。

## 更新锁文件

下面命令仅解析依赖元数据。使用 uv 0.11.6，禁止下载额外 Python；不会在当前 Python 环境安装训练栈。更新不是普通启动的一部分，更新后应检查差异并重跑接口与恢复验证。

```sh
UV_PYTHON_DOWNLOADS=never uv pip compile pyproject.toml containers/training-baseline.in \
  --extra train --extra logging --torch-backend cu126 \
  --python-version 3.10.19 --python-platform x86_64-manylinux_2_28 \
  --no-build --generate-hashes --emit-index-url --emit-index-annotation \
  --output-file containers/requirements-linux-py310-cu126.lock
```

## 获得独立构建授权后

必须先确认目标架构、Docker 磁盘及网络条件，再执行镜像准备。以下是操作模板，不是实际执行记录。构建可能下载数 GB CUDA wheel。构建上下文使用白名单复制，只包括源码、包元数据、锁和构建文件；不会复制 `.git`、`.env`、SSH 文件、数据、研究产物、输出、缓存或虚拟环境。

```sh
python scripts/prepare_container_build.py /absolute/new-build-context
# 读取 new-build-context/build-identity.json 的 tree_sha256，显式填入下面变量。
CODE_IDENTITY=实际的tree_sha256
docker build --platform linux/amd64 \
  --build-arg CODE_IDENTITY="$CODE_IDENTITY" \
  -f /absolute/new-build-context/containers/Dockerfile \
  -t agentic-arxiv:local-verified /absolute/new-build-context
```

`build-identity.json` 保存实际文件清单及哈希、Git 提交与 dirty 状态。源码有未提交修改时 `dirty=true`，不能声称镜像只对应某个干净提交。Dockerfile 会检查构建身份参数及源码哈希，正常构建 wheel 并进行非编辑式安装，离开源码目录检查包位置与帮助，保存 `pip inspect`、完整锁和 wheel。镜像元数据位于 `/opt/agentic-build/`。

原始锁文件保持字节不变，安装使用其派生副本 `install-requirements.lock`，在末尾显式追加 PyTorch cu126 索引。原因是 pip 读取锁内的 `--index-url` 时会重置命令行额外索引，单靠 CLI 的 `--extra-index-url` 会漏掉 CUDA wheel。派生副本不改变任何包版本或分发哈希，与原锁一起保留在镜像中。

镜像还设置 `USER` 和 `LOGNAME` 环境变量，使 PyTorch 缓存初始化中的 `getpass.getuser()` 支持不在容器 `/etc/passwd` 中的宿主数字 UID。它们只是用户名提示，不改变实际进程 UID/GID，也不授予权限；运行身份仍由 `--user` 或 Compose 的 `user` 显式指定。

构建后需要核对安装检查、项目 wheel SHA256、完整 `pip inspect` 与版本、训练库所需接口、构建上下文清单、OCI 身份标签和实际镜像摘要。目标 profile 必须填写真正的 `仓库@sha256:摘要` 并在本地 Docker 中已存在；不能填写示例摘要或可变标签。仅有本地 image ID 时，不将它伪称为 registry RepoDigest。镜像发布、导入或拉取也是单独准备操作，管理器不代办。

非 GPU 入口与只读输入预检也应在获准构建后验证；这不等于真实单卡或四卡训练恢复通过。

## 管理员需独立确认的宿主条件

1. **驱动**：宿主 NVIDIA 驱动必须支持 CUDA 12.6；容器不安装内核驱动。管理员应核对 NVIDIA 官方兼容说明及服务器实际 `nvidia-smi`，历史记录不能证明现在可用。
2. **GPU 容器运行时**：安装和配置 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 是管理员单独授权步骤，不由管理器执行。安装软件、修改 runtime、重启 Docker 都可能影响其他任务，不能自动完成。
3. **Docker 权限**：[Docker 官方说明](https://docs.docker.com/engine/install/linux-postinstall/)指出 docker 组提供近似 root 的能力。是否授予、使用哪种 daemon 或 rootless 路径由管理员决定；管理器只报告权限问题，不执行提权、加组或 chmod socket。
4. **存储**：分别检查 Docker data-root、containerd image store、BuildKit 缓存和日志所在文件系统。仅把模型放到数据盘不会迁移镜像层；参见 [Docker daemon 配置说明](https://docs.docker.com/engine/daemon/)。配置存储迁移、日志轮转及服务重启需独立授权。管理器不清理其他镜像、容器或历史尝试。
5. **目录与设备**：提前创建归目标普通 UID/GID 所有的输出、控制、日志和缓存目录，核对空间；明确列出 GPU UUID 及 rank 顺序。不得挂载整个用户主目录、数据盘、可变源码或 Docker socket。输入及旧尝试只读，当前尝试和指定缓存可写。

## 单任务 Compose 边界

安装包中的 `resources/experiments/compose.json` 为单任务基线，渲染后的 JSON 本身是合法 Compose 配置。只使用已存在的摘要镜像，`pull_policy=never`、`restart=no`，不自动 build、不自动删除容器。普通 UID/GID、显式 GPU 设备、共享内存、只读根文件系统和输入、独立 output/control/logs/cache、`/tmp` tmpfs、禁用网络、全部 capability 移除及禁止提权均显式记录。

配置不包含特权模式、宿主 IPC/PID、Docker socket、管理端口或可执行 shell 字符串。可使用 `docker compose -f rendered-compose.json config --quiet` 验证最终渲染，仅做解析不会启动任务。运行记录保存实际使用的配置；Compose 不提供队列、自动换卡或全局 GPU 独占。外部用户仍可能在检查与启动之间争用设备。

## 回退与未验证项

默认保留原虚拟环境、训练入口、已有模型及研究产物，不覆盖或升级历史 manifest。只有旧资产实际仍存在且可验证时，才可以回退到原来的显式命令和虚拟环境；旧 checkpoint 不因此取得新受管恢复资格。不要以回退为名删除新失败证据。2026-10-16 用户已明确要求清空 `/data/ding`，该服务器旧资产未备份，不存在旧环境回退路径。

实际部署应分别记录镜像构建与正常包安装、容器内非 GPU/只读预检、Docker 权限及 GPU runtime 检查。非 GPU 部署通过不等于真实 SFT/DPO/GRPO 单卡及四卡恢复通过；真实训练、评估和输入准备仍需单独授权。
