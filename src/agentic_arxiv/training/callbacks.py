"""可在基础环境检查的停止策略基类。

安装训练依赖时使用原 Transformers 回调；缺依赖时只允许纯策略检查，
实际训练仍在入口检查并导入训练库，绝不替换训练器或模型。
"""

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        """无训练依赖时供纯停止策略使用，不提供训练执行能力。"""
