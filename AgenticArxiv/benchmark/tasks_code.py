"""代码生成域参数化任务集 (轨道 A): 12 模板 x 3 参数组 = 36 任务。

每任务: 函数签名 + 6-8 个参数化单元测试(check lambda 源) + 干扰项。
确定性: 测试由模板参数生成, 无随机; run_tests 子进程隔离执行。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_TMPL: List[Dict[str, Any]] = [
    dict(id="cd_add_signed", sig="def f(a, b):", desc="返回 a+b",
         tests="check('t_pos', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,B)!=A+B else None)\n"
               "check('t_neg', lambda: (_ for _ in ()).throw(AssertionError()) if f(-A,-B)!=-(A+B) else None)\n"
               "check('t_mix', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,-B)!=A-B else None)\n"
               "check('t_zero', lambda: (_ for _ in ()).throw(AssertionError()) if f(0,B)!=B else None)"),
    dict(id="cd_mul_shift", sig="def f(a, b):", desc="返回 a*b",
         tests="check('t_basic', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,B)!=A*B else None)\n"
               "check('t_one', lambda: (_ for _ in ()).throw(AssertionError()) if f(1,B)!=B else None)\n"
               "check('t_neg', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,-B)!=-A*B else None)\n"
               "check('t_zero', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,0)!=0 else None)"),
    dict(id="cd_maxmin", sig="def f(a, b):", desc="返回较大者; 相等返回 a",
         tests="check('t_a', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,B)!=max(A,B) else None)\n"
               "check('t_b', lambda: (_ for _ in ()).throw(AssertionError()) if f(B,A)!=max(A,B) else None)\n"
               "check('t_eq', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,A)!=A else None)"),
    dict(id="cd_count_vowel", sig="def f(s):", desc="返回 s 中元音字母个数(小写化后)",
         tests="check('t_lower', lambda: (_ for _ in ()).throw(AssertionError()) if f(WORD)!=sum(1 for c in WORD.lower() if c in 'aeiou') else None)\n"
               "check('t_upper', lambda: (_ for _ in ()).throw(AssertionError()) if f(WORD.upper())!=f(WORD) else None)\n"
               "check('t_none', lambda: (_ for _ in ()).throw(AssertionError()) if f('xyz')!=0 else None)"),
    dict(id="cd_rev_words", sig="def f(s):", desc="返回单词顺序反转的句子(保词内顺序)",
         tests="check('t_n', lambda: (_ for _ in ()).throw(AssertionError()) if f(SENT)!=' '.join(reversed(SENT.split())) else None)\n"
               "check('t_one', lambda: (_ for _ in ()).throw(AssertionError()) if f(WORD)!=WORD else None)"),
    dict(id="cd_sum_dig", sig="def f(n):", desc="返回 n 的十进制各位数字和(取绝对值)",
         tests="check('t_pos', lambda: (_ for _ in ()).throw(AssertionError()) if f(N)!=sum(int(d) for d in str(abs(N))) else None)\n"
               "check('t_neg', lambda: (_ for _ in ()).throw(AssertionError()) if f(-N)!=f(N) else None)\n"
               "check('t_zero', lambda: (_ for _ in ()).throw(AssertionError()) if f(0)!=0 else None)"),
    dict(id="cd_filter_thresh", sig="def f(xs, t):", desc="返回 xs 中严格大于 t 的元素列表(保序)",
         tests="check('t_basic', lambda: (_ for _ in ()).throw(AssertionError()) if f(LST,T)!=[x for x in LST if x>T] else None)\n"
               "check('t_all', lambda: (_ for _ in ()).throw(AssertionError()) if f(LST,-999)!=LST else None)\n"
               "check('t_none', lambda: (_ for _ in ()).throw(AssertionError()) if f(LST,999)!=[] else None)"),
    dict(id="cd_pair_sum", sig="def f(xs, t):", desc="返回 xs 中和恰为 t 的无序对数量(不重复计)",
         tests="check('t_cnt', lambda: (_ for _ in ()).throw(AssertionError()) if f(LST,T)!=sum(1 for i in range(len(LST)) for j in range(i+1,len(LST)) if LST[i]+LST[j]==T) else None)\n"
               "check('t_none', lambda: (_ for _ in ()).throw(AssertionError()) if f(LST,-999)!=0 else None)"),
    dict(id="cd_norm2", sig="def f(xs):", desc="返回向量 xs 的欧氏范数(浮点)",
         tests="check('t_val', lambda: (_ for _ in ()).throw(AssertionError()) if abs(f(LST)-sum(x*x for x in LST)**0.5)>1e-9 else None)\n"
               "check('t_zero', lambda: (_ for _ in ()).throw(AssertionError()) if f([0]*3)!=0.0 else None)"),
    dict(id="cd_anagram", sig="def f(a, b):", desc="两字符串是否互为字母重排(大小写不敏感)",
         tests="check('t_true', lambda: (_ for _ in ()).throw(AssertionError()) if f(WORD,SHUF)!=True else None)\n"
               "check('t_false', lambda: (_ for _ in ()).throw(AssertionError()) if f(WORD,OTHER)!=False else None)"),
    dict(id="cd_run_len", sig="def f(s):", desc="返回行程编码 [(字符,连续次数),...]",
         tests="check('t_enc', lambda: (_ for _ in ()).throw(AssertionError()) if f(STR)!= [(k, len(list(g))) for k,g in __import__('itertools').groupby(STR)] else None)\n"
               "check('t_single', lambda: (_ for _ in ()).throw(AssertionError()) if f(WORD)!=[(c,1) for c in WORD] else None)"),
    dict(id="cd_median3", sig="def f(a, b, c):", desc="返回三数中位数",
         tests="check('t_m', lambda: (_ for _ in ()).throw(AssertionError()) if f(A,B,C)!=sorted([A,B,C])[1] else None)\n"
               "check('t_perm', lambda: (_ for _ in ()).throw(AssertionError()) if f(C,B,A)!=f(A,B,C) else None)"),
]
_PARAMS = [  # (A, B, C, N) 数值参数组; 字符串参数由 WORD/SENT/SHUF/OTHER/STR/LST/T 派生
    dict(A=7, B=3, C=5, N=4321, WORD="banana", SENT="the quick brown fox", SHUF="nabana", OTHER="cherry",
         STR="aaabbbc", LST=[1,5,9,2,7], T=8),
    dict(A=12, B=4, C=9, N=908, WORD="elevator", SENT="machine learning builds systems", SHUF="televator", OTHER="rocket",
         STR="xxxyy", LST=[3,3,6,1], T=6),
    dict(A=25, B=8, C=17, N=7777, WORD="question", SENT="a b c d e f", SHUF="netouiqs", OTHER="answer",
         STR="abbcccdddd", LST=[10,-2,4,4,0], T=2),
]

_POOL: Dict[str, Dict[str, Any]] = {}
for _t in _TMPL:
    for _pi, _p in enumerate(_PARAMS):
        import re as _re
        # 词边界替换: 参数名作为完整标识符替换("A"不再吃AssertionError, "N"不吃None)
        _checks = _t["tests"]
        for _k, _v in _p.items():
            _checks = _re.sub(rf"\b{_re.escape(_k)}\b", repr(_v), _checks)
        _POOL[f"{_t['id']}_p{_pi}"] = dict(
            id=f"{_t['id']}_p{_pi}", sig=_t["sig"], desc=_t["desc"],
            checks_src=_checks, category="code", max_iterations=4,
        )


def get_code_task(task_id: str) -> Optional[Dict[str, Any]]:
    return _POOL.get(task_id)


def get_code_tasks() -> List[Dict[str, Any]]:
    return list(_POOL.values())
