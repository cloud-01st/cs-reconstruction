# -*- coding: utf-8 -*-
"""演示系统自动化冒烟测试（Streamlit AppTest，无需浏览器）

模拟真实用户操作路径：
  1. 默认加载（示例图 cameraman + 25% + 方法对照）
  2. 切换采样率 → 10% / 50%
  3. 切换"三档全对比"
  4. 关闭传统方法对照
每步校验：脚本执行无异常、关键元素齐全。

用法：python scripts/app_smoke_test.py
（可作为论文 5.4「演示系统测试」的自动化测试证据）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app.py")
PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "✓" if cond else "✗"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))


def run_case(label: str, mutator=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    if mutator is not None:
        mutator(at)
        at.run()
    print(f"\n【{label}】异常数: {len(at.exception)}")
    for e in at.exception:
        print(f"    {e.value}")
    return at


def main() -> int:
    print("=" * 66)
    print("演示系统冒烟测试（AppTest）")
    print("=" * 66)

    # ---- 场景 1：默认加载（示例图 + 25% + 对照） ----
    at = run_case("场景1 默认加载")
    check("无脚本异常", not at.exception)
    check("页面标题正确", any("演示系统" in t.value for t in at.title))
    check("侧边栏含采样率选项",
          any("采样率" in str(r.label) for r in at.sidebar.radio))

    # ---- 场景 2：切换采样率 ----
    at2 = run_case("场景2 切换采样率=10%",
                   lambda a: a.sidebar.radio[0].set_value("10%"))
    check("无脚本异常", not at2.exception)

    # ---- 场景 3：三档全对比 ----
    at3 = run_case("场景3 三档全对比",
                   lambda a: a.sidebar.radio[0].set_value("三档全对比"))
    check("无脚本异常", not at3.exception)
    subs = [s.value for s in at3.subheader]
    check("三个采样率小节齐全",
          all(any(f"{p}%" in s for s in subs) for p in (10, 25, 50)),
          " | ".join(subs[:4]))

    # ---- 场景 4：关闭传统方法对照 ----
    at4 = run_case("场景4 关闭对照（仅本方法）",
                   lambda a: a.sidebar.checkbox[0].set_value(False))
    check("无脚本异常", not at4.exception)

    # ---- 场景 5：模型对比曲线（折线图） ----
    at5 = run_case("场景5 模型对比曲线")
    check("无脚本异常", not at5.exception)
    try:
        n_charts = len(at5.get("plotly_chart"))
    except Exception:  # noqa: BLE001  AppTest 版本过低时不支持该元素类型
        n_charts = -1
    check("折线图已渲染（≥6 张）", n_charts >= 6, f"plotly_chart 元素 {n_charts} 个")
    check("含消融/训练等分页标签", len(at5.tabs) >= 5, f"tabs={len(at5.tabs)}")

    # ---- 场景 6：勾选"标注数值"后仍正常 ----
    def _mark_values(a):
        for cb in a.checkbox:
            if "标注数值" in str(cb.label):
                cb.set_value(True)
                return

    at6 = run_case("场景6 开启数值标注", _mark_values)
    check("无脚本异常", not at6.exception)
    try:
        n6 = len(at6.get("plotly_chart"))
    except Exception:  # noqa: BLE001
        n6 = -1
    check("折线图仍正常渲染", n6 >= 6, f"plotly_chart 元素 {n6} 个")

    print("\n" + "=" * 66)
    print(f"结果：通过 {PASS} 项，失败 {FAIL} 项")
    print("=" * 66)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
