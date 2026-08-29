"""M3：spec §4.3 分类矩阵 + 平台查询失败的保守路径。

用例清单（实现时逐一补齐，TDD 先行）：
- 平台无记录 → ENQUEUE / NEW_MODEL
- 平台有记录但 GPU 不同 → ENQUEUE / NEW_ADAPTATION
- 同模型同 GPU 已适配 → SKIP_DUPLICATE
- 本地已有活跃记录 → SKIP_DUPLICATE（不查平台）
- 悬赏候选 → ENQUEUE / BOUNTY
- search_adaptations 抛异常 → SKIP_UNCERTAIN（宁漏勿重）
"""
import pytest


@pytest.mark.skip(reason="M3 未实现")
def test_classification_matrix():
    ...
