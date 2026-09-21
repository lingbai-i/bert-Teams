"""验证 EDA 的数据解析、重复率分母和启发式主题归一化。

使用小型人工样本固定统计口径，避免真实数据中的重复掩盖指标计算错误。
仅依赖 src.eda 的纯函数，不加载模型或训练数据。
"""
import pytest

from src.eda import core_text, overlap_stats, read_rows, summarize


def test_reader_preserves_text_and_last_tab(tmp_path):
    """文本内制表符应保留，标签从最后一个制表符后读取。"""
    path = tmp_path / "sample.txt"
    path.write_text("  年假\t申请  \t0\nVPN\t1\n", encoding="utf-8")
    assert read_rows(path, 2) == [("  年假\t申请  ", 0), ("VPN", 1)]


@pytest.mark.parametrize("content", ["无标签\n", "问句\t9\n", "\t0\n", "问句\tx\n", "\n"])
def test_reader_rejects_invalid_rows(tmp_path, content):
    """格式异常必须带行号报错，不能静默丢弃。"""
    path = tmp_path / "bad.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="bad.txt:1"):
        read_rows(path, 9)


def test_overlap_uses_target_rows_and_reports_label_disagreement():
    """命中率按目标集行数计算，唯一文本和冲突行数分别报告。"""
    source = [("年假", 0), ("年假", 0), ("合同", 1)]
    target = [("年假", 0), ("年假", 0), ("合同", 2), ("新问句", 0)]
    result = overlap_stats(source, target)
    assert result == {"target_rows": 4, "matched_rows": 3,
                      "matched_unique_texts": 2, "matched_rate": 0.75,
                      "label_disagreement_rows": 1}


def test_summary_counts_missing_classes_and_conflicting_labels():
    """重复文本以文本判定，标签冲突独立计数，缺失类别保留零值。"""
    result = summarize([("年假", 0), ("年假", 1), ("VPN", 1)], 3)
    assert result["class_counts"] == [1, 2, 0]
    assert result["duplicate_excess_rows"] == 1
    assert result["conflicting_texts"] == 1
    assert result["length"]["max"] == 3


def test_core_removes_only_boundary_patterns():
    """边界装饰可重复剥离，保留语义关键词及正文内部字符。"""
    assert core_text("请问请问年假怎么申请谢谢啦？") == "年假"
    assert core_text("年假有效期要求?") == "年假有效期"
    assert core_text("不能请假怎么办") == "不能请假"
    assert core_text("关于合同的规则是什么") == "合同"
    assert core_text("请问") == "请问"
