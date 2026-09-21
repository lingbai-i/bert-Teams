"""src 包：BERT 中文意图分类项目的全部核心模块。

模块划分：
- config   全局配置（路径、超参数默认值）
- data     数据加载与预处理
- model    模型构建与学生模型初始化
- train    基线训练入口
- evaluate 评估指标与 bad case 导出
- distill  知识蒸馏
- prune    剪枝与性能测量
- predict  推理封装（供部署层调用）
"""
