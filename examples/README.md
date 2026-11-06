# 数据与参数示例

这些文件展示格式，不包含真实角色音源、模型权重或训练结果。

- `design_spec.example.json`：可传给 `rvcre design --spec` 的手动声线设计要求。
- `performance.example.json`：API 中的表达参数结构，CLI 使用对应命令行选项。
- `source_note.example.json`：素材来源记录示意，供项目整理使用；不是必须通过的授权检查，也不是自动导入接口。
- `train.example.jsonl` / `validation.example.jsonl`：VoxCPM 清单格式，路径是占位符。请用 `rvcre dataset prepare` 对自己的材料生成真实清单。
- `evaluation_texts.json`：人工编写的固定试听文本，不是训练语料包。

训练和验证示例中的参考只在各自划分内引用，不能因为文件格式合法就把不存在的路径当成数据准备完成。来源整理、音频处理及训练命令见 [数据指南](../docs/DATASET_TRAINING.md)。
