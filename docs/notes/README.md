# 过程笔记索引

> 按 CLAUDE.md "分类记录过程文档"原则，把调试记录、设计决策放在这里，
> 方便以后回顾"为什么这么改"。

| 文件 | 主题 |
|---|---|
| [clothing-keypoint-issues.md](clothing-keypoint-issues.md) | 3 张测试衣服图（蓝旗袍 / 浅粉 T 恤 / 白衬衫）的关键点问题诊断 |
| [v2-curvature-experiment.md](v2-curvature-experiment.md) | V2 曲率方法在三张测试图上全部劣于 V1 的失败诊断 |
| [v3-width-profile-experiment.md](v3-width-profile-experiment.md) | V3 宽度剖面法在 mask 含阴影时失败（T 恤/白衬衫），qipao 勉强可用的诊断 |
| [mask-postprocess-experiment.md](mask-postprocess-experiment.md) | 基础 mask 后处理（闭运算+最大 CC+填洞+腐蚀）对 V1 无影响、对 V3 几乎无效的诊断 |
