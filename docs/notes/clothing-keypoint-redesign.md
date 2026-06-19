# 衣服关键点重设计（v4/v5 迭代记录）

承接 `clothing-keypoint-issues.md` 的领口检测 V1-V4 迭代。在 TPS
试衣实际效果上发现 8 点语义不够，触发本次重设计。分支
`feat/clothing-keypoint-redesign`，6 个 commit。

## 重设计目标

- **以人体关键点为主**（用户原话）：衣服侧 8 点里有 `*_bottom`
  这种"衣服图最底端"语义错位（旗袍到脚踝 vs 人体髋部），且
  `armpit→elbow` 是穿着后近似不准
- **加 hip 关键点**：用衣身最大宽度位置取代下摆，作为"衣服胯部"

## v4 失败（commit 0811330 + 后续）：TPS 控制点冲突

按用户决策把 CORRESPONDENCE 改成 `armpit→shoulder`（语义上腋下 ≈
肩部高度，最接近解剖对应），配合新的 `_find_max_width_y`（hip）
和 `_find_armpit_y`（width-jump 算法）。

**结果**：3 张图全部 warp 失败，衣服扭曲成身体右侧一小块。

**根因**：`armpit→shoulder` 让两个不同的衣服 src 映射到同一个人体
dst——`cloth_left_shoulder` 和 `cloth_left_armpit` 都映射到
`body_left_shoulder`。TPS 系数矩阵 L 中第 i 行依赖 `body_i`（通过
K[i, :] 和 P[i, :]），`body_2 = body_4` ⇒ `L[2, :] = L[4, :]`，L 出
现重复行 singular 或病态，`np.linalg.solve` 返回退化解。

**TPS 几何硬约束**：N 对控制点的 dst 必须全部唯一。即使 src 不同，
N 个不同的身体点不能用 N 个相同的 dst 坐标。

## v5 修复（commit 6ce373b）：恢复 armpit→elbow

把 CORRESPONDENCE 改回 v3 的 `armpit→elbow`。`elbow` 是人体离躯干
中线最远的可用点，且没被其他对应占用。语义上 elbow 只是"穿着后
近似"，但 TPS 几何稳定。

3 张图衣身都可见，但暴露了新问题。

## v5 暴露的新问题：hip 位置过低

`_find_max_width_y` 在旗袍上把 hip 放在 y=0.94 ch（接近裙摆底部）。
TPS 控制点 (105, 768) → (439, 737) 把衣身底部"拉回"人体胯部——
和 118f9e5 commit 想修掉的 `bottom→hip` 强对应几乎同样的效果。

旗袍长度从 v3 的"中长款到胯"维持，没有回到脚踝。

T 恤/衬衫：衣身可见但袖子仍有"翼"（`armpit→elbow` 的固有副作用，
v3 已知问题）。

## 后续待办

### 选项 1：改 hip 检测为按比例推算
- 用"领到腋下"作衣服上半身长度参考，hip = 领口下方
  (`衣服上半身长度`) × (`人体 (hip_y - shoulder_y) / (shoulder_y - neck_y)`)
- 短款 T 恤：hip ≈ 衣身下半部（接近 v5 现状）
- 长款旗袍：hip 在衣身中段（不再是最宽的裙摆）
- 工作量：1-2 小时

### 选项 2：把 hip 移出 CORRESPONDENCE
- 7 点 clothing schema 保留，hip 仍可检测和可视化
- TPS 控制点回退到 5 对（top + shoulders + armpits）
- 行为等价于 v3（旗袍中长款到胯、T 恤/衬衫袖子翼）
- 工作量：5 分钟

## 当前 v5 状态小结

| 维度 | 状态 |
|---|---|
| 衣服关键点检测算法 | ✓ 7 点 schema 落地，width-jump 和 max-width 两种检测均工作 |
| TPS 控制点冲突 | ✓ v5 修复 |
| 旗袍长款延伸 | ✗ hip 位置过低，未达脚踝 |
| T 恤/衬衫袖子翼 | ✗ `armpit→elbow` 固有副作用，v3 已知问题 |
| V 领/翻领 | ✗ V4 算法已知局限（独立问题） |

分块 warp（plan agent 方案 D）才是治本——分块后袖子受独立控制点约束，
不会牵动衣身。建议作为下一个 PR 起点。
