# SMC Camera-Ready Reviewer Change Log

日期：2026-07-17

格式权威：`D:\Documents\Bus Project\SMC\论文模板SMC.doc`（IEEE SMC，US Letter、0.75 in 页边距、双栏正文）。

主稿交付物：

- 带修订痕迹：`D:\Documents\Bus Project\SMC\Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-tracked.docx`
- 干净终稿：`D:\Documents\Bus Project\SMC\Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-final.docx`
- Word 原生 PDF：`D:\Documents\Bus Project\SMC\Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-final.pdf`

## 意见映射

| 来源 | 意见 | 主稿位置 | 处理与证据 | 状态 |
|---|---|---|---|---|
| AE / 总体 | 形成可核验、六页、IEEE 格式的 camera-ready 稿 | 全文；第 3–5 页图表 | 依据指定模板统一 Letter 版心、双栏正文、图题在下、表题在上；图表按首次解释分布于三页；Word 原生分页为 6 页 | 完成 |
| Reviewer 17512-1 | 文中称五类相关研究但只列四类 | Section II-E，第 2 页 | 新增 “Simulation Calibration and Bayesian Optimization”，并补入 BO 基础文献 [14], [15] | 完成 |
| Reviewer 17512-2 | 定义误差项 `e_i` | Section III-B，第 2 页 | 明确其为匹配下游站点的模拟累计到达时间均值减观测均值，并说明相对首个匹配站、最少三个下游站点 | 完成 |
| Reviewer 17512-3 | Figure 1 上方文本块未垂直对齐 | Section IV-A / Figure 1，第 2 页 | 精简并补足两栏周围正文，使图前两栏基本齐底；流程图图像未修改 | 完成 |
| Reviewer 17512-4 | 解释 GP 与 “new candidate” | Section IV-B，第 3 页 | 定义训练对、完整有界候选向量、预测均值/标准差、EI 与下一未评估候选的选择 | 完成 |
| Reviewer 17512-5 | 强化创新点 | Abstract、Section I、II-E、VII | 将贡献限定为可审计的语义/冻结协议、共同预算比较和跨日证据，不声称新 BO 或 smoother 算法 | 完成 |
| Reviewer 17513-1 | 增加定量比较并澄清新颖性 | Abstract、Sections VI–VII | 写入 Rule C、A0–A4、BO/LHS 的真实均值、标准差和反例，明确无统一最优配置 | 完成 |
| Reviewer 17513-2 | 比较阈值替代方案 | Section VI-A、Figure 3 | 加入 Rule C 3×3 敏感性、MAD、Isolation Forest、worst-window 与 IRN 小分母诊断 | 完成 |
| Reviewer 17513-3 | 改进引言与文献综述 | Sections I–II | 重组为问题、缺口、三项合同、证据链及五类相关工作 | 完成 |
| Reviewer 17513-4 | 统一符号与公式 | Sections III–IV，第 2–3 页 | 统一 `θ_bus`、`x_corr`、`minGap_bus`/`minGap_background`、Greek 符号及公式 (7) 的 analyzed/forecast 下标 | 完成 |
| Reviewer 17514-1 | 阐明创新并避免过强结论 | Abstract、Sections I、VI-E、VII | 报告不利的 development/worst-window 结果与 seed 3 LHS 反例，限定为两线路跨日协议证据 | 完成 |
| Reviewer 17514-2 | 精简冗长文字 | 全文 | 删除重复引言与旧结果叙述，采用结果先行且可审计的短段落 | 完成 |
| Reviewer 17514-3 | 改进 Figures 2–5 与 captions | 第 3–5 页 | Figures 2–5 使用本次冻结实验生成的 300-DPI PNG并按首次解释分散排版；扩写数据来源、样本量、阈值和解释边界 | 完成 |

## Figure 1 专项决定

`Decision on SMC 2026 submission 8` 中没有要求修改 Figure 1 流程图内容。Reviewer 17512 仅指出图上方两个文本块的垂直对齐。因此遵照用户指示：不替换、不编辑、不重新生成主稿内 Figure 1；最终 DOCX 中对应媒体 SHA-256 为 `6d65e98734cb94851b545f11d744c32535fd00efb6318a1de203aa28333e5e84`，与原始主稿一致。

## 验证摘要

- Word 审阅稿：原生比较生成，194 revisions；结构报告为 `ins=208, del=238, moveTo=12, moveFrom=12`。关闭表格结构比较后，接受全部修订可保持正确 6×7 表格。
- 干净终稿：接受全部修订后四类修订计数均为 0。
- PDF：Microsoft Word 原生导出，共 6 页，逐页检查无裁切、重叠或图题错位。
- 嵌入数据图：Figures 2–5 的字节 SHA-256 与本次生产 PNG 完全一致。Table I 直接读取冻结 `table_i.csv` 并生成原生、可编辑的 6×7 Word 表格；Table I PNG 未嵌入主稿。
- 原始主稿与模板未覆盖：原稿 SHA-256 `886e511c233f4002a039ba533df86f697d81703585f3d2bc032bb6b5f812560e`；模板 SHA-256 `9cbb2172d681593ea4b395a1c0deb3bfc3e91725eaf714ed01556126c83a8e29`。
