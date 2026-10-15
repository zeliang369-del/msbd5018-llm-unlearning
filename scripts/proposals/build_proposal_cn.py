from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/MSBD5018_Group_Project_Proposal_中文.pdf'
pdfmetrics.registerFont(TTFont('CN', '/System/Library/Fonts/STHeiti Light.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNBold', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNBold', italic='CN', boldItalic='CNBold')
body = ParagraphStyle('Body', fontName='CN', fontSize=10, leading=14, spaceAfter=5, wordWrap='CJK')
heading = ParagraphStyle('Heading', parent=body, fontName='CNBold', spaceBefore=5, spaceAfter=3)
title = ParagraphStyle('Title', parent=heading, fontSize=16, leading=21, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle('Subtitle', parent=body, fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=8)
ref = ParagraphStyle('Reference', parent=body, fontSize=8, leading=10.5, spaceAfter=3)
story = []
def p(text, style=body):
    story.append(Paragraph(text, style))

p('大语言模型遗忘：效果、能力保留与鲁棒性', title)
p('MSBD 5018 小组项目提案 | 2026 年秋季 | 中文对照版', sub)
p('任务与意义', heading)
p('删除训练资料的源文件后，语言模型仍可能保留相关信息。选择性遗忘旨在减少指定资料的影响，同时保留有用知识，可用于数据撤回和模型维护。基于 TOFU [1] 与 OpenUnlearning [2]，我们拟轻量复现并实证分析遗忘效果、能力保留及提示变化鲁棒性之间的权衡。实验使用虚构作者问答作为可控代理任务，不声称已验证真实隐私或版权资料的彻底删除。')
p('研究问题与方法', heading)
p('我们关注：（1）固定资源预算下，哪些方法具有更好的遗忘与能力保留平衡？（2）保留作者表现更好，是否意味着整体知识能力保留更好？（3）表面遗忘后，改变提示方式能否再次套出相关事实？实验从已学习 4,000 条问答的 TOFU 微调模型 Llama-3.2-1B-Instruct 出发，forget01 指定 40 条遗忘，其余 3,960 条构成保留池。未遗忘模型是基线，官方仅保留资料模型的评价记录作为明确标注的导入参照。')
p('核心比较拟包含梯度上升 GA、梯度差分 GD、负偏好优化 NPO 和无需参照模型的 SimNPO。GA/GD 已完成；NPO/SimNPO 将先做资源检查。表征误导方法 RMU 作为可选第五种方法，其参照模型、中间激活及参数选择方式需额外适配，并将单独报告可训练参数预算。项目比较已有目标函数，不宣称提出新遗忘算法。')
p('在方法兼容时，固定模型起点、数据划分、处理文本、样本抽取、学习率、有效批大小及更新预算，记录各方法系数和参数量。16 GB Mac 已通过 CPU/磁盘承载 FP32 优化器状态和依次反向传播完成 GA/GD 全参数训练。后续不会用 LoRA 结果替代这些基线；若有减少可训练参数的实验，将另列比较。Windows/CUDA 尚未验证。')
p('评价与初步可行性', heading)
p('核心评价使用全部 657 题：40 道遗忘题、400 道保留作者题、100 道真实作者题及 117 道世界知识题。报告遗忘质量 FQ（与仅保留资料参照作 KS 检验的 p 值）、模型效用 MU（九项非遗忘分数的调和平均值）、回答案例及资源消耗。FQ 不是准确率，也不证明不可逆遗忘。以下为已完成的种子 0 结果；每种训练方法均执行 10 次参数更新：')
table=Table([['模型','遗忘质量 FQ','模型效用 MU'],['未遗忘模型','0.0143','0.5993'],['GA','0.4046','0.4019'],['GD','0.7659','0.3887']],colWidths=[224,122,122],hAlign='LEFT')
table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'CN'),('FONTNAME',(0,0),(-1,0),'CNBold'),('FONTSIZE',(0,0),(-1,-1),9.5),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('ALIGN',(1,0),(-1,-1),'RIGHT'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
story.extend([table,Spacer(1,5)])
p('两种方法 FQ 均提高，但 GD 总体 MU 未优于 GA，与已发表对照不同，说明仅部分复现，也促使我们分析能力分量。计划测试 GD 保留权重 {0.5, 1, 2}，对核心 GA/GD 补充种子 {0, 1, 2}，并根据实际设置绘制遗忘与效用关系。另用直接、上下文、角色扮演、指令挑战四类提示，对相同遗忘问题及对应保留对照做鲁棒性测试。答案重合度仅用于筛查，是否恢复事实须人工复核。以上扩展尚未完成，不作为已有结果。')
p('交付与协作', heading)
p('四位组员共同承担文献与数据、方法与训练、评价与鲁棒性、分析与展示，交叉审阅。计划 11 月 17 日前提交主要结果，12 月 9 日前提交 ACL 格式报告、代码与数据。交付固定配置、原始回答、权衡图及可复现的对照展示。')

p('参考文献', heading)
p('[1] Maini 等（2024）. TOFU: A Task of Fictitious Unlearning for LLMs. <link href="https://arxiv.org/abs/2401.06121" color="black">arXiv:2401.06121</link>.', ref)
p('[2] Dorna 等（2025）. OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics. <link href="https://arxiv.org/abs/2506.12618" color="black">arXiv:2506.12618</link>.', ref)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=43, leftMargin=43,
                        topMargin=31, bottomMargin=29,
                        title='大语言模型遗忘：效果、能力保留与鲁棒性 - 中文项目提案',
                        author='MSBD 5018 Project Team')
doc.build(story)
print(OUT)
