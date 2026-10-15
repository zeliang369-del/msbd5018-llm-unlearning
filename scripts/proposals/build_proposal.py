from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

for name, filename in [
    ('Times-Roman', 'Times New Roman.ttf'),
    ('Times-Bold', 'Times New Roman Bold.ttf'),
    ('Times-Italic', 'Times New Roman Italic.ttf'),
    ('Times-BoldItalic', 'Times New Roman Bold Italic.ttf'),
]:
    pdfmetrics.registerFont(TTFont(name, '/System/Library/Fonts/Supplemental/' + filename))
pdfmetrics.registerFontFamily('Times-Roman', normal='Times-Roman', bold='Times-Bold',
                             italic='Times-Italic', boldItalic='Times-BoldItalic')

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/MSBD5018_Group_Project_Proposal.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)

body = ParagraphStyle('Body', fontName='Times-Roman', fontSize=10.5, leading=12.4, spaceAfter=5)
heading = ParagraphStyle('Heading', parent=body, fontName='Times-Bold', spaceBefore=5, spaceAfter=3)
title = ParagraphStyle('Title', parent=body, fontName='Times-Bold', fontSize=16, leading=18, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle('Subtitle', parent=body, fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=9)
ref = ParagraphStyle('Reference', parent=body, fontSize=8.5, leading=10, spaceAfter=3)
story = []
def p(text, style=body):
    story.append(Paragraph(text, style))

p('LLM Unlearning: Forgetting, Utility, and Robustness', title)
p('MSBD 5018 Group Project Proposal | Fall 2026', sub)
p('Task and motivation', heading)
p('Language models may retain training information after its source files are deleted. Selective unlearning aims to reduce the influence of designated records while preserving useful knowledge, with applications to data withdrawal and model maintenance. Based on TOFU [1] and OpenUnlearning [2], we propose a lightweight empirical study of the trade-off between forgetting, retained capability, and robustness to changed prompts. We use fictional-author question answering as a controlled proxy, rather than claiming verified deletion of real private or copyrighted material.')
p('Research questions and methods', heading)
p('We ask: (1) Which methods offer a better forgetting-utility balance under a fixed resource budget? (2) Does higher retained-author performance imply better preservation of broader knowledge? (3) Can information still be elicited by changing the prompt after apparent forgetting? We start from the TOFU-finetuned Llama-3.2-1B-Instruct checkpoint, which learned 4,000 question-answer pairs. The forget01 split removes 40 pairs; the remaining 3,960 form the retention pool. The unchanged checkpoint is our baseline, and the published retain-only evaluation is an explicitly imported reference.')
p('Our core comparison targets Gradient Ascent (GA), Gradient Difference (GD), Negative Preference Optimization (NPO), and reference-free SimNPO. GA and GD are completed baselines; NPO and SimNPO will first undergo resource checks. Representation Misdirection for Unlearning (RMU) is an optional fifth method because its reference model, intermediate activations, and parameter-selection scheme require additional adaptation. We will report its trainable-parameter budget separately. We compare published objectives rather than propose a new unlearning algorithm.')
p('We fix the initial checkpoint, split, processed text, data draws, learning rate, effective batch size, and update budget where compatible, while documenting method-specific coefficients and parameter counts. Native full-parameter GA/GD already run on a 16 GB Mac using CPU/disk-backed FP32 optimizer states and sequential backward passes. Future methods will not silently replace these baselines with LoRA; any reduced-parameter experiment will be reported separately. Windows/CUDA execution remains unvalidated.')
p('Evaluation and preliminary feasibility', heading)
p('Core evaluation uses all 657 questions: 40 forget, 400 retained-author, 100 real-author, and 117 world-fact questions. We report Forget Quality (FQ), a KS-test p-value against the retain-only reference, and Model Utility (MU), the harmonic mean of nine non-forget scores, plus answer-level cases and resource use. FQ is not accuracy or proof of irreversible erasure. Completed seed-0 measurements, each trained method using 10 optimizer updates, are:')
table = Table([['Model', 'Forget Quality', 'Model Utility'], ['Unchanged checkpoint', '0.0143', '0.5993'], ['GA', '0.4046', '0.4019'], ['GD', '0.7659', '0.3887']], colWidths=[224,122,122], hAlign='LEFT')
table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTNAME',(0,1),(-1,-1),'Times-Roman'),('FONTSIZE',(0,0),(-1,-1),10),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('ALIGN',(1,0),(-1,-1),'RIGHT'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
story.extend([table, Spacer(1,5)])
p('Both methods increase FQ, but GD does not outperform GA in overall MU, unlike the published comparison. This is a partial reproduction and motivates our component-level analysis. Planned extensions include GD retention weights {0.5, 1, 2}, seeds {0, 1, 2} for the core GA/GD comparison, and forgetting-utility plots across recorded settings. A separate prompt-robustness test will use direct, contextual, role-play, and instruction-challenge prompts on the same forget questions and matched retention controls. Answer overlap is only a screening signal; factual recovery will require manual review. These extensions are not completed results.')
p('Deliverables and teamwork', heading)
p('Four members will share literature/data preparation, methods and training, evaluation/robustness, and analysis/presentation, with joint review. We plan main results by November 17 and the ACL-format report, code, and data by December 9. Deliverables include pinned configurations, raw outputs, trade-off plots, and a reproducible comparison viewer.')

p('References', heading)
p('[1] Maini et al. (2024). <i>TOFU: A Task of Fictitious Unlearning for LLMs.</i> <link href="https://arxiv.org/abs/2401.06121" color="black">arXiv:2401.06121</link>.', ref)
p('[2] Dorna et al. (2025). <i>OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics.</i> <link href="https://arxiv.org/abs/2506.12618" color="black">arXiv:2506.12618</link>.', ref)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=43, leftMargin=43,
                        topMargin=33, bottomMargin=32,
                        title='LLM Unlearning: Forgetting, Utility, and Robustness',
                        author='MSBD 5018 Project Team',
                        subject='One-page group project proposal, Fall 2026')
doc.build(story)
print(OUT)
