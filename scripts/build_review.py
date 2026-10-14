"""Build a readable PDF from the editable revision, bibliography and pilot figures."""
import re,json,html
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from common import ROOT

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#233d4b')))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6,textColor=colors.HexColor('#536572')))
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=28,leading=33,spaceAfter=16,textColor=colors.HexColor('#123342')))
styles.add(ParagraphStyle(name='H1Custom',fontName='Helvetica-Bold',fontSize=18,leading=23,spaceBefore=16,spaceAfter=10,textColor=colors.HexColor('#123342'),keepWithNext=True))
styles.add(ParagraphStyle(name='H2Custom',fontName='Helvetica-Bold',fontSize=13,leading=17,spaceBefore=12,spaceAfter=8,textColor=colors.HexColor('#087e8b'),keepWithNext=True))
styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=7.7,leading=10,textColor=colors.HexColor('#233d4b')))

def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',lambda m:'<font face="Courier">'+m[1]+'</font>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'\*([^*]+)\*',r'<i>\1</i>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m: ('<link href="'+m[2]+'" color="#087e8b">'+m[1]+'</link>') if m[2].startswith('https:') else m[1],s)
    return s

def para(t,style='BodyCustom'):return Paragraph(inline(t),styles[style])

def markdown(text):
    blocks=[];lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',x) for x in cells):rows.append([para(c,'CellCustom') for c in cells])
                i+=1
            cols=len(rows[0]);widths=([127]+[(475-127)/(cols-1)]*(cols-1)) if cols>2 else [145,330]
            t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eff1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#087e8b'))]));blocks.extend([t,Spacer(1,10)]);continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));blocks.append(para(line.lstrip('# '),'H1Custom' if level==1 else 'H2Custom'));i+=1;continue
        group=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')):
            group.append(lines[i].strip());i+=1
        # Keep explicitly numbered lines distinct.
        if re.match(r'^\d+\.',line):
            blocks.extend(para(x) for x in group)
        else:blocks.append(para(' '.join(group)))
    return blocks

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d5e2e7'));canvas.line(55,46,540,46)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#667986'));canvas.drawString(55,32,'THESIS REVISION  |  23 SEPTEMBER 2026  |  PILOT EVIDENCE');canvas.drawRightString(540,32,str(doc.page));canvas.restoreState()

def main():
    story=[Spacer(1,30),para('Graph-aware\nentity typing'.replace('\n','<br/>'),'TitleCustom')]
    # Explicit title paragraph to preserve the line break without escaping.
    story[1]=Paragraph('Graph-aware<br/>entity typing',styles['TitleCustom'])
    story+=[para('Revision draft and pilot findings','H1Custom'),para('Sindi Teqja | Working revision | 23 September 2026','SmallCustom'),Spacer(1,15),para('A clearer account of what taxonomy context changes, why it changes, and what the current experiments can establish.'),Spacer(1,12)]
    for h,t in [
        ('The research story','Oracle context often recovers more annotated types, while retrieved context does not consistently reproduce that pattern. Early specificity, full-list coverage, and output length need separate interpretation.'),
        ('What is included','Editable thesis sections; separate MRR, nDCG, most-specific and gold-coverage plots; complete rank and recall heatmaps; compact prompts; a controlled-cohort generator; and verified references.'),
        ('Status of the evidence','All 200 principal-run outputs were checked against saved scores. New v4 prompts and the 100-entity cohort have not been run. This document revises the supplied exposé and experiment project; a full manuscript was not attached.'),
        ('Where to continue','The accompanying project contains the editable text in docs/thesis_revision.md, the detailed protocol, all 17 figures as SVG/PNG, analysis CSVs, and reproducible scripts.')]:
        story += [para(h,'H2Custom'),para(t)]
    story.append(PageBreak())
    text=(ROOT/'docs/thesis_revision.md').read_text()
    # The cover replaces the first title; references are embedded below.
    text=text[text.index('## 1. Introduction'):text.index('## References')]
    story+=markdown(text)
    story.append(PageBreak());story.append(para('References','H1Custom'));story.append(para('Primary-source metadata checked 23 September 2026. Numbers correspond to citations in the draft.','SmallCustom'))
    for r in json.loads((ROOT/'docs/references.json').read_text()):
        story.append(para(r['citation'],'SmallCustom'))
        link='https://doi.org/'+r['doi'] if r['doi'] else r['url']
        story.append(Paragraph('<link href="'+html.escape(link)+'" color="#087e8b">'+html.escape(link)+'</link>',styles['SmallCustom']))
    captions=[
        ('01_mrr','MRR weights the earliest deepest-gold match. The historical metric uses maximum annotated depth; it is not a general measure of all correct types.'),
        ('02_ndcg','The current nDCG uses depth/max-depth relevance and exponential gains [7]. The gain choice is project-specific. Scores are unchanged from the archived evaluation.'),
        ('03_most_specific','Hit-anywhere gives credit if any deepest gold type occurs in the output. It differs from Hit@1 and should be read with output length and fixed-k results.'),
        ('04_gold_coverage','Coverage is macro-averaged gold recall over entities. It includes generic gold labels. Error bars are exploratory entity-bootstrap intervals, not evidence of causal relation effects.'),
        ('05_context_source','Oracle and retrieved contexts represent different evidence regimes. Candidate visibility and context quantity also differ in the archived prompts.'),
        ('06_output_lengths','Each dot represents one entity; the diamond is the mean. A longer list creates more opportunities for gold recovery. Common-cutoff scores are supplied separately.')]
    for idx,(file,caption) in enumerate(captions):
        if idx%2==0:story.extend([PageBreak(),para('Pilot figures'+('' if idx==0 else ' (continued)'),'H1Custom')])
        img=ROOT/'figures'/f'{file}.png';w,h=ImageReader(str(img)).getSize();story.extend([Image(str(img),width=475,height=475*h/w),para(caption,'SmallCustom'),Spacer(1,12)])
    story.extend([PageBreak(),para('Reading the type diagnostics','H1Custom'),para('The project includes all 224 entity/gold-type pairs across six main conditions, spread across seven rank heatmaps, plus four type-recall heatmaps covering 110 gold classes. Full-resolution versions are in figures/.'),para('A cell marked M means that the gold type was not returned. There is no observable output rank for an absent type. Compare another condition to see whether the same type is recovered and at what rank. Depth labels describe annotation specificity; tied depths are not an authoritative total gold ranking.'),para('Type-recall heatmaps label the number of annotated entities supporting each class. A 100% cell with n=1 is a case observation, not evidence of reliable class-level generalization.'),para('Next step','H2Custom'),para('Freeze the revised design after development review, then run the nine conditions under one model/backend configuration. Add a matched-total-budget control before interpreting full-context gains as relation complementarity. Keep oracle and retrieved conclusions distinct. The detailed instructions and pending comparisons are in docs/experimental_protocol.md.')])
    SimpleDocTemplate(str(ROOT/'Thesis-Revision-Review.pdf'),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=55,bottomMargin=62,title='Graph-aware entity typing: revision draft and pilot findings',author='Sindi Teqja; revision assistance').build(story,onFirstPage=footer,onLaterPages=footer)
    print(ROOT/'Thesis-Revision-Review.pdf')
if __name__=='__main__':main()
