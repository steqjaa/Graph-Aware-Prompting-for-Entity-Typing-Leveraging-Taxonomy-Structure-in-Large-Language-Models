from pathlib import Path
import fitz,re,io,html
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'output/ThesisDraft_reviewed.pdf';d=fitz.open(p)
heads=[]
for i,page in enumerate(d):
 if i<3:continue
 for b in page.get_text('dict')['blocks']:
  if 'lines' not in b:continue
  spans=[s for l in b['lines'] for s in l['spans']]
  if spans and max(s['size'] for s in spans)>18:
   title=' '.join(s['text'] for s in spans); heads.append([1,title,i+1])
assert len(heads)==14,(len(heads),heads)
for n,f in [('S','DejaVuSans.ttf'),('B','DejaVuSans-Bold.ttf'),('T','DejaVuSerif.ttf')]:pdfmetrics.registerFont(TTFont(n,'/usr/share/fonts/truetype/dejavu/'+f))
buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=A4);c.setTitle('Contents');c.setFont('B',19);c.drawString(62,786,'Contents and reading conventions')
y=745
for _,t,pg in heads:
 c.setFont('S',9.5);c.drawString(62,y,t);c.drawRightString(530,y,str(pg));y-=23
style=ParagraphStyle('p',fontName='T',fontSize=10,leading=14.2)
for t in ['“Direct” is the manuscript term for conditions named standard in the experiment files. “Graph” denotes the local WiKC taxonomy and associated graph instructions. Llama 0, Llama 42 and Llama 123 identify the three fixed-seed runs of the same checkpoint.', 'Scores are means of entity-level metrics on a 0–1 scale unless stated otherwise. Main effect tables use matched completed entities. Full-condition tables use each condition’s completed cohort; their differences can therefore differ from paired effects when failures differ.', 'Tables print four decimals; supplementary CSV files retain full precision. All empirical results in this revision are recomputed from the four newly supplied result archives. PDF bookmarks provide chapter navigation.']:
 para=Paragraph(html.escape(t),style);w,h=para.wrap(470,300);y-=h+10;para.drawOn(c,62,y)
c.setFont('S',8);c.setFillColorRGB(.32,.4,.46);c.drawString(62,34,'Graph-Aware Prompting for Entity Typing');c.drawRightString(533,34,'3');c.save()
toc=fitz.open(stream=buf.getvalue(),filetype='pdf');d.delete_page(2);d.insert_pdf(toc,start_at=2)
ytop=841.89-745
for _,t,pg in heads:
 d[2].insert_link({'kind':fitz.LINK_GOTO,'from':fitz.Rect(60,ytop-12,535,ytop+7),'page':pg-1});ytop+=23
# Add section bookmarks from bold subheadings.
outline=[]
for level,t,pg in heads:
 outline.append([1,t,pg]);start=pg-1;nextpg=next((x[2]-1 for x in heads if x[2]>pg),len(d))
 for i in range(start,nextpg):
  for b in d[i].get_text('dict')['blocks']:
   if 'lines' not in b:continue
   spans=[s for l in b['lines'] for s in l['spans']]
   if spans and 11.9<=max(s['size'] for s in spans)<=12.1:
    tt=' '.join(s['text'] for s in spans)
    if re.match(r'^(?:\d+\.\d+|[BC]\.\d+) ',tt):outline.append([2,tt,i+1])
d.set_toc(outline);tmp=p.with_suffix('.tmp.pdf');d.save(tmp,garbage=4,deflate=True);d.close();tmp.replace(p)
print('Pages',len(fitz.open(p)));print(heads)
