import json,pandas as pd,re,string
records=json.load(open('output/qualitative_outputs.json'));records=[r for r in records if r['entity'] in [1,2,5]]
def norm(s):return re.sub(r'\s+',' ',s.lower().replace('_',' ')).strip().strip(string.punctuation+' ').strip()
sources={1:'https://www.bandainamcostudios.com/en/creator-interviews/2827',2:'https://www.sony.co.uk/electronics/support/a-mount-body-dslra200-series/dslr-a230/specifications',5:'https://www.starship-ent.com/musician/ive'}
supported={
1:set('agent|employee|entertainment industry worker|game developer|game development professional|game industry professional|game producer|human|individual|industry professional|living thing|media professional or artist|national|physical body|video game industry person|video game industry person or organization|video game industry professional|video game professional|japanese producer|body|entertainment professional|game creator|media creator|media producer|media worker|person or human|thing|japanese video game producer|creative professional'.split('|')),
2:set('artifact|artefact|camera model|consumer electronics|device|digital camera model|digital single-lens reflex camera|electronic device|imaging device|photographic equipment|physical object|product model|camera body|electronic appliance|equipment|photographic camera|thing|consumer electric product|consumer good|digital device|electrical device|electronically controlled device|good|goods|man-made object|tangible object|consumer product|digital appliance|item|material entity'.split('|')),
5:set('album|audio work|music release|musical release|sound recording|collection|content|item|media content|music single album|music|product|release|thing|artistic work|creation|cultural work|information artifact|intangible|intangible object|media item|musical album|publication|published work|artefact|artwork|audio|creative output|cultural artefact|cultural artifact|information entity|informational content|music album|music product|music recording|published works'.split('|'))}
contradicted={1:set('business|organization|electronic game|activity|game|media content|media product|media product or content|video game|entertainment'.split('|')),2:set(),5:set('corporate body|person or organization'.split('|'))}
malformed={1:set('japanese game leaser is a possible matching, but video game producer has classification in its name which is valid|recreational activity, indoor games are a sub class of this might be too broad so video game is a good description|ічнихeconomic entity'.split('|')),2:set(),5:{'music widget','2022 second single album by ive','subcategory of music'}}
rows=[]
for r in records:
 i=r['entity'];G=set(map(norm,r['gold']))
 for label in set(map(norm,r['predictions'])):
  if label in G:cat='gold match (not externally certified)';basis='Exact benchmark label; no automatic factual endorsement.'
  elif label in supported[i]:cat='supported addition';basis='Source identifies the entity kind or role; broad/general labels are reviewer inferences. Product/model metonymy accepted for camera typing.'
  elif label in contradicted[i]:cat='contradicted category';basis='Source identifies a human producer or a music release, not the incompatible entity kind asserted.'
  elif label in malformed[i]:cat='malformed or non-type';basis='String is a description, meta-commentary, corruption, or unusable type expression.'
  else:cat='unresolved';basis='Ambiguity or specificity not settled by the cited evidence; not counted as false.'
  rows.append({'model':r['model'],'condition':r['condition'],'entity':i,'name':r['entity_label'],'label':label,'judgment':cat,'basis':basis,'evidence_url':sources[i]})
df=pd.DataFrame(rows);df.to_csv('output/qualitative_claim_review.csv',index=False)
summary=df.groupby(['model','condition','judgment']).size().unstack(fill_value=0);summary.to_csv('output/qualitative_review_summary.csv')
print(df.groupby(['model','judgment']).size().unstack(fill_value=0).to_string())
print('UNRESOLVED',df[df.judgment=='unresolved'][['entity','label']].drop_duplicates().to_string(index=False))
print('supported',df[df.judgment=='supported addition'][['entity','label']].drop_duplicates().shape[0],'total',len(df),'unique',df[['entity','label']].drop_duplicates().shape[0])
