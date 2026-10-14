import json
from common import ROOT
# id, key, type, title, authors in BibTeX order, year, venue, url, doi
refs=[
(1,'peng2024wikc','inproceedings','Refining Wikidata Taxonomy using Large Language Models','Peng, Yiwen and Bonald, Thomas and Alam, Mehwish',2024,'Proceedings of the 33rd ACM International Conference on Information and Knowledge Management','https://arxiv.org/abs/2409.04056','10.1145/3627673.3679156'),
(2,'shen2025taxoinstruct','inproceedings','A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion','Shen, Yanzhen and Zhang, Yu and Zhang, Yunyi and Han, Jiawei',2025,'Findings of the Association for Computational Linguistics: ACL 2025','https://aclanthology.org/2025.findings-acl.167/','10.18653/v1/2025.findings-acl.167'),
(3,'xia2024attempt','misc','Find Parent then Label Children: A Two-stage Taxonomy Completion Method with Pre-trained Language Model','Xia, Fei and Weng, Yixuan and He, Shizhu and Liu, Kang and Zhao, Jun',2024,'arXiv:2406.17739','https://arxiv.org/abs/2406.17739','10.48550/arXiv.2406.17739'),
(4,'ma2026drfet','inproceedings','Leveraging Label Semantics and Entity Description Generation for LLM-based Fine-grained Entity Typing','Ma, Yingying and Dai, Hongliang and Zhang, Xia and Li, Piji',2026,'Findings of the Association for Computational Linguistics: ACL 2026','https://aclanthology.org/2026.findings-acl.1584/','10.18653/v1/2026.findings-acl.1584'),
(5,'wei2022cot','inproceedings','Chain-of-Thought Prompting Elicits Reasoning in Large Language Models','Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Ichter, Brian and Xia, Fei and Chi, Ed and Le, Quoc and Zhou, Denny',2022,'Advances in Neural Information Processing Systems 35','https://arxiv.org/abs/2201.11903',''),
(6,'chia2023contrastive','misc','Contrastive Chain-of-Thought Prompting','Chia, Yew Ken and Chen, Guizhen and Luu, Anh Tuan and Poria, Soujanya and Bing, Lidong',2023,'arXiv:2311.09277','https://arxiv.org/abs/2311.09277','10.48550/arXiv.2311.09277'),
(7,'jarvelin2002gain','article','Cumulated Gain-Based Evaluation of IR Techniques','Järvelin, Kalervo and Kekäläinen, Jaana',2002,'ACM Transactions on Information Systems, 20(4), 422-446','https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques/','10.1145/582415.582418'),
(8,'shi2023distracted','inproceedings','Large Language Models Can Be Easily Distracted by Irrelevant Context','Shi, Freda and Chen, Xinyun and Misra, Kanishka and Scales, Nathan and Dohan, David and Chi, Ed H. and Schärli, Nathanael and Zhou, Denny',2023,'Proceedings of the 40th International Conference on Machine Learning, PMLR 202, 31210-31227','https://proceedings.mlr.press/v202/shi23a.html',''),
(9,'liu2024lost','article','Lost in the Middle: How Language Models Use Long Contexts','Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy',2024,'Transactions of the Association for Computational Linguistics, 12, 157-173','https://aclanthology.org/2024.tacl-1.9/','10.1162/tacl_a_00638'),
(10,'jiang2023structgpt','inproceedings','StructGPT: A General Framework for Large Language Model to Reason over Structured Data','Jiang, Jinhao and Zhou, Kun and Dong, Zican and Ye, Keming and Zhao, Xin and Wen, Ji-Rong',2023,'Proceedings of EMNLP 2023, 9237-9251','https://aclanthology.org/2023.emnlp-main.574/','10.18653/v1/2023.emnlp-main.574'),
(11,'ghamlouch2025taxoria','inproceedings','Enriching Taxonomies Using Large Language Models','Ghamlouch, Zeinab and Alam, Mehwish',2025,'ECAI 2025, Demo Track','https://hal.science/hal-05173197v1','10.3233/FAIA251441'),
(12,'pan2024roadmap','article','Unifying Large Language Models and Knowledge Graphs: A Roadmap','Pan, Shirui and Luo, Linhao and Wang, Yufei and Chen, Chen and Wang, Jiapu and Wu, Xindong',2024,'IEEE Transactions on Knowledge and Data Engineering','https://arxiv.org/abs/2306.08302','10.1109/TKDE.2024.3352100'),
(13,'sun2024tog','inproceedings','Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph','Sun, Jiashuo and Xu, Chengjin and Tang, Lumingyuan and Wang, Saizhuo and Lin, Chen and Gong, Yeyun and Ni, Lionel M. and Shum, Heung-Yeung and Guo, Jian',2024,'International Conference on Learning Representations','https://arxiv.org/abs/2307.07697','')]
allrefs=[];bib=[];md=['# References','', 'Primary-source metadata checked 23 September 2026. Numbering is shared across the revision documents. Preprints are identified explicitly; publication year is used for published papers.','']
for n,key,kind,title,authors,year,venue,url,doi in refs:
    names=[]
    for name in authors.split(' and '):
        last,first=name.split(', ',1);names.append(first+' '+last)
    citation=f'[{n}] '+', '.join(names)+f'. ({year}). {title}. {venue}.'
    md += [citation+f' [Source]({url})'+(f' DOI: [{doi}](https://doi.org/{doi}).' if doi else ''),'']
    data=dict(number=n,key=key,citation=citation,url=url,doi=doi);allrefs.append(data)
    venuekey={'article':'journal','misc':'howpublished','inproceedings':'booktitle'}[kind]
    fields=dict(title=title,author=authors,year=str(year),**{venuekey:venue},url=url)
    if doi:fields['doi']=doi
    bib.append('@'+kind+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}\n')
(ROOT/'docs/references.md').write_text('\n'.join(md))
(ROOT/'docs/references.bib').write_text('\n'.join(bib))
(ROOT/'docs/references.json').write_text(json.dumps(allrefs,indent=2,ensure_ascii=False))
