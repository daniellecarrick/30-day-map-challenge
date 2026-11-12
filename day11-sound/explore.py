import json, pandas as pd, sys
def cat(t,d):
    d=(d or '').lower()
    if 'Helicopter' in t: return 'Helicopters'
    if 'ice cream' in d: return 'Ice cream truck'
    if 'dog' in d or 'animal' in d: return 'Animals'
    if t=='Noise - Vehicle' or 'car/truck' in d or 'boat' in d: return 'Vehicles'
    if 'construction' in d or 'jack' in d or 'carting' in d or 'manufactur' in d or 'lawn' in d: return 'Construction & machines'
    if 'alarm' in d or 'condition' in d: return 'Alarms & AC'
    if 'banging' in d: return 'Banging/pounding'
    if 'talking' in d or 'television' in d: return 'Talking & TV'
    if 'music' in d: return 'Music & parties'
    return 'Other'
d=pd.DataFrame(json.load(open('by_cb_2025.json'))); d.n=d.n.astype(int)
d['cat']=[cat(t,x) for t,x in zip(d.complaint_type,d.descriptor)]
d=d[d.community_board.str.match(r'^\d\d [A-Z]') & ~d.community_board.str.match(r'^\d\d Unspec') & (d.cat!='Other')]
print(d.community_board.nunique(),'boards')
p=d.pivot_table(index='community_board',columns='cat',values='n',aggfunc='sum',fill_value=0)
p=p[p.sum(1)>500]
city=p.sum()/p.values.sum()
print((city*100).round(1).sort_values(ascending=False))
share=p.div(p.sum(1),axis=0)
print('\nTop by raw count:'); print(share.idxmax(1).value_counts())
lq=(share/city).where(p>=100)
lq2=lq.drop(columns=['Music & parties'])
print('\nTop by over-index (excl. music):'); print(lq.idxmax(1).value_counts())
out=pd.DataFrame({'total':p.sum(1),'top':share.idxmax(1),'top_share':share.max(1).round(2),'sig':lq.idxmax(1),'sig_lq':lq.max(1).round(1)})
pd.set_option('display.width',200,'display.max_rows',100)
print(out.sort_values('sig'))
h=pd.DataFrame(json.load(open('by_hour_2025.json'))); h.n=h.n.astype(int); h.hh=h.hh.astype(int)
h['cat']=[cat(t,x) for t,x in zip(h.complaint_type,h.descriptor)]
hp=h.pivot_table(index='hh',columns='cat',values='n',aggfunc='sum',fill_value=0)
print('\nPeak hour per category:'); print(hp.idxmax())
print((hp.div(hp.sum(),axis=1)*100).round(1).T.to_string())
