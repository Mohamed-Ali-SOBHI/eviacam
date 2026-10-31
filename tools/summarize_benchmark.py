"""Summarize C++ results and fail if the documented comparative acceptance gates fail."""
import csv
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'tests/generated/results.csv').open()))
summary={}
for name in ('haar','yunet'):
    subset=[r for r in rows if r['detector']==name]
    pos=[r for r in subset if int(r['expected'])]
    neg=[r for r in subset if not int(r['expected'])]
    ms=[float(r['ms']) for r in subset]
    summary[name]={'positive_cases':len(pos),'detected_faces':sum(int(r['true_positive']) for r in pos),
      'negative_cases':len(neg),'negative_false_positives':sum(int(r['false_positive']) for r in neg),
      'all_false_positives':sum(int(r['false_positive']) for r in subset),
      'mean_ms':float(np.mean(ms)),'median_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95)),
      'groups':{}}
    for group in sorted(set(r['group'] for r in pos)):
        cases=[r for r in pos if r['group']==group]
        summary[name]['groups'][group]={'cases':len(cases),'detected':sum(int(r['true_positive']) for r in cases)}
h,y=summary['haar'],summary['yunet']
gates={
 'overall_gain_at_least_20_percentage_points':(y['detected_faces']-h['detected_faces'])/y['positive_cases']>=.2,
 'reference_no_regression':y['groups']['reference']['detected']>=h['groups']['reference']['detected'],
 'light_gain':y['groups']['light']['detected']>h['groups']['light']['detected'],
 'distance_gain':y['groups']['distance']['detected']>h['groups']['distance']['detected'],
 'negative_false_positives_no_increase':y['negative_false_positives']<=h['negative_false_positives'],
 'p95_under_default_400ms_detection_period':y['p95_ms']<400,
}
output={'summary':summary,'gates':gates,'scope':'126 correlated synthetic stress cases from 3 portraits, not a representative benchmark; IoU >= 0.3; two CPU threads; 320x240; exact upstream Haar parameters on the same modern OpenCV build.'}
(root/'tests/generated/summary.json').write_text(json.dumps(output,indent=2),encoding='utf8')
print(json.dumps(output,indent=2))
raise SystemExit(0 if all(gates.values()) else 1)
