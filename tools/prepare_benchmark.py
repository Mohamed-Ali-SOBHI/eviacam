"""Reproducible, camera-free stress fixtures. Requires numpy, opencv-python, skimage, matplotlib.
Manual face boxes (not detector-generated labels). Augmentations of three portraits
are correlated and are NOT a representative face-detection validation dataset.
"""
from pathlib import Path
import hashlib
import json
import shutil
import os
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
DATA = Path('tests/data')
OUT = Path('tests/generated')
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
from skimage import data
import matplotlib.cbook
cv2.imwrite(str(DATA/'astronaut.png'), cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
cv2.imwrite(str(DATA/'coffee.png'), cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR))
cv2.imwrite(str(DATA/'brick.png'), data.brick())
shutil.copy(matplotlib.cbook.get_sample_data('grace_hopper.jpg', asfileobj=False), DATA/'grace_hopper.jpg')
if not (DATA/'lena.jpg').exists():
    urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/lena.jpg', DATA/'lena.jpg')

portraits = [('astronaut.png',(177,65,96,116)), ('lena.jpg',(210,190,180,200)),
             ('grace_hopper.jpg',(154,125,217,260))]
manifest=[]
rng=np.random.default_rng(20261031)
examples=[]

def save(case, group, image, box):
    filename=f'tests/generated/{case}.png'
    cv2.imwrite(filename,image)
    manifest.append(f'{case}\t{group}\t{filename}\t'+ '\t'.join(str(int(v)) for v in box))

for filename, box in portraits:
    original=cv2.imread(str(DATA/filename))
    x,y,w,h=box
    for width in (24,32,48,64,96,120):
        scale=width/w
        matrix=np.float32([[scale,0,160-(x+w/2)*scale],[0,scale,120-(y+h/2)*scale]])
        scene=cv2.warpAffine(original,matrix,(320,240),flags=cv2.INTER_AREA,borderValue=(95,95,95))
        truth=(round(160-width/2),round(120-h*scale/2),width,round(h*scale))
        for brightness in (1.0,0.6,0.3,0.15):
            group=('distance' if width<65 else 'reference') if brightness==1 else ('combined' if width<65 else 'light')
            image=np.clip(scene.astype(float)*brightness,0,255).astype('uint8')
            case=f'{Path(filename).stem}_w{width}_b{brightness}'
            save(case,group,image,truth)
            if width in (32,96) and brightness in (1,.3): examples.append(image)
        if width in (48,96):
            for kind in ('noise','side_light','gamma','blur'):
                image=scene.astype(float)
                if kind=='noise': image=image*.35+rng.normal(0,7,image.shape)
                elif kind=='side_light': image*=np.linspace(.15,1,320)[None,:,None]
                elif kind=='gamma': image=255*np.power(image/255,2.2)
                else: image=cv2.GaussianBlur(image,(5,5),1.2)
                save(f'{Path(filename).stem}_w{width}_{kind}',kind,np.clip(image,0,255).astype('uint8'),truth)

for filename in ('coffee.png','brick.png'):
    original=cv2.imread(str(DATA/filename))
    for i in range(12):
        image=cv2.resize(original,(320,240))
        if i%2: image=cv2.flip(image,1)
        image=np.clip(image.astype(float)*(1,.6,.3,.15)[i%4]+rng.normal(0,i%3*3,image.shape),0,255).astype('uint8')
        save(f'negative_{Path(filename).stem}_{i}','negative',image,(0,0,0,0))
for i in range(6):
    image=np.full((240,320,3),i*40,np.uint8) if i<3 else rng.integers(0,256,(240,320,3),dtype='uint8')
    save(f'negative_synthetic_{i}','negative',image,(0,0,0,0))

(OUT/'manifest.tsv').write_text('\n'.join(manifest)+'\n',encoding='utf8')
cv2.imwrite(str(OUT/'examples.jpg'),np.vstack([np.hstack(examples[i:i+4]) for i in range(0,len(examples),4)]))
# Replay through the real app: movements, distance, lighting, then no face.
writer=cv2.VideoWriter(str(OUT/'replay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240))
original=cv2.imread(str(DATA/'astronaut.png'))
x,y,w,h=portraits[0][1]
for frame in range(900):
    t=frame/30
    width=90 if t<10 else 35 if t<20 else 70
    scale=width/w
    cx=160+35*np.sin(t)
    cy=120+15*np.cos(t*.8)
    matrix=np.float32([[scale,0,cx-(x+w/2)*scale],[0,scale,cy-(y+h/2)*scale]])
    image=cv2.warpAffine(original,matrix,(320,240),borderValue=(95,95,95))
    brightness=1 if t<6 else .3 if t<16 else .6
    image=np.clip(image.astype(float)*brightness,0,255).astype('uint8')
    if t>=26: image[:]=35
    writer.write(image)
writer.release()
provenance={'seed':20261031,'size':[320,240],'positive_cases':96,'negative_cases':30,
 'boxes':dict(portraits),'sources':{
 'astronaut':'https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut (NASA, public domain)',
 'grace_hopper':'matplotlib sample_data/grace_hopper.jpg (US Navy portrait)',
 'lena':'https://github.com/opencv/opencv/blob/4.10.0/samples/data/lena.jpg',
 'negatives':'skimage.data.coffee and brick, plus generated constant/noise images'},
 'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir() if p.name in [x[0] for x in portraits]+['coffee.png','brick.png']}}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
print(f'Prepared {len(manifest)} cases and a 30-second replay. No camera opened.')
