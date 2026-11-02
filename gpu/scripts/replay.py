"""Build an offline viewer for selected GPU match traces; no game or server required."""
import argparse,gzip,json,struct
from pathlib import Path
import numpy as np

def portable(path):
    if path.startswith('\\\\?\\'):path=path[4:]
    return Path('/mnt/'+path[0].lower()+'/'+path[3:].replace('\\','/')) if not Path(path).exists() and len(path)>1 and path[1]==':' else Path(path)

def metadata_width(header):
    width=header['width']-(header['unit_capacity']*header['unit_fields']+
        header['projectile_capacity']*header['projectile_fields']+header['tree_count']+2*header.get('signal_fields',0))
    if width not in (32,40):raise ValueError(f'Unsupported simulator metadata width: {width}')
    return width

def main():
    p=argparse.ArgumentParser();p.add_argument('logs',nargs='+');p.add_argument('--output',required=True);p.add_argument('--match',type=int,default=0);p.add_argument('--first-episode',action='store_true');a=p.parse_args()
    frames=[];last_cycle=None;gaps=[];first_header=None
    for file in sorted(a.logs):
        raw=gzip.decompress(Path(file).read_bytes())
        if raw[:8]!=b'SFDUEL02':raise ValueError('Unknown simulator trace')
        size=struct.unpack_from('<I',raw,8)[0];h=json.loads(raw[12:12+size]);data=np.frombuffer(raw[12+size:],dtype='<f4')
        if data.size!=h['frames']*h['matches']*h['width']:raise ValueError('Truncated trace')
        if not 0<=a.match<h['matches']:raise ValueError('Match is not recorded')
        if last_cycle is not None and h['cycle']!=last_cycle+1:gaps.append(len(frames))
        last_cycle=h['cycle'];first_header=first_header or h
        for row in data.reshape(h['frames'],h['matches'],h['width'])[:,a.match]:
            u0=metadata_width(h);u1=u0+h['unit_capacity']*h['unit_fields'];s1=u1+h['projectile_capacity']*h['projectile_fields']
            units=row[u0:u1].reshape(h['unit_capacity'],h['unit_fields']);shots=row[u1:s1].reshape(h['projectile_capacity'],h['projectile_fields'])
            frames.append({'tick':float(row[0]),'episode':int(row[1]),'ended':bool(row[2]),
                'units':[[i,*[round(float(x),4) for x in u]] for i,u in enumerate(units) if u[7]>0 or i<6],
                'shots':shots[shots[:,0]>0].round(4).tolist(),'trees':np.flatnonzero(row[s1:s1+h['tree_count']]>0).tolist(),
                'signals':row[s1+h['tree_count']:].reshape(2,h['signal_fields']).round(5).tolist() if h.get('signal_fields') else []})
            if a.first_episode and frames[-1]['ended']:break
        if a.first_episode and frames[-1]['ended']:break
    calibration=portable(first_header['calibration']);metadata=calibration/'engine.json'
    if not metadata.exists():metadata=calibration/'calibration.json'
    cap=json.loads(metadata.read_text())
    grid=np.fromfile(calibration/cap.get('geometry','geometry.f32'),dtype='<f4').reshape(512,512,2)[::4,::4,0].astype(int).tolist()
    payload=json.dumps({'frames':frames,'gaps':gaps,'trees':cap['trees'],'terrain':grid,'match':a.match,'source':a.logs},separators=(',',':')).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>SF Duel — GPU replay</title>
<style>body{margin:0;background:#111921;color:#e2ebf4;font:15px system-ui}main{max-width:1100px;margin:auto;padding:20px}h1{font-size:24px}p{color:#9fb1c1}canvas{background:#17252b;width:100%;max-height:70vh;aspect-ratio:1;object-fit:contain}nav{display:flex;align-items:center;gap:14px;margin:15px 0;flex-wrap:wrap}button,select{background:#263b49;border:1px solid #617c8e;color:white;border-radius:5px;padding:8px}#seek{flex:1}#status{font-variant-numeric:tabular-nums}small{color:#9fb1c1}</style>
<main><h1>SF Duel · GPU replay</h1><p>Полное состояние для разбора. Эта информация не является входом политики.</p>
<nav><button id="play">▶ Воспроизвести</button><button id="jump-end">К завершению</button><input id="seek" type="range" min="0" value="0"><select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option><option>8</option></select><span>×</span><span id="status"></span></nav>
<canvas id="map" width="900" height="900"></canvas><p id="info"></p><pre id="events" style="white-space:pre-wrap;color:#b9cbd7"></pre><small id="source"></small></main>
<script>const D=PAYLOAD;const C=document.querySelector('#map'),ctx=C.getContext('2d'),seek=document.querySelector('#seek'),play=document.querySelector('#play');seek.max=D.frames.length-1;const terminal=Math.max(0,D.frames.findIndex(f=>f.ended||(f.signals||[]).some(s=>s[9]>0)));let playing=false,index=location.hash==='#end'?terminal:Math.max(0,Math.min(D.frames.length-1,Math.floor(Number(location.hash.slice(1)))||0)),last=0,carry=0;
const bg=document.createElement('canvas');bg.width=bg.height=900;const b=bg.getContext('2d');const colors=['#284653','#385647','#456044','#54704d','#67745a','#7b826a'];for(let y=0;y<128;y++)for(let x=0;x<128;x++){b.fillStyle=colors[Math.min(5,D.terrain[y][x])];b.fillRect(x*900/128,(127-y)*900/128,900/128+1,900/128+1)}
const xy=(x,y)=>[(x+8192)/16384*900,(8192-y)/16384*900];
function draw(){const f=D.frames[index];ctx.drawImage(bg,0,0);ctx.fillStyle='#172e20';for(const i of f.trees){const t=D.trees[i],p=xy(t.x,t.y);ctx.beginPath();ctx.arc(p[0],p[1],2.2,0,7);ctx.fill()}
for(const u of f.units){const [id,x,y,hp,max,facing,team,kind,active,mana,souls,windup,castEnd,spell,target]=u;if(!active)continue;const p=xy(x,y),r=kind===0?6:kind===1?8:3;ctx.fillStyle=team===0?'#68dda2':'#f47c81';ctx.beginPath();if(kind===1)ctx.rect(p[0]-r,p[1]-r,r*2,r*2);else ctx.arc(p[0],p[1],r,0,7);ctx.fill();ctx.strokeStyle='#f4f8ff';ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(p[0]+Math.cos(facing)*12,p[1]-Math.sin(facing)*12);ctx.stroke();ctx.fillStyle='#10191e';ctx.fillRect(p[0]-10,p[1]-13,20,3);ctx.fillStyle='#c3facd';ctx.fillRect(p[0]-10,p[1]-13,20*Math.max(0,hp/max),3);
if(u[22]){ctx.strokeStyle='#ffe29c';ctx.strokeRect(p[0]-5,p[1]-5,10,10)}
if(kind===0&&spell>=1&&spell<=3){const range=[200,450,700][spell-1],q=xy(x+Math.cos(facing)*range,y+Math.sin(facing)*range);ctx.strokeStyle='#ffc270';ctx.beginPath();ctx.arc(q[0],q[1],250/16384*900,0,7);ctx.stroke()}}
ctx.fillStyle='#ffe4a1';for(const s of f.shots){const p=xy(s[1],s[2]);ctx.beginPath();ctx.arc(p[0],p[1],2,0,7);ctx.fill()}
document.querySelector('#status').textContent=`${(f.tick/30).toFixed(2)} с · эпизод ${f.episode}${f.ended?' · завершён':''}`;document.querySelector('#info').textContent=f.units.filter(u=>u[0]<2).map(u=>`${u[0]===0?'Radiant':'Dire'}: HP ${Math.round(u[3])}/${Math.round(u[4])}, mana ${Math.round(u[9])}, души ${Math.round(u[10])}${u.length>23?', смерти '+Math.round(u[23]):''}${!u[8]?' · мёртв':''}`).join(' · ')+(D.gaps.includes(index)?' · ПРОПУСК ЛОГА':'');seek.value=index}
const drawMap=draw;draw=function(){drawMap();const f=D.frames[index];document.querySelector('#events').textContent=(f.signals||[]).map((s,i)=>`${i===0?'Radiant':'Dire'}: action=${s[0]} buy=${s[1]} item=${s[2]} target=${s[3]} xy=(${s[4]},${s[5]}) reward=${s[8].toFixed(5)} terminal=${s[9]} memory_reset=${s[10]} policy=${s[12]}`).join('\\n')};
play.onclick=()=>{playing=!playing;play.textContent=playing?'⏸ Пауза':'▶ Воспроизвести'};document.querySelector('#jump-end').onclick=()=>{index=terminal;carry=0;draw()};seek.oninput=()=>{index=Number(seek.value);carry=0;draw()};function frame(t){if(playing&&last){carry+=(t-last)/1000*Number(document.querySelector('#speed').value);while(carry>=4/30){carry-=4/30;if(index<D.frames.length-1)index++;else{playing=false;break}}draw()}last=t;requestAnimationFrame(frame)}document.querySelector('#source').textContent=D.source.join(' · ');draw();requestAnimationFrame(frame);</script>'''.replace('PAYLOAD',payload)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(html,encoding='utf-8')
    print(json.dumps({'output':str(output),'frames':len(frames),'gaps':gaps,'match':a.match}))
if __name__=='__main__':main()
