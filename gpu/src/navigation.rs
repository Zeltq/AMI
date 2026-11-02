//! Offline public-map flow tables. Runtime path selection is a device lookup.
use anyhow::{Result,ensure};
use std::{path::Path,collections::{BinaryHeap,VecDeque},cmp::Reverse};
pub const SIDE:usize=256;pub const GOALS:usize=64;
pub const WORDS_PER_GOAL:usize=SIDE*SIDE/8;
const DX:[i32;8]=[1,1,0,-1,-1,-1,0,1];const DY:[i32;8]=[0,1,1,1,0,-1,-1,-1];

pub fn build(engine:&Path)->Result<serde_json::Value>{
    let mut config:serde_json::Value=serde_json::from_slice(&std::fs::read(engine)?)?;
    let root=std::path::PathBuf::from(config["calibration"].as_str().unwrap());
    let movement=config["movement_geometry"].as_str().filter(|s|!s.is_empty());
    let bytes=std::fs::read(root.join(movement.unwrap_or(config["geometry"].as_str().unwrap())))?;
    ensure!(bytes.len()==512*512*8,"incomplete map");
    let geometry:Vec<f32>=bytes.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
    let pass=|x:i32,y:i32|x>=0 && y>=0 && x<512 && y<512 && geometry[(y as usize*512+x as usize)*2+1]>0.5;
    let mut edges=vec![0u8;SIDE*SIDE];
    for y in 0..SIDE as i32 {for x in 0..SIDE as i32 {
        let p=y as usize*SIDE+x as usize;
        for direction in 0..8 {
            let nx=x+DX[direction];let ny=y+DY[direction];
            if nx<0 || ny<0 || nx>=SIDE as i32 || ny>=SIDE as i32 {continue;}
            let scale=512/SIDE as i32;let centre=scale/2;
            let clear=(0..=scale).all(|s|pass(x*scale+centre+DX[direction]*s,y*scale+centre+DY[direction]*s));
            let corners=direction%2==0 || (pass(nx*scale+centre,y*scale+centre)&&pass(x*scale+centre,ny*scale+centre));
            if clear&&corners {edges[p]|=1<<direction;}
        }
    }}
    let mut components=vec![usize::MAX;SIDE*SIDE];let mut sizes=vec![];
    for start in 0..edges.len(){if edges[start]!=0 && components[start]==usize::MAX {
        let component=sizes.len();let mut count=0;let mut queue=VecDeque::from([start]);components[start]=component;
        while let Some(p)=queue.pop_front(){count+=1;for direction in 0..8 {if edges[p]&(1<<direction)!=0 {
            let q=(p as i32+DX[direction]+DY[direction]*SIDE as i32) as usize;
            if components[q]==usize::MAX {components[q]=component;queue.push_back(q);}
        }}}sizes.push(count);
    }}
    let main=sizes.iter().enumerate().max_by_key(|(_,n)|*n).map(|(i,_)|i).ok_or_else(||anyhow::anyhow!("no traversable component"))?;
    let mut packed=vec![0x99999999u32;GOALS*GOALS*SIDE*SIDE/8];
    let mut goal_offsets=vec![];
    for gy in 0..GOALS {for gx in 0..GOALS {
        let cx=(gx*2+1)*(SIDE/GOALS);let cy=(gy*2+1)*(SIDE/GOALS);
        let goal=(0..SIDE*SIDE).filter(|p|components[*p]==main).min_by_key(|p|{
            let dx=(*p%SIDE*2+1) as i32-cx as i32;let dy=(*p/SIDE*2+1) as i32-cy as i32;dx*dx+dy*dy
        }).unwrap();
        let mut distances=vec![u32::MAX;SIDE*SIDE];let mut directions=vec![9u32;SIDE*SIDE];
        let mut heap=BinaryHeap::from([Reverse((0u32,goal))]);distances[goal]=0;directions[goal]=8;
        while let Some(Reverse((cost,p)))=heap.pop(){
            if cost!=distances[p]{continue;}
            for direction in 0..8 {if edges[p]&(1<<direction)!=0 {
                let q=(p as i32+DX[direction]+DY[direction]*SIDE as i32) as usize;
                let candidate=cost+if direction%2==0 {128}else{181};
                if candidate<distances[q] {distances[q]=candidate;directions[q]=((direction+4)%8) as u32;heap.push(Reverse((candidate,q)));}
            }}
        }
        let offset=(gy*GOALS+gx)*SIDE*SIDE/8;
        for (i,chunk) in directions.chunks_exact(8).enumerate(){packed[offset+i]=chunk.iter().enumerate().fold(0u32,|word,(shift,d)|word|(*d<<(shift*4)));}
        goal_offsets.push(goal);
    }}
    let name=if movement.is_some(){"movement-navigation64.u32"}else{"navigation64.u32"};let file=root.join(name);let temporary=file.with_extension("tmp");
    use std::io::Write;let mut output=std::io::BufWriter::new(std::fs::File::create(&temporary)?);
    for word in &packed {output.write_all(&word.to_le_bytes())?;}output.flush()?;
    drop(output);std::fs::rename(temporary,&file)?;
    config["navigation"]=serde_json::json!(name);crate::artifacts::atomic_json(engine,&config)?;
    let report=serde_json::json!({"file":file,"bytes":packed.len()*4,"nav_cell":64,"goal_cell":256,
        "connected_cells":sizes[main],"components":sizes.len(),"goal_nodes":goal_offsets,
        "approximation":"Public static passability. Exact local steering takes over in direct line of sight. Tree cuts do not regenerate this static table."});
    crate::artifacts::atomic_json(&root.join("navigation64.json"),&report)?;Ok(report)
}
