use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

extern crate rand;
use rand::prelude::*;

#[derive(Debug)]
struct Vertex {
    id: VertId,
    w: Weight,
}

struct Link {
    src: VertId,
    dst: VertId,
    w: Weight,
}

type VertId = u32;
type ProcId = i32;
type Tick = u32;
type TaskId = u32;
type Weight = u32;
type Importance = u32;

#[derive(Debug)]
struct Task {
    id: TaskId,
    w: Weight,
    imp: Importance,
    children: Vec<(TaskId, Weight)>,
    parents: Vec<(TaskId, Weight)>,
}

fn tasks_from(vertices: Vec<Vertex>, links: Vec<Link>) -> Vec<Task> {
    let mut tasks = Vec::new();
    let length = vertices.len() as u32;

    // Create tasks
    for Vertex { id, w } in vertices.into_iter() {
        if id >= length {
            panic!("Using VertId beyond upper bound.")
        }
        if id != (tasks.len() as u32) {
            panic!("Pushing for wrong index");
        }

        tasks.push(Task {
            id: id,
            w: w,
            imp: 0,
            children: Vec::new(),
            parents: Vec::new(),
        });
    }

    // Add links
    for Link { src, dst, w } in links.into_iter() {
        if src >= (tasks.len() as u32) || dst >= (tasks.len() as u32) {
            panic!("Link from or to out-of-bounds Vertex");
        }
        tasks[src as usize].children.push((dst, w));
        tasks[dst as usize].parents.push((src, w));
    }

    // Set importance
    // Assumes that at each iteration there will be at least 1 Task whose all
    // children have their Importance set. Utilizes exactly 1 such Task at each iteration.
    for _ in 0..tasks.len() {
        let (index, imp) = tasks
            .iter()
            .filter(|Task { imp, .. }| *imp == 0)
            .find_map(
                |Task { id, w, children, .. }| {
                    children.iter()
                        .try_fold(w.clone(), |acc, &(child_id, _)| {
                            let imp = tasks[child_id as usize].imp;
                            if imp != 0 {
                                Some(acc + imp)
                            } else {
                                None
                            }
                        })
                        .and_then(|imp| Some((id.clone(), imp)))
                },
            )
            .unwrap(); // can safely unwrap because of the assumption described above
        tasks[index as usize].imp = imp;
        // println!("{} for {}", imp, index);
    }

    tasks
}


fn print_tasks(tasks: &Vec<Task>) {
    for Task {id, w, children, parents, imp} in tasks.iter() {
        println!("Task #{} [{}] <{}>:", id+1, w, imp);
        print!("\t->");
        for (dst, w) in children.iter() {
            print!(" #{}[{}]", dst+1, w);
        }
        println!();
        print!("\t<-");
        for (src, w) in parents.iter() {
            print!(" #{}[{}]", src+1, w);
        }
        println!();
    }
}

fn main() {
    let (vertices, links) = populate_random();
    // let (vertices, links) = (populate_vertices(), populate_links());
    let tasks = tasks_from(vertices, links);
    print_tasks(&tasks);
    let mut s = System::new(tasks);
    s.plan();
    s.export("planning.txt".to_string());
}


struct OutTask {
    start: Tick,
    proc: ProcId,
    weight: Weight,
    id: TaskId,
}

#[derive(Debug)]
struct OutLink {
    src_core: ProcId,
    dst_core: ProcId,
    start: Tick,
    weight: Weight,
    src_task: TaskId,
    dst_task: TaskId,
}

impl OutLink {
    fn serialize(&self, leftmost_proc: &ProcId) -> String {
        let mut s = "OutLink\n".to_string();

        s += "src_core:";
        s += &(self.src_core - leftmost_proc).to_string();
        s += "\n";

        s += "dst_core:";
        s += &(self.dst_core - leftmost_proc).to_string();
        s += "\n";

        s += "weight:";
        s += &self.weight.to_string();
        s += "\n";

        s += "start:";
        s += &(self.start + 1).to_string();
        s += "\n";

        s += "src_task:";
        s += &self.src_task.to_string();
        s += "\n";

        s += "dst_task:";
        s += &self.dst_task.to_string();
        s += "\n";

        s
    }
}

impl OutTask {
    fn serialize(&self, leftmost_proc: &ProcId) -> String {
        let mut s = "OutTask\n".to_string();

        s += "start:";
        s += &(self.start + 1).to_string();
        s += "\n";

        s += "proc:";
        s += &(self.proc - leftmost_proc).to_string();
        s += "\n";

        s += "weight:";
        s += &self.weight.to_string();
        s += "\n";

        s += "id:";
        s += &self.id.to_string();
        s += "\n";

        s
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct ProcPair {
    left: ProcId,
    right: ProcId,
}

impl ProcPair {
    fn new(left: ProcId, right: ProcId) -> ProcPair {
        if left < right { ProcPair {left: left,  right: right} }
        else            { ProcPair {left: right, right: left}  }
    }
}

fn clone_buses(buses: &Buses) -> Buses {
    let mut new_buses = HashMap::new();
    for (proc_pair, tick) in buses.iter() {
        new_buses.insert(proc_pair.clone(), tick.clone());
    }
    new_buses
}

// proc -> vec<is_busy>
type Processors = HashMap<ProcId, Vec<bool>>;
// leftproc
type Buses = HashMap<ProcPair, Tick>;
// where the tasks was planned and where it finished
type PlannedTasks = HashMap<TaskId, (ProcId, Tick)>;


struct System {
    unplanned_tasks: Vec<Task>,
    out_tasks: Vec<OutTask>,
    out_links: Vec<OutLink>,
    leftmost_proc: ProcId,
    rightmost_proc: ProcId,
    processors: Processors,
    buses: Buses,
    planned_tasks: PlannedTasks,
}

#[derive(Debug)]
struct Scenario {
    proc: ProcId,
    buses: Buses,
    start: Tick,
    new_links: Vec<OutLink>,
    score: u32,
}


fn gen_paths(src: ProcId, dst: ProcId) -> Vec<(ProcId, ProcId)> {
    if src < dst { (src..dst).map(|x| (x, x+1))      .collect() }
    else         { (dst..src).map(|x| (x+1, x)).rev().collect() }
}


impl System {
    fn export(&self, path: String) {
        let f = File::create(path).expect("Unable to create file");
        let mut f = BufWriter::new(f);

        self.out_tasks.iter().for_each(|task|
            f.write_all(task.serialize(&self.leftmost_proc).as_bytes())
                .expect("Unable to write data"));
        self.out_links.iter().for_each(|link|
            f.write_all(link.serialize(&self.leftmost_proc).as_bytes())
                .expect("Unable to write data"));
    }

    fn plan(&mut self) {
        while let Some(task) = self.pop_next() {
            // println!("Working with {:?}", task);
            let Scenario {proc, buses, start, mut new_links, ..} =
                (self.leftmost_proc..=self.rightmost_proc)
                .map(|proc| self.play_scenario(proc, &task))
                // .inspect(|scenario| println!("      Considering {:#?}", scenario))
                .min_by_key(|scenario| scenario.score)
                .unwrap();
            self.buses = buses;
            self.out_links.append(&mut new_links);
            self.out_tasks.push(OutTask {
                start,
                proc,
                weight: task.w,
                id: task.id,
            });
            self.place_at_proc(&proc, &start, &task.w);
            self.planned_tasks.insert(task.id, (proc, start + task.w));
            self.enlarge_if_needed(proc);
        }
    }

    fn place_at_proc(&mut self, proc: &ProcId, start: &Tick, w: &Weight) {
        let start = start.clone();
        let length = self.processors[proc].len();
        for _i in length..(start as usize) { // if start > length
            self.processors.get_mut(proc).unwrap().push(false);
        }
        let length = self.processors[proc].len();
        for i in (start as usize)..length {
            self.processors.get_mut(proc).unwrap()[i] = true;
        }
        for _i in length..((start+w) as usize) {
            self.processors.get_mut(proc).unwrap().push(true);
        }
    }

    fn enlarge_if_needed(&mut self, proc: ProcId) {
        if proc == self.leftmost_proc {
            self.leftmost_proc -= 1;
            self.processors.insert(self.leftmost_proc, Vec::new());
            self.buses.insert(ProcPair::new(self.leftmost_proc, self.leftmost_proc + 1), 0);
        }
        if proc == self.rightmost_proc {
            self.rightmost_proc += 1;
            self.processors.insert(self.rightmost_proc, Vec::new());
            self.buses.insert(ProcPair::new(self.rightmost_proc, self.rightmost_proc - 1), 0);
        }
    }

    fn finish_of(&self, task: &TaskId) -> Tick {
        let (_, tick) = self.planned_tasks.get(task).unwrap();
        tick.clone()
    }

    fn proc_of(&self, task: &TaskId) -> ProcId {
        let (proc, _) = self.planned_tasks.get(task).unwrap();
        proc.clone()
    }

    fn bus_finish_of(&self, src: &ProcId, dst: &ProcId, buses: &Buses) -> Tick {
        buses.get(&ProcPair::new(src.clone(), dst.clone())).unwrap().clone()
    }

    fn play_scenario(&self, proc: ProcId, task: &Task) -> Scenario {
        let mut buses = clone_buses(&self.buses);
        let (transmission_finish, new_links): (Tick, Vec<OutLink>) = task.parents.iter()
            .filter(|(parent, _)| self.proc_of(parent) != proc)
            .map(|(parent, weight)| {
                let (transmission_finish, new_links) = gen_paths(self.proc_of(&parent), proc).into_iter()
                    .fold((self.finish_of(&parent), Vec::new()),
                            |(src_finish, mut links), (src_core, dst_core)| {
                        let start = std::cmp::max(
                            src_finish,
                            // work upon the being updated buses
                            self.bus_finish_of(&src_core, &dst_core, &buses)
                        );
                        links.push(OutLink {
                            src_core,
                            dst_core,
                            start: start.clone(),
                            weight: weight.clone(),
                            src_task: parent.clone(),
                            dst_task: task.id.clone(),
                        });
                        (start + weight.clone(), links)
                    });
                // Imprint these links, so that all consequent paths of links
                // work upon the updated buses.
                for OutLink {src_core, dst_core, start, weight, ..} in new_links.iter() {
                    buses.insert(
                        ProcPair::new(src_core.clone(), dst_core.clone()),
                        (start + weight).clone());
                }
                (transmission_finish, new_links)
            })
            .fold((0, Vec::new()), |(finish, mut links), (cur_finish, mut cur_links)| {
                links.append(&mut cur_links);
                (std::cmp::max(finish, cur_finish), links)
            });

        let start = self.find_consecutive_block(&proc, &transmission_finish, task.w.clone());
        print!("Found block with w={} starting at {}", task.w, start);
        // let start = std::cmp::max(free_on_proc, &transmission_finish).clone();
        Scenario {
            score: start.clone(),
            start,
            new_links,
            buses,
            proc,
        }
    }

    fn find_consecutive_block(&self, proc_id: &ProcId, starting_tick: &Tick, w: Weight) -> Tick {
        let proc = &self.processors[proc_id];
        let mut cur: Tick = starting_tick.clone();
        loop {
            if cur as usize >= proc.len() { return cur; }
            if !proc[cur as usize] { // is free
                let mut succ = true;
                for i in cur..cur+w {
                    if i as usize >= proc.len() { break; }
                    if proc[i as usize] { // is busy
                        succ = false;
                        break;
                    }
                }
                if succ { return cur; }
            }
            cur += 1;
        };
    }

    fn new(tasks: Vec<Task>) -> System {
        let mut processors = HashMap::new();
        processors.insert(0, Vec::new());
        System {
            unplanned_tasks: tasks,
            out_tasks: vec![],
            out_links: vec![],
            leftmost_proc: 0,
            rightmost_proc: 0,
            processors: processors,
            buses: HashMap::new(),
            planned_tasks: HashMap::new(),
        }
    }

    fn pop_next(&mut self) -> Option<Task> {
        if self.unplanned_tasks.is_empty() { None }
        else {
            let (index, _) = self.unplanned_tasks.iter().enumerate()
                .max_by_key(|(_, Task {imp, ..})| imp)
                .unwrap();
            Some(self.unplanned_tasks.remove(index))
        }
        // if self.unplanned_tasks.len() > 0
        //      { Some(self.unplanned_tasks.remove(0)) }
        // else { None }
    }
}


fn populate_random() -> (Vec<Vertex>, Vec<Link>) {
    let vertex_count = 15;
    let min_per_layer = 2;
    let max_per_layer = vertex_count / 2;
    let min_vertex_weight = 2;
    let max_vertex_weight = 8;
    let seed = [6,4,3,8, 7,9,8,10, 14,18,12,12, 14,15,16,17];
    let mut rng = SmallRng::from_seed(seed);

    // Vertices
    let mut done_vertices_count = 0;
    let mut id: VertId = 0;
    let mut layers = Vec::new();
    while done_vertices_count < vertex_count {
        let mut layer = Vec::new();
        let count: u32 = {
            let r = rng.gen_range(min_per_layer, max_per_layer + 1);
            let left = vertex_count - done_vertices_count;
            if left < r {left} else {r}
        };
        for _ in 0..count {
            let weight: Weight = rng.gen_range(min_vertex_weight, max_vertex_weight + 1);
            layer.push(Vertex {id: id, w: weight});
            id += 1;
        }
        done_vertices_count += count;

        layers.push(layer);
    }
    // println!("{:#?}", layers);

    // Links
    let links_count = (vertex_count * (vertex_count - 1) / 2) / 10;
    // println!("Will create {} links", links_count);
    let min_link_weight = 1;
    let max_link_weight = 3;
    let mut links = Vec::new();
    let belongs_to_chunk = |id: u32, chunks: &Vec<Vec<VertId>>| -> Option<usize> {
        chunks.iter()
            .enumerate()
            .find_map(|(index, chunk)|
                if chunk.iter()
                    .find(|&& x| x == id)
                    .is_some() {Some(index)}
                else {None}
            )
    };
    let mut done_links_count = 0;
    let mut chunks = Vec::new();
    let mut converged = false;
    while !converged || done_links_count < links_count {
        // if !converged { println!("Have chunks: {:#?}", chunks); }
        let layer_src_index = rng.gen_range(0, layers.len() - 1);
        let layer_dst_index = rng.gen_range(layer_src_index + 1, layers.len());
        let layer_src = layers.get(layer_src_index).unwrap();
        let layer_dst = layers.get(layer_dst_index).unwrap();
        let src_index = layer_src.get(rng.gen_range(0, layer_src.len())).unwrap().id.clone();
        let dst_index = layer_dst.get(rng.gen_range(0, layer_dst.len())).unwrap().id.clone();
        // Retry if such link already exists

        if links.iter()
            .find(|&& Link {src, dst, ..}| (src == src_index) && (dst == dst_index))
            .is_some() { continue; }
        let weight: u32 = rng.gen_range(min_link_weight, max_link_weight + 1);
        links.push(Link { src: src_index, dst: dst_index, w: weight });
        done_links_count += 1;

        // Have nothing to trach if have already converged
        if converged { continue; }

        let chunk_src = belongs_to_chunk(src_index, &chunks);
        let chunk_dst = belongs_to_chunk(dst_index, &chunks);
        // println!("Have: {} and {}", src_index, dst_index);
        if chunk_src.is_none() && chunk_dst.is_none() {
            // Create new chunk
            // println!("Creating new chunk");
            chunks.push(vec![src_index, dst_index]);
        } else if chunk_src.is_none() && chunk_dst.is_some() {
            // Add src to dst_chunk
            // println!("Adding src to dst");
            chunks.get_mut(chunk_dst.unwrap()).unwrap().push(src_index);
        } else if chunk_src.is_some() && chunk_dst.is_none() {
            // Add dst to src_chunk
            // println!("Adding dst to src");
            chunks.get_mut(chunk_src.unwrap()).unwrap().push(dst_index);
        } else { // both Some
            let src = chunk_src.unwrap();
            let dst = chunk_dst.unwrap();
            // println!("{} , {}", src, dst);
            if src != dst {
                // println!("Not same, merging");
                // Merge chunks
                let mut other = chunks.remove(dst);
                chunks.get_mut(if dst < src {src - 1} else {src}).unwrap().append(&mut other);
                // converted=true was here
            } else {
                // println!("Same, nothing");
            }
        }
        // Check if all are connected
        if chunks.len() == 1 && chunks[0].len() == vertex_count as usize {
            converged = true;
        }
    }

    // println!("Done {} links total", done_links_count);

    (layers.into_iter().flatten().collect(), links)
}

fn populate_vertices() -> Vec<Vertex> {
    vec![
        Vertex {
            id: 0,
            w: 3,
        },
        Vertex {
            id: 1,
            w: 4,
        },
        Vertex {
            id: 2,
            w: 5,
        },
        Vertex {
            id: 3,
            w: 3,
        },
        Vertex {
            id: 4,
            w: 3,
        },
        Vertex {
            id: 5,
            w: 2,
        },
        Vertex {
            id: 6,
            w: 4,
        },
    ]
}

fn populate_links() -> Vec<Link> {
    vec![
        Link {
            src: 0,
            dst: 3,
            w: 1,
        },
        Link {
            src: 0,
            dst: 2,
            w: 2,
        },
        Link {
            src: 1,
            dst: 2,
            w: 1,
        },
        Link {
            src: 1,
            dst: 6,
            w: 2,
        },
        Link {
            src: 3,
            dst: 4,
            w: 1,
        },
        Link {
            src: 3,
            dst: 5,
            w: 2,
        },
        Link {
            src: 2,
            dst: 5,
            w: 1,
        },
    ]
}
