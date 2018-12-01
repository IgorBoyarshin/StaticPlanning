use std::collections::HashMap;
use std::fmt;

extern crate rand;
use rand::prelude::*;
// use rand::rngs::StdRng;

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

// #[derive(PartialEq, Eq, Ord, PartialOrd)]

type VertId = u32;
type ProcId = i32;
type Tick = u32;
type TaskId = u32;
type StartTime = u32;
type Weight = u32;
type Importance = u32;

#[derive(PartialEq, Eq, Clone)]
enum Cell {
    Free,
    Taken(TaskId),
    Snd(TaskId),
    Rcv(TaskId),
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Cell::Free => write!(f, "____"),
            Cell::Taken(id) => write!(f, "#{:2}#", id),
            Cell::Snd(id) => write!(f, "-{:2}-", id),
            Cell::Rcv(id) => write!(f, "+{:2}+", id),
        }
    }
}

#[derive(Clone)]
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
        println!("Task #{} [{}] <{}>:", id, w, imp);
        print!("\t->");
        for (dst, w) in children.iter() {
            print!(" #{}[{}]", dst, w);
        }
        println!();
        print!("\t<-");
        for (src, w) in parents.iter() {
            print!(" #{}[{}]", src, w);
        }
        println!();
    }
}

struct Place {
    proc: ProcId,
    tick: Tick,
}

type Proc = Vec<Cell>;

#[derive(Clone)]
struct System {
    processors: HashMap<ProcId, Proc>,
    planned_tasks: Vec<Task>, // TODO: can remove
    unplanned_tasks: Vec<Task>,
    leftmost_proc: ProcId,
    rightmost_proc: ProcId,
    start_times: HashMap<TaskId, (ProcId, StartTime, Weight)>,
}

impl System {
    fn print_planning(&self) {
        // Header
        println!();
        print!("======||");
        for _ in 1..=(self.rightmost_proc - self.leftmost_proc - 1) {
            print!("======|");
        }
        println!();
        print!(" Tick ||");
        for i in (self.leftmost_proc + 1)..self.rightmost_proc {
            print!("  {:2}  |", i);
        }
        println!();
        print!("======||");
        for _ in 1..=(self.rightmost_proc - self.leftmost_proc - 1) {
            print!("======|");
        }

        // Content
        let mut index = 0;
        let mut finished = false;
        while !finished {
            println!();
            finished = true; // until proven otherwise
            print!("[ {:3}]||", index);
            // let mut prev: Option<Cell> = None;
            let mut prev_for_me = false;
            for proc in (self.leftmost_proc + 1)..self.rightmost_proc {
                let processor = &self.processors[&proc];
                let cell = if index >= processor.len() {
                    Cell::Free
                } else {
                    finished = false;
                    processor[index].clone()
                };
                if let Cell::Snd(task) = cell {
                    if prev_for_me {
                        print!(" <{:2}< |", task);
                    } else {
                        print!(" >{:2}> |", task);
                    }
                    prev_for_me = !prev_for_me;
                } else if let Cell::Rcv(task) = cell {
                    if prev_for_me {
                        print!(" >{:2}> |", task);
                    } else {
                        print!(" <{:2}< |", task);
                    }
                    prev_for_me = !prev_for_me;
                } else {
                    print!(" {} |", cell);
                }
            }
            index += 1;
        }
        println!(" ===== Total ticks");
    }

    fn new(tasks: Vec<Task>) -> System {
        let mut map: HashMap<ProcId, Vec<Cell>> = HashMap::new();
        map.insert(0, Vec::new());

        System {
            leftmost_proc: 0,
            rightmost_proc: 0,
            processors: map,
            planned_tasks: Vec::new(),
            unplanned_tasks: tasks,
            start_times: HashMap::new(),
        }
    }

    fn last_free_or_taken(proc: &Proc, min: Tick) -> Tick {
        for curr in (min..proc.len() as Tick).into_iter().rev() {
            let cell = proc.get(curr as usize).unwrap();
            if let Cell::Rcv(_) = cell {return curr}
            if let Cell::Snd(_) = cell {return curr}
        }
        min
    }

    fn fall(system: &mut System, src_proc: ProcId, dst_proc: ProcId,
            w: Weight, min_start: Tick, target: TaskId) -> Tick {
        println!("  Falling for {}>{} with min_start={}", src_proc, dst_proc, min_start);
        let (start, _free_start): (u32, Option<u32>) = {
            let proc_src = system.processors.get(&src_proc).unwrap();
            let proc_dst = system.processors.get(&dst_proc).unwrap();
            let start_in_src = System::last_free_or_taken(proc_src, min_start);
            let start_in_dst = System::last_free_or_taken(proc_dst, min_start);
            (std::cmp::max(start_in_src, start_in_dst), None)
        };

        System::displace(&mut system.start_times,
             system.processors.get_mut(&src_proc).unwrap(), src_proc, start.clone(), w.clone());
        System::displace(&mut system.start_times,
             system.processors.get_mut(&dst_proc).unwrap(), dst_proc, start.clone(), w.clone());
         
        System::place(system, Cell::Snd(target), Place {proc: src_proc, tick: start}, w);
        System::place(system, Cell::Rcv(target), Place {proc: dst_proc, tick: start}, w);

        start + w
    }

    fn displace(start_times: &mut HashMap<TaskId, (ProcId, StartTime, Weight)>,
                proc: &mut Vec<Cell>, proc_id: ProcId, start: Tick, w: Weight) {
        println!("Displacing [{} to {}) inside proc={}", start, start + w, proc_id);
        if start as usize >= proc.len() {
            println!("  Processor length ({}) is insufficient => nothing to do", proc.len());
            return;
        }
        let cell_start = proc.get(start as usize).unwrap();
        if let Cell::Rcv(_) = cell_start {
            panic!("  Displacing starting from Cell::Rcv");
        }
        if let Cell::Snd(_) = cell_start {
            panic!("  Displacing starting from Cell::Snd");
        }
        let mut actual_start = start as usize;
        if let Cell::Taken(id) = cell_start {
            println!("  Displacing Taken({})", id);
            while let Cell::Taken(curr_id) = proc.get(actual_start).unwrap_or(&Cell::Free) {
                if curr_id != id {break;}
                if actual_start == 0 {break;}
                actual_start -= 1;
            }
            println!("  Actual start={}", actual_start);
        }
        // TODO: improve
        for _ in 0..(w + (start - actual_start as u32)) {
            proc.insert(actual_start, Cell::Free);
        }
        for (task_id, (possible_proc_id, start_time, _)) in start_times {
            if *possible_proc_id != proc_id {continue;}
            if *start_time >= actual_start as u32 {
                *start_time += w;
                println!("    Increasing start_time of {} to {}", task_id, start_time);
            }
        }
    }

    // fn could_shift(start: Tick, index: Tick, end: Tick,
    //                 imp_index: Importance, imp_task: Importance) -> bool {
    //     ((index - start + 1) as f32 / (end - index - 1) as f32) <=
    //         (imp_index as f32 / imp_task as f32)
    // }
    //
    // vec<(path, link_w, parent_finish)>
    fn get_paths_to(proc: ProcId, parents: &Vec<(TaskId, Weight)>,
            start_times: &HashMap<TaskId, (ProcId, StartTime, Weight)>)
                -> Vec<(Vec<(ProcId, ProcId)>, Weight, Tick)> {
        parents.iter()
            .map(|(parent_task_id, link_w)| {
                let (parent_proc, parent_start, parent_w) =
                    start_times.get(parent_task_id).unwrap().clone();
                (parent_proc, link_w.clone(), (parent_start + parent_w).clone())
            })
            .filter(|(parent_proc, ..)| *parent_proc != proc)
            .map(|(parent_proc, link_w, parent_finish)| {
                let left_to_right = parent_proc < proc;
                let pairs = (if parent_proc < proc
                             {(parent_proc..proc)}
                        else {(proc..parent_proc)})
                    .map(|x| if left_to_right {(x, x+1)} else {(x+1, x)});
                let path = if left_to_right {pairs.collect()} else {pairs.rev().collect()};
                (path, link_w, parent_finish)
            })
            .collect()
    }

    fn plan(&mut self) {
        while self.unplanned_tasks.len() > 0 {
            self.print_planning();
            let Task {id: task_id, w: task_w, parents, ..} = self.rmv_earliest();
            // No parents => can start immediately => into a new processor
            if parents.is_empty() {
                let the_place = Place {proc: self.leftmost_proc.clone(), tick: 0};
                System::place(self, Cell::Taken(task_id), the_place, task_w);
                continue;
            }

            println!("------------------Finding best for task {}---------------------", task_id);
            let (new_system, _score) =
                (self.leftmost_proc..=self.rightmost_proc).into_iter()
                .inspect(|proc| println!("Evaluating proc {}", proc))
                .map(|proc| {
                    let paths = System::get_paths_to(proc, &parents, &self.start_times);
                    // TODO: don't account for transmission_cost
                    let transmission_cost: u32 = paths.iter()
                        .map(|(path, step_w, ..)| (path.len() as u32) * step_w.clone())
                        .sum();
                    let mut system = self.clone();
                    let mut max_finish: u32 = parents.iter()
                        .map(|(parent_task_id, _)| {
                            let (_, parent_start, parent_w) =
                                self.start_times.get(parent_task_id).expect("Parent not planned yet").clone();
                            parent_start + parent_w
                        })
                        .max().unwrap(); // can safely unwrap() because there is at least 1 parent
                    for (path, link_w, parent_finish) in paths.into_iter() {
                        let mut last_finish = parent_finish;
                        for (src_proc, dst_proc) in path.into_iter() { // for each step
                            last_finish = System::fall(&mut system,
                                src_proc, dst_proc, link_w, last_finish, task_id.clone());
                            println!("    Last finish for {}->{} is {}", src_proc, dst_proc, last_finish);
                        }
                        max_finish = std::cmp::max(max_finish, last_finish);
                    }

                    println!("====Transmission cost={}, max finish={}", transmission_cost, max_finish);
                    (system, max_finish + transmission_cost)
                })
                .inspect(|(_, score)| println!("=================Has score {}", score))
                .min_by_key(|(_, score)| score.clone())
                .unwrap();
            println!("Best score={}", _score);
            self.processors = new_system.processors;
            self.leftmost_proc = new_system.leftmost_proc;
            self.rightmost_proc = new_system.rightmost_proc;
            self.start_times = new_system.start_times;
            break;
        }

        self.print_planning();
    }








        //     // println!(" Finding for task {}", task_id);
        //     let (proc, mut task_start) = self.find_best_proc(task_w, &parents);
        //     // Place data transmissions -- for each parent put transmission paths
        //     for (parent_task_id, link_w) in parents.iter() {
        //         let (parent_proc, parent_start, parent_w) =
        //             self.start_times.get(parent_task_id).unwrap().clone();
        //         if proc == parent_proc {
        //             continue; // on the same processor => nothing to transmit
        //         }
        //         // Hot to get from parent to proc
        //         let paths: Vec<(i32, i32)> = {
        //             let left_to_right = parent_proc < proc;
        //             let pairs = (if parent_proc < proc
        //                          {(parent_proc..proc)}
        //                     else {(proc..parent_proc)})
        //                 .map(|x| if left_to_right {(x, x+1)} else {(x+1, x)});
        //             if left_to_right {pairs.collect()} else {pairs.rev().collect()}
        //         };
        //         // Plan (put) paths
        //         let mut tick = (parent_start + parent_w) as usize;
        //         for (src, dst) in paths.into_iter() {
        //             loop { // Until we find an empty slot
        //                 let mut found = true;
        //                 for t in tick..(tick + *link_w as usize) {
        //                     let processor_src = self.processors.get(&src).unwrap();
        //                     if let Cell::Free = processor_src.get(t).unwrap_or(&Cell::Free) {
        //                         let processor_dst = self.processors.get(&dst).unwrap();
        //                         if let Cell::Free = processor_dst.get(t).unwrap_or(&Cell::Free) {
        //                             continue;
        //                         }
        //                     }
        //                     // At least for one _t_ couldn't find Free => failed for this _tick_
        //                     found = false;
        //                     break;
        //                 };
        //                 if found {break};
        //                 tick += 1;
        //             }
        //             // Actually place the transmission
        //             self.place(Cell::Snd(task_id), Place {proc: src, tick: tick as u32}, *link_w);
        //             self.place(Cell::Rcv(task_id), Place {proc: dst, tick: tick as u32}, *link_w);
        //
        //             tick += *link_w as usize;
        //         }
        //         // _tick_ now contains the tick when the data transmission from
        //         // current parent has ended. The actual start time is thus the max
        //         // of all such _tick_s.
        //         task_start = std::cmp::max(task_start, tick as u32);
        //     }
        //
        //     // Have received all data for this task => can start it
        //     self.place(Cell::Taken(task_id), Place {proc: proc, tick: task_start}, task_w);
        // }

    fn place(system: &mut System, cell: Cell, Place { proc, tick }: Place, amount: u32) {
        {
            let processor = system.processors.get_mut(&proc).unwrap();
            while (processor.len() as u32) < tick {
                // create cells up to _tick_
                processor.push(Cell::Free);
            }
            let mut start = tick;
            while processor.get(start as usize).unwrap_or(&Cell::Free) != &Cell::Free {
                start += 1; // find first empty spacce (or the end, which is empty as well)
            }

            for curr in start..(start + amount) {
                // let processor = self.processors.get_mut(&proc).unwrap();
                let cell = cell.clone();

                if curr >= (processor.len() as u32) {
                    processor.push(cell);
                } else {
                    processor[curr as usize] = cell;
                }
            }
        }

        if let Cell::Taken(task_id) = cell {
            system.start_times.insert(task_id, (proc, tick, amount));
        }

        // Add one processor to the left and one processor to the right, if needed
        if proc == system.leftmost_proc {
            system.leftmost_proc -= 1;
            system.processors.insert(system.leftmost_proc, Vec::new());
        }
        if proc == system.rightmost_proc {
            system.rightmost_proc += 1;
            system.processors.insert(system.rightmost_proc, Vec::new());
        }
    }

    fn find_consecutive_block(&self, proc: ProcId, w: Weight, s: StartTime) -> StartTime {
        let processor = self.processors.get(&proc).unwrap();
        if s as usize >= processor.len() {return s;}

        for i in (s as usize)..processor.len() {
            if let Cell::Free = processor.get(i).unwrap() {
                let mut all_good = true;
                for ii in (i + 1)..(i + w as usize) {
                    if let Cell::Free = processor.get(ii).unwrap_or(&Cell::Free) {
                        continue;
                    } else {
                        all_good = false;
                        break;
                    }
                }
                if all_good { return i as u32; }
            }
        }

        (processor.len() as u32)
    }

    fn find_best_proc(&self, w: Weight, parents: &Vec<(TaskId, Weight)>) -> (ProcId, StartTime) {
        let eval_proc = |proc_id: i32| -> u32 {
            let transmissions: Vec<(_,_)> = parents.iter()
                .map(|(task_id, w)| {
                    let (other_proc_id, other_start, other_w) =
                        self.start_times.get(task_id).expect("Parent not planned yet");
                    let other_finish = other_start + other_w;
                    let dst: u32 = (proc_id - other_proc_id).abs() as u32;
                    (dst * w, other_finish + dst * w)
                })
                .collect();
                // .sum();
            let (trans_sum, trans_last): (u32, u32) = (
                transmissions.iter().map(|(score, _)| score).sum(),
                *transmissions.iter().map(|(_, finish)| finish).max().unwrap_or(&0)
            );
            // println!("Trans {}", transmission_score);
            // self.find_consecutive_block(proc_id, w, start_time) + transmission_score * 1
            self.find_consecutive_block(proc_id, w, trans_last) + trans_sum
        };

        // Start = max(parents_end_time)
        let start: u32 = parents.iter()
            .map(|(task_id, _)| {
                let (_, parent_start_time, parent_w) = self.start_times.get(task_id).unwrap();
                parent_start_time + parent_w
            })
            .max()
            .unwrap();
        // println!("Start at {}", start);

        // Result = proc_index with best(min) score
        ((self.leftmost_proc..=self.rightmost_proc)
            .map(|proc| (proc, eval_proc(proc)))
            // .inspect(|(proc, score)| println!("Considering score={} for proc={}", score, proc))
            .min_by(|(_, score1), (_, score2)| score1.cmp(score2))
            // .map(|(proc, score)| {println!("  Settling for score={} for proc={}", score, proc); (proc, score)})
            .map(|(index, _)| index)
            .unwrap() as i32,
        start)
    }

    fn rmv_earliest(&mut self) -> Task {
        let max = self.unplanned_tasks.iter()
            .enumerate()
            .max_by_key(|(_, Task {imp, ..})| imp)
            .map(|(index, _)| index)
            .unwrap();
        let removed = self.unplanned_tasks.remove(max);
        self.planned_tasks.push(removed.clone());
        removed
    }
}

fn main() {
    // let (vert, links) = populate_random();
    // let tasks = tasks_from(vert, links);
    // print_tasks(&tasks);
    // System::new(tasks).plan();
    System::new(tasks_from(populate_vertices(), populate_links())).plan();
}

// fn populate_random() -> (Vec<Vertex>, Vec<Link>) {
//     let vertex_count = 8;
//     let min_per_layer = 1;
//     let max_per_layer = vertex_count / 2;
//     let min_vertex_weight = 3;
//     let max_vertex_weight = 8;
//     let seed = [6,4,4,4, 5,6,7,8, 9,10,11,12, 13,14,15,16];
//     let mut rng = SmallRng::from_seed(seed);
//
//     // Vertices
//     let mut done_vertices_count = 0;
//     let mut id: VertId = 0;
//     let mut layers = Vec::new();
//     while done_vertices_count < vertex_count {
//         let mut layer = Vec::new();
//         let count: u32 = {
//             let r = rng.gen_range(min_per_layer, max_per_layer + 1);
//             let left = vertex_count - done_vertices_count;
//             if left < r {left} else {r}
//         };
//         for _ in 0..count {
//             let weight: Weight = rng.gen_range(min_vertex_weight, max_vertex_weight + 1);
//             layer.push(Vertex {id: id, w: weight});
//             id += 1;
//         }
//         done_vertices_count += count;
//
//         layers.push(layer);
//     }
//     println!("{:#?}", layers);
//
//     // Links
//     let links_count = (vertex_count * (vertex_count - 1) / 2) / 3;
//     println!("Will create {} links", links_count);
//     let min_link_weight = 1;
//     let max_link_weight = 2;
//     let mut links = Vec::new();
//     // There must be at least one path with a Vertex in each layer.
//     // Gen that path. Since all vertices are random,
//     // can just take 0th at each layer at this stage
//     let belongs_to_chunk = |id: u32, chunks: &Vec<Vec<VertId>>| -> Option<usize> {
//         chunks.iter()
//             .enumerate()
//             .find_map(|(index, chunk)|
//                 if chunk.iter()
//                     .find(|&& x| x == id)
//                     .is_some() {Some(index)}
//                 else {None}
//             )
//     };
//     let mut done_links_count = 0;
//     let mut chunks = Vec::new();
//     let mut converged = false;
//     // let mut iii = 0;
//     while !converged || done_links_count < links_count {
//         // if !converged && iii < 200 { println!("Have chunks: {:#?}", chunks); iii += 1 }
//         let layer_src_index = rng.gen_range(0, layers.len() - 1);
//         let layer_dst_index = rng.gen_range(layer_src_index + 1, layers.len());
//         let layer_src = layers.get(layer_src_index).unwrap();
//         let layer_dst = layers.get(layer_dst_index).unwrap();
//         let src_index = layer_src.get(rng.gen_range(0, layer_src.len())).unwrap().id.clone();
//         let dst_index = layer_dst.get(rng.gen_range(0, layer_dst.len())).unwrap().id.clone();
//         // Retry if such link already exists
//         if links.iter()
//             .find(|&& Link {src, dst, ..}| (src == src_index) && (dst == dst_index))
//             .is_some() { continue; }
//         let weight: u32 = rng.gen_range(min_link_weight, max_link_weight + 1);
//         links.push(Link { src: src_index, dst: dst_index, w: weight });
//         done_links_count += 1;
//
//         // Have nothing to trach if have already converged
//         if converged { continue; }
//
//         let all_connected = |chunks: &Vec<Vec<u32>>| {
//             if chunks.len() == 1 {
//                 let chunk = chunks.get(0).unwrap();
//                 layers.iter().flatten()
//                     .map(|Vertex {id, ..}| chunk.iter().find(|&&x| &x == id).is_some())
//                     .all(|found| found)
//             } else {
//                 false
//             }
//         };
//         let chunk_src = belongs_to_chunk(src_index, &chunks);
//         let chunk_dst = belongs_to_chunk(dst_index, &chunks);
//         // println!("Have: {} and {}", src_index, dst_index);
//         if chunk_src.is_none() && chunk_dst.is_none() {
//             // Create new chunk
//             // println!("Creating new chunk");
//             chunks.push(vec![src_index, dst_index]);
//         } else if chunk_src.is_none() && chunk_dst.is_some() {
//             // Add src to dst_chunk
//             // println!("Adding src to dst");
//             chunks.get_mut(chunk_dst.unwrap()).unwrap().push(src_index);
//             if all_connected(&chunks) { converged = true; }
//         } else if chunk_src.is_some() && chunk_dst.is_none() {
//             // Add dst to src_chunk
//             // println!("Adding dst to src");
//             chunks.get_mut(chunk_src.unwrap()).unwrap().push(dst_index);
//             if all_connected(&chunks) { converged = true; }
//         } else { // both Some
//             let src = chunk_src.unwrap();
//             let dst = chunk_dst.unwrap();
//             // println!("{} , {}", src, dst);
//             if src != dst {
//                 // println!("Not same, merging");
//                 // Merge chunks
//                 let mut other = chunks.remove(dst);
//                 chunks.get_mut(if dst < src {src - 1} else {src}).unwrap().append(&mut other);
//
//                 if all_connected(&chunks) { converged = true; }
//             } else {
//                 // println!("Same, nothing");
//             }
//         }
//     }
//
//     println!("Done {} links total", done_links_count);
//
//     (layers.into_iter().flatten().collect(), links)
// }

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


            // let mut curr: usize = std::cmp::max(proc_src.len(), proc_dst.len()) - 1;
            // let start = loop {
            //     if curr < min_start as usize {break curr + 1}
            //     let src = proc_src.get(curr).unwrap_or(&Cell::Free);
            //     let dst = proc_dst.get(curr).unwrap_or(&Cell::Free);
            //
            //     curr -= 1;
                // let mut unwind_till_another = |id: TaskId, proc: &Vec<Cell>| {
                //     let (_, task_start, task_w) = start_times.get(&id).unwrap().clone();
                //     let imp_task = planned_tasks.iter()
                //         .find(|&&Task{id: task_id, ..}| task_id == id)
                //         .map(|Task{imp, ..}| imp)
                //         .unwrap().clone();
                    // if System::could_shift(task_start, curr as u32,
                    //                task_start + task_w, imp, imp_task) {
                    //     return curr;
                    // } else {
                    //     // Won't be able to shuft till the end of current task =>
                    //     // => find end and return it
                    //     while let Cell::Taken(task_id) =
                    //             proc.get(curr + 1).unwrap_or(&Cell::Free) {
                    //         // Another task starts at next
                    //         if task_id.clone() != id {break}
                    //         curr += 1;
                    //     }
                    //     return curr + 1; // next
                    // }
                // };
                // match (src, dst) {
                //     (Cell::Rcv(_), Cell::Rcv(_)) | (Cell::Rcv(_), Cell::Snd(_)) |
                //     (Cell::Snd(_), Cell::Rcv(_)) | (Cell::Snd(_), Cell::Snd(_)) |
                //     (Cell::Rcv(_), Cell::Free) | (Cell::Snd(_), Cell::Free) |
                //     (Cell::Free, Cell::Rcv(_)) | (Cell::Free, Cell::Snd(_)) => {
                //         break curr + 1;
                //     },
                //     (Cell::Rcv(_), Cell::Taken(id)) | (Cell::Snd(_), Cell::Taken(id)) => {
                //         break unwind_till_another(id.clone(), &proc_dst);
                //     },
                //     (Cell::Taken(id), Cell::Rcv(_)) | (Cell::Taken(id), Cell::Snd(_)) => {
                //         break unwind_till_another(id.clone(), &proc_src);
                //     },
                //     _ => {}
                // }
                // curr -= 1
            // };

            // (start as u32, None)
        // };
