use std::collections::HashMap;
use std::fmt;

// #[derive(PartialEq, Eq, Ord, PartialOrd)]
struct VertId(u32);

struct Vertex {
    id: VertId,
    w: Weight,
}

struct Link {
    src: VertId,
    dst: VertId,
    w: Weight,
}

type ProcId = i32;

type Tick = u32;
type TaskId = u32;
type StartTime = u32;
type Weight = u32;
type Importance = u32;

#[derive(Clone)]
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
    for Vertex {id: VertId(v_index), w} in vertices.into_iter() {
        if v_index >= length {
            panic!("Using VertId beyond upper bound.")
        }
        if v_index != (tasks.len() as u32) {
            panic!("Pushing for wrong index");
        }

        tasks.push(Task {
            id: v_index,
            w: w,
            imp: 0,
            children: Vec::new(),
            parents: Vec::new(),
        });
    }

    // Add links
    for Link {src: VertId(src), dst: VertId(dst), w} in links.into_iter() {
        if src >= (tasks.len() as u32) || dst >= (tasks.len() as u32) {
            panic!("Link from or to out-of-bounds Vertex");
        }
        tasks[src as usize].children.push((dst, w.clone()));
        tasks[dst as usize].parents.push((src, w));
    }

    // Set importance
    // Assumes that at each iteration there will be at least 1 Task whose all
    // children have their Importance set. Utilizes exactly 1 such Task at each iteration.
    let mut set_amount = 0;
    while set_amount < tasks.len() {
        let mut res = (0, 0);
        'tasks_loop: for (index, task) in tasks.iter().enumerate() {
            if task.imp == 0 { // not set (not computed) yet
                let mut sum = task.w.clone();
                for (t_id, _) in task.children.iter() {
                    let imp = tasks[*t_id as usize].imp;
                    if imp == 0 {
                        continue 'tasks_loop;
                    } else {
                        sum += imp;
                    }
                }
                res = (index, sum);
                break 'tasks_loop; // done for this loop
            } else {
                continue 'tasks_loop; // nothing to do here
            }
        }
        // Assumes to_set has been set
        let (index, imp) = res;
        tasks[index].imp = imp;
        set_amount += 1;
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

struct System {
    processors: HashMap<ProcId, Vec<Cell>>,
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
            unplanned_tasks: tasks,
            start_times: HashMap::new(),
        }
    }

    fn plan(&mut self) {
        while self.unplanned_tasks.len() > 0 {
            let Task {id: task_id, w, parents, ..} = self.rmv_earliest();
            // No parents => can start immediately => into a new processor
            if parents.is_empty() {
                let place = Place {proc: self.rightmost_proc.clone(), tick: 0};
                self.place(Cell::Taken(task_id), place, w);
                continue;
            }

            let proc = self.find_best_proc(&parents);
            // Place data transmissions -- for each parent put transmission paths
            // let mut tick; // there is at leats one parentA => will be initialized for sure
            for (parent_task_id, link_w) in parents.iter() {
                let (parent_proc, parent_start, parent_w) =
                    self.start_times.get(parent_task_id).unwrap().clone();
                if proc == parent_proc {
                    continue; // Nothing to transmit
                }
                // Hot to get from parent to proc
                let paths: Vec<(i32, i32)> = {
                    let left_to_right = parent_proc < proc;
                    let pairs = (if parent_proc < proc
                                 {(parent_proc..proc)}
                            else {(proc..parent_proc)})
                        .map(|x| if left_to_right {(x, x+1)} else {(x+1, x)});
                    if left_to_right {pairs.collect()} else {pairs.rev().collect()}
                };
                // Plan (put) paths
                let mut tick = (parent_start + parent_w) as usize;
                for (src, dst) in paths.into_iter() {
                    loop { // Until we find an empty slot
                        let mut found = true;
                        for t in tick..(tick + *link_w as usize) {
                            let processor_src = self.processors.get(&src).unwrap();
                            if let Cell::Free = processor_src.get(t).unwrap_or(&Cell::Free) {
                                let processor_dst = self.processors.get(&dst).unwrap();
                                if let Cell::Free = processor_dst.get(t).unwrap_or(&Cell::Free) {
                                    continue;
                                }
                            }
                            // At least for one _t_ couldn't find Free => failed for this _tick_
                            found = false;
                            break;
                        };
                        if found {break};
                        tick += 1;
                    }
                    // Enlarge each Vec of Cells if needed
                    {
                        let mut enlarge = |proc_id| {
                            let processor = self.processors.get_mut(proc_id).unwrap();
                            while processor.len() < (tick + *link_w as usize) {
                                // println!("Pushing for {} to {}", proc_id, processor.len());
                                processor.push(Cell::Free);
                            }
                        };
                        enlarge(&src);
                        enlarge(&dst);
                    }
                    // Actually place the transmission
                    self.place(Cell::Snd(task_id), Place {proc: src, tick: tick as u32}, *link_w);
                    self.place(Cell::Rcv(task_id), Place {proc: dst, tick: tick as u32}, *link_w);

                    tick += *link_w as usize;
                }
                // break;
            }

            // Have received all data for this task => can start it
            let start = self.processors.get(&proc).unwrap().len() as u32;
            self.place(Cell::Taken(task_id), Place {proc: proc, tick: start}, w);
            break;
        }

        self.print_planning();
    }

    fn place(&mut self, cell: Cell, Place {proc, tick}: Place, amount: u32) {
        for curr in tick..(tick + amount) {
            let processor = self.processors.get_mut(&proc).unwrap();
            let cell = cell.clone();

            if curr >= (processor.len() as u32) {
                processor.push(cell);
            } else {
                processor[curr as usize] = cell;
            }
        }

        if let Cell::Taken(task_id) = cell {
            self.start_times.insert(task_id, (proc, tick, amount));
        }

        // Add one processor to the left and one processor to the right, if needed
        if proc == self.leftmost_proc {
            self.leftmost_proc -= 1;
            self.processors.insert(self.leftmost_proc, Vec::new());
        }
        if proc == self.rightmost_proc {
            self.rightmost_proc += 1;
            self.processors.insert(self.rightmost_proc, Vec::new());
        }
    }

    fn find_best_proc(&self, parents: &Vec<(TaskId, Weight)>) -> ProcId {
        let eval_proc = |proc_id: i32| -> u32 {
            parents.iter()
                .map(|(task_id, w)| {
                    let other_proc_id = self.start_times.get(task_id).expect("Parent not planned yet").0;
                    let dst: u32 = (proc_id - other_proc_id).abs() as u32;
                    dst * w
                })
                .sum()
        };

        (self.leftmost_proc..=self.rightmost_proc)
            .map(|proc| (proc, eval_proc(proc)))
            // .inspect(|(proc, score)| println!("Considering score={} for proc={}", score, proc))
            .min_by(|(_, score1), (_, score2)| score1.cmp(score2))
            .map(|(index, _)| index)
            .unwrap() as i32
    }

    fn rmv_earliest(&mut self) -> Task {
        // TODO
        self.unplanned_tasks.remove(0)
    }
}

fn main() {
    let mut system = System::new(tasks_from(populate_vertices(), populate_links()));
    system.plan();
}

// fn cumulative_weight(vertices: &Vec<Vertex>) -> u32 {
//     vertices.iter()
//         .map(|Vertex{w: Weight(w), ..}| w)
//         .sum()
// }

fn populate_vertices() -> Vec<Vertex> {
    vec![
        Vertex {
            id: VertId(0),
            w: 3,
        },
        Vertex {
            id: VertId(1),
            w: 4,
        },
        Vertex {
            id: VertId(2),
            w: 5,
        },
        Vertex {
            id: VertId(3),
            w: 3,
        },
        Vertex {
            id: VertId(4),
            w: 3,
        },
        Vertex {
            id: VertId(5),
            w: 2,
        },
        Vertex {
            id: VertId(6),
            w: 4,
        },
    ]
}

fn populate_links() -> Vec<Link> {
    vec![
        Link {
            src: VertId(0),
            dst: VertId(3),
            w: 1,
        },
        Link {
            src: VertId(0),
            dst: VertId(2),
            w: 2,
        },
        Link {
            src: VertId(1),
            dst: VertId(2),
            w: 1,
        },
        Link {
            src: VertId(1),
            dst: VertId(6),
            w: 2,
        },
        Link {
            src: VertId(3),
            dst: VertId(4),
            w: 1,
        },
        Link {
            src: VertId(3),
            dst: VertId(5),
            w: 2,
        },
        Link {
            src: VertId(2),
            dst: VertId(5),
            w: 1,
        },
    ]
}
