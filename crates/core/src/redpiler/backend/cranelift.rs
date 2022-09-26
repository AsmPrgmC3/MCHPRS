use std::cell::UnsafeCell;
use std::collections::{HashMap, VecDeque};
use std::mem;
use std::pin::Pin;

use cranelift_jit::{JITBuilder, JITModule};
use log::warn;

use mchprs_blocks::block_entities::BlockEntity;
use mchprs_blocks::BlockPos;
use mchprs_world::{TickEntry, TickPriority};

use crate::blocks::Block;
use crate::plot::PlotWorld;
use crate::redpiler::backend::JITBackend;
use crate::redpiler::{block_powered_mut, bool_to_ss, CompileNode};
use crate::world::World;

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
enum CLTickPriority {
    Highest = 0,
    Higher = 1,
    High = 2,
    Normal = 3,
}

impl From<TickPriority> for CLTickPriority {
    fn from(priority: TickPriority) -> Self {
        match priority {
            TickPriority::Highest => CLTickPriority::Highest,
            TickPriority::Higher => CLTickPriority::Higher,
            TickPriority::High => CLTickPriority::High,
            TickPriority::Normal => CLTickPriority::Normal,
        }
    }
}

impl From<CLTickPriority> for TickPriority {
    fn from(priority: CLTickPriority) -> Self {
        match priority {
            CLTickPriority::Highest => TickPriority::Highest,
            CLTickPriority::Higher => TickPriority::Higher,
            CLTickPriority::High => TickPriority::High,
            CLTickPriority::Normal => TickPriority::Normal,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeIndex(usize);

type TickFunction = (
    unsafe extern "C" fn(*mut TickScheduler, usize, u8, *mut u8, *mut bool) -> (),
    usize,
);

pub struct CraneliftBackend {
    num_nodes: usize,
    power_data: RawSlice<u8>,
    locked_data: RawSlice<bool>,
    scheduled_data: RawSlice<bool>,
    changed_data: RawSlice<bool>,
    tick_functions: Box<[TickFunction]>,
    module: JITModule,
    nodes: Box<[CompileNode]>,
    blocks: Box<[(BlockPos, Block)]>,
    is_io_block: Box<[bool]>,
    pos_map: HashMap<BlockPos, NodeIndex>,
    scheduler: Pin<Box<UnsafeCell<TickScheduler>>>,
}

impl Default for CraneliftBackend {
    fn default() -> Self {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names());
        let module = JITModule::new(builder.unwrap());
        Self {
            num_nodes: 0,
            power_data: Default::default(),
            locked_data: Default::default(),
            scheduled_data: Default::default(),
            changed_data: Default::default(),
            tick_functions: Box::new([]),
            module,
            nodes: Box::new([]),
            blocks: Box::new([]),
            is_io_block: Box::new([]),
            pos_map: Default::default(),
            scheduler: Box::pin(UnsafeCell::new(Default::default())),
        }
    }
}

impl CraneliftBackend {
    fn schedule_tick(&self, tick_fn: TickFunction, delay: usize, priority: CLTickPriority) {
        let scheduler = unsafe { &mut *self.scheduler.get() };
        scheduler.schedule_tick(tick_fn, delay, priority);
    }
}

impl JITBackend for CraneliftBackend {
    fn compile(&mut self, nodes: Vec<CompileNode>, ticks: Vec<TickEntry>) {
        self.num_nodes = nodes.len();
        self.blocks = nodes.iter().map(|node| (node.pos, node.state)).collect();
        self.nodes = nodes.into_boxed_slice();
        for i in 0..self.nodes.len() {
            self.pos_map.insert(self.blocks[i].0, NodeIndex(i));
        }
        let mut power_data = vec![];
        let mut locked_data = vec![];
        let mut scheduled_data = vec![];
        let mut changed_data = vec![];
        let mut is_io_block = vec![];
        for node in &*self.nodes {
            scheduled_data.push(false);

            power_data.push(node.output_power());

            let locked = match node.state {
                Block::RedstoneRepeater { repeater } => repeater.locked,
                _ => false,
            };
            locked_data.push(locked);

            changed_data.push(false);

            let is_io = matches!(
                node.state,
                Block::RedstoneLamp { .. }
                    | Block::Lever { .. }
                    | Block::StoneButton { .. }
                    | Block::StonePressurePlate { .. }
                    | Block::IronTrapdoor { .. }
            );
            is_io_block.push(is_io);
        }

        let scheduler = unsafe { &mut *self.scheduler.get() };
        for i in 0..self.num_nodes {
            let node_index = NodeIndex(i);
            let node = &self.nodes[node_index.0];
            let tick_func: TickFunction;
            if matches!(node.state, Block::RedstoneWire { .. }) {
                tick_func = (wire_tick, node_index.0);
            } else {
                tick_func = (generic_tick, node_index.0);
            }
            scheduler.schedule_tick(tick_func, 1, CLTickPriority::Normal);
        }
        // for tick_entry in ticks {
        //     let node_index = self.pos_map[&tick_entry.pos];
        //     let node = &self.nodes[node_index.0];
        //     let tick_func: TickFunction;
        //     if matches!(node.state, Block::RedstoneWire { .. }) {
        //         tick_func = (wire_tick, node_index.0);
        //     } else {
        //         tick_func = (generic_tick, node_index.0);
        //     }
        //     scheduler.schedule_tick(
        //         tick_func,
        //         tick_entry.ticks_left as usize,
        //         tick_entry.tick_priority.into(),
        //     );
        // }

        self.changed_data = changed_data.into();
        self.is_io_block = is_io_block.into();
        self.locked_data = locked_data.into();
        self.power_data = power_data.into();
        self.scheduled_data = scheduled_data.into();
    }

    fn tick(&mut self, _plot: &mut PlotWorld) {
        let mut queues = unsafe { &mut *self.scheduler.get() }.queues_this_tick();
        for (tick_func, index) in queues.drain_iter() {
            unsafe {
                tick_func(
                    self.scheduler.get(),
                    index,
                    0,
                    self.power_data.ptr as _,
                    self.changed_data.ptr as _,
                )
            };
        }
        unsafe { &mut *self.scheduler.get() }.end_tick(queues);
    }

    fn on_use_block(&mut self, _plot: &mut PlotWorld, pos: BlockPos) {
        let node_index = self.pos_map[&pos];
        let node = &self.nodes[node_index.0];
        match node.state {
            Block::StoneButton { .. } => {
                unsafe {
                    self.tick_functions[node_index.0].0(
                        self.scheduler.get(),
                        node_index.0,
                        1,
                        self.power_data.ptr as _,
                        self.changed_data.ptr as _,
                    )
                };
                self.schedule_tick(
                    self.tick_functions[node_index.0],
                    10,
                    CLTickPriority::Normal,
                );
            }
            Block::Lever { .. } => {
                unsafe {
                    self.tick_functions[node_index.0].0(
                        self.scheduler.get(),
                        node_index.0,
                        0,
                        self.power_data.ptr as _,
                        self.changed_data.ptr as _,
                    )
                };
            }
            _ => warn!("Tried to use a {:?} redpiler node", node.state),
        }
    }

    fn set_pressure_plate(&mut self, _plot: &mut PlotWorld, pos: BlockPos, powered: bool) {
        let node_id = self.pos_map[&pos];
        let node = &self.nodes[node_id.0];
        match node.state {
            Block::StonePressurePlate { .. } => {
                unsafe {
                    self.tick_functions[node_id.0].0(
                        self.scheduler.get(),
                        node_id.0,
                        powered as _,
                        self.power_data.ptr as _,
                        self.changed_data.ptr as _,
                    )
                };
            }
            _ => warn!("Tried to set pressure plate state for a {:?}", node.state),
        }
    }

    fn flush(&mut self, plot: &mut PlotWorld, io_only: bool) {
        let changed_data = unsafe { self.changed_data.as_mut_slice() };
        let power_data = unsafe { self.power_data.as_slice() };

        for node_index in 0..self.num_nodes {
            let changed = &mut changed_data[node_index];
            let (pos, block) = &mut self.blocks[node_index];
            if *changed && (!io_only || self.is_io_block[node_index]) {
                let power = power_data[node_index];
                if let Some(powered) = block_powered_mut(block) {
                    *powered = power > 0;
                }
                if let Block::RedstoneWire { wire, .. } = block {
                    wire.power = power;
                }
                plot.set_block(*pos, *block);
            }
            *changed = false;
        }
    }

    fn reset(&mut self, plot: &mut PlotWorld, io_only: bool) {
        unsafe { &mut *self.scheduler.get() }.reset(plot, &self.blocks);

        let power_data = unsafe { self.power_data.as_slice() };

        for node_index in 0..self.num_nodes {
            let (pos, block) = self.blocks[node_index];

            if matches!(block, Block::RedstoneComparator { .. }) {
                let block_entity = BlockEntity::Comparator {
                    output_strength: power_data[node_index],
                };
                plot.set_block_entity(pos, block_entity);
            }

            if io_only && !self.is_io_block[node_index] {
                plot.set_block(pos, block);
            }
        }
    }
}

#[derive(Default, Clone)]
struct Queues([Vec<TickFunction>; TickScheduler::NUM_PRIORITIES]);

impl Queues {
    fn drain_iter(&mut self) -> impl Iterator<Item = TickFunction> + '_ {
        let [q0, q1, q2, q3] = &mut self.0;
        let [q0, q1, q2, q3] = [q0, q1, q2, q3].map(|q| q.drain(..));
        q0.chain(q1).chain(q2).chain(q3)
    }
}

#[derive(Default)]
struct TickScheduler {
    queues_deque: VecDeque<Queues>,
}

impl TickScheduler {
    const NUM_PRIORITIES: usize = 4;

    fn reset(&mut self, plot: &mut PlotWorld, blocks: &[(BlockPos, Block)]) {
        for (delay, queues) in self.queues_deque.iter().enumerate() {
            for (entries, priority) in queues.0.iter().zip(Self::priorities()) {
                for node in entries {
                    let pos = blocks[node.1].0;
                    plot.schedule_tick(pos, delay as u32, priority.into());
                }
            }
        }
        self.queues_deque.clear();
    }

    fn schedule_tick(&mut self, function: TickFunction, delay: usize, priority: CLTickPriority) {
        if delay >= self.queues_deque.len() {
            self.queues_deque.resize(delay + 1, Default::default());
        }

        self.queues_deque[delay].0[Self::priority_index(priority)].push(function);
    }

    fn queues_this_tick(&mut self) -> Queues {
        if self.queues_deque.len() == 0 {
            self.queues_deque.push_back(Default::default());
        }
        mem::take(&mut self.queues_deque[0])
    }

    fn end_tick(&mut self, mut queues: Queues) {
        self.queues_deque.pop_front();

        for queue in &mut queues.0 {
            queue.clear();
        }
        self.queues_deque.push_back(queues);
    }

    fn priorities() -> [CLTickPriority; Self::NUM_PRIORITIES] {
        [
            CLTickPriority::Highest,
            CLTickPriority::Higher,
            CLTickPriority::High,
            CLTickPriority::Normal,
        ]
    }

    fn priority_index(priority: CLTickPriority) -> usize {
        match priority {
            CLTickPriority::Highest => 0,
            CLTickPriority::Higher => 1,
            CLTickPriority::High => 2,
            CLTickPriority::Normal => 3,
        }
    }
}

struct RawSlice<T> {
    ptr: *mut [T],
    length: usize,
}

impl<T> Default for RawSlice<T> {
    fn default() -> Self {
        vec![].into_boxed_slice().into()
    }
}

impl<T> From<Box<[T]>> for RawSlice<T> {
    fn from(slice: Box<[T]>) -> Self {
        let length = slice.len();
        let ptr = Box::into_raw(slice);
        Self { ptr, length }
    }
}

impl<T> From<Vec<T>> for RawSlice<T> {
    fn from(vec: Vec<T>) -> Self {
        vec.into_boxed_slice().into()
    }
}

impl<T> RawSlice<T> {
    unsafe fn into_slice(self) -> Box<[T]> {
        Box::from_raw(self.ptr)
    }

    unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr as *const T, self.length)
    }

    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr as *mut T, self.length)
    }
}

unsafe extern "C" fn schedule_tick(
    scheduler: *mut TickScheduler,
    tick_func: TickFunction,
    delay: usize,
    priority: CLTickPriority,
) {
    (&mut *scheduler).schedule_tick(tick_func, delay, priority);
}

unsafe extern "C" fn wire_tick(
    scheduler: *mut TickScheduler,
    index: usize,
    _data: u8,
    power_data: *mut u8,
    changed_data: *mut bool,
) {
    let power = power_data.offset(index as _).read();
    let new_power = (power + 1) % 15;
    power_data.offset(index as _).write(new_power);
    changed_data.offset(index as _).write(true);
    schedule_tick(scheduler, (wire_tick, index), 1, CLTickPriority::Normal);
}

unsafe extern "C" fn generic_tick(
    scheduler: *mut TickScheduler,
    index: usize,
    _data: u8,
    power_data: *mut u8,
    changed_data: *mut bool,
) {
    let current = power_data.offset(index as _).read() > 0;
    let new = bool_to_ss(!current);
    power_data.offset(index as _).write(new);
    changed_data.offset(index as _).write(true);
    schedule_tick(scheduler, (generic_tick, index), 1, CLTickPriority::High);
}
