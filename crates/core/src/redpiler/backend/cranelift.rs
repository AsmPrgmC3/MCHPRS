use std::cell::UnsafeCell;
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::{array, mem};

use cranelift::codegen;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::prelude::{types, AbiParam, Block as CLBlock, InstBuilder, MemFlags, Type, Value};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use itertools::Itertools;
use log::warn;

use mchprs_blocks::block_entities::BlockEntity;
use mchprs_blocks::BlockPos;
use mchprs_world::{TickEntry, TickPriority};

use crate::blocks::{Block, RedstoneRepeater};
use crate::plot::PlotWorld;
use crate::redpiler::backend::JITBackend;
use crate::redpiler::{block_powered_mut, CompileNode, NodeId};
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

type TickFunction = unsafe extern "C" fn(*mut TickScheduler, u8) -> ();

pub struct CraneliftBackend {
    num_nodes: usize,
    program: Program,
    nodes: Box<[CompileNode]>,
    blocks: Box<[(BlockPos, Block)]>,
    is_io_block: Box<[bool]>,
    pos_map: HashMap<BlockPos, NodeIndex>,
}

impl Default for CraneliftBackend {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            program: Default::default(),
            nodes: Box::new([]),
            blocks: Box::new([]),
            is_io_block: Box::new([]),
            pos_map: Default::default(),
        }
    }
}

impl CraneliftBackend {
    fn schedule_tick(&self, node_id: NodeId, delay: usize, priority: CLTickPriority) {
        let scheduler = unsafe { &mut *self.program.scheduler.get() };
        scheduler.schedule_tick(node_id, delay, priority);
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
        let mut is_io_block = vec![];
        for node in &*self.nodes {
            power_data.push(node.output_power());

            let locked = match node.state {
                Block::RedstoneRepeater { repeater } => repeater.locked,
                _ => false,
            };
            locked_data.push(locked);

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

        self.is_io_block = is_io_block.into();

        self.program = Program::compile(&self.nodes, power_data, locked_data);

        let scheduler = unsafe { &mut *self.program.scheduler.get() };
        for i in 0..self.num_nodes {
            scheduler.schedule_tick(i, 1, CLTickPriority::Normal);
        }
    }

    fn tick(&mut self, _plot: &mut PlotWorld) {
        let mut queues = unsafe { &mut *self.program.scheduler.get() }.queues_this_tick();
        for node_id in queues.drain_iter() {
            let tick_func = self.program.tick_functions[node_id];
            unsafe { tick_func(self.program.scheduler.get(), 0) };
        }
        unsafe { &mut *self.program.scheduler.get() }.end_tick(queues);
    }

    fn on_use_block(&mut self, _plot: &mut PlotWorld, pos: BlockPos) {
        let node_index = self.pos_map[&pos];
        let node = &self.nodes[node_index.0];
        match node.state {
            Block::StoneButton { .. } => {
                unsafe {
                    self.program.tick_functions[node_index.0](self.program.scheduler.get(), 1)
                };
                self.schedule_tick(node_index.0, 10, CLTickPriority::Normal);
            }
            Block::Lever { .. } => {
                unsafe {
                    self.program.tick_functions[node_index.0](self.program.scheduler.get(), 0)
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
                    self.program.tick_functions[node_id.0](
                        self.program.scheduler.get(),
                        powered as _,
                    )
                };
            }
            _ => warn!("Tried to set pressure plate state for a {:?}", node.state),
        }
    }

    fn flush(&mut self, plot: &mut PlotWorld, io_only: bool) {
        let changed_data = unsafe { self.program.changed_data.as_mut_slice() };
        let power_data = unsafe { self.program.power_data.as_slice() };

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
        unsafe { &mut *self.program.scheduler.get() }.reset(plot, &self.blocks);

        let power_data = unsafe { self.program.power_data.as_slice() };

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

struct Program {
    power_data: RawSlice<u8>,
    locked_data: RawSlice<bool>,
    scheduled_data: RawSlice<bool>,
    changed_data: RawSlice<bool>,
    scheduler: Pin<Box<UnsafeCell<TickScheduler>>>,
    tick_functions: Box<[TickFunction]>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            power_data: Default::default(),
            locked_data: Default::default(),
            scheduled_data: Default::default(),
            changed_data: Default::default(),
            scheduler: Box::pin(Default::default()),
            tick_functions: Box::new([]),
        }
    }
}

impl Program {
    fn compile(nodes: &[CompileNode], power_data: Vec<u8>, locked_data: Vec<bool>) -> Self {
        let num_nodes = nodes.len();
        assert_eq!(num_nodes, power_data.len());
        assert_eq!(num_nodes, locked_data.len());

        let mut program = Program {
            power_data: power_data.into(),
            locked_data: locked_data.into(),
            scheduled_data: vec![false; num_nodes].into(),
            changed_data: vec![false; num_nodes].into(),
            scheduler: Box::pin(Default::default()),
            tick_functions: Box::new([]),
        };

        let mut builder = JITBuilder::new(default_libcall_names()).unwrap();
        let mut module = JITModule::new(builder);

        let mut context = JitContext {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            ptr_type: module.target_config().pointer_type(),
            globals: JitGlobals {
                jit_schedule_tick: jit_schedule_tick as _,
                power_data: program.power_data.ptr as *mut u8 as _,
                locked_data: program.locked_data.ptr as *mut bool as _,
                scheduled_data: program.scheduled_data.ptr as *mut bool as _,
                changed_data: program.changed_data.ptr as *mut bool as _,
            },
            module,
        };

        let funcs = nodes
            .iter()
            .enumerate()
            .map(|(node_id, node)| context.compile_node(node, node_id))
            .collect_vec();

        let mut module = context.module;

        module.finalize_definitions();

        let funcs = funcs
            .into_iter()
            .map(|func_id| {
                let addr = module.get_finalized_function(func_id);
                unsafe { mem::transmute(addr) }
            })
            .collect_vec();

        program.tick_functions = funcs.into_boxed_slice();

        program
    }
}

#[derive(Copy, Clone)]
struct JitGlobals {
    jit_schedule_tick: usize,
    power_data: usize,
    locked_data: usize,
    scheduled_data: usize,
    changed_data: usize,
}

struct JitContext {
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    ptr_type: Type,
    globals: JitGlobals,
    module: JITModule,
}

impl JitContext {
    fn compile_node(&mut self, node: &CompileNode, node_id: NodeId) -> FuncId {
        self.ctx
            .func
            .signature
            .params
            .push(AbiParam::new(self.ptr_type));
        self.ctx
            .func
            .signature
            .params
            .push(AbiParam::new(types::I8));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let mut translator = FunctionTranslator {
            ptr_type: self.ptr_type,
            module: &mut self.module,
            scheduler: builder.block_params(entry_block)[0],
            argument: builder.block_params(entry_block)[1],
            builder,
            globals: self.globals,
            variable_index: 0,
        };

        translator.translate_node(node, node_id);

        translator.builder.ins().return_(&[]);

        translator.builder.finalize();

        let func_id = self
            .module
            .declare_function(
                &format!("tick_{node_id}"),
                Linkage::Export,
                &self.ctx.func.signature,
            )
            .unwrap();
        self.module.define_function(func_id, &mut self.ctx).unwrap();

        self.module.clear_context(&mut self.ctx);

        func_id
    }
}

struct FunctionTranslator<'a> {
    ptr_type: Type,
    module: &'a mut JITModule,
    scheduler: Value,
    argument: Value,
    builder: FunctionBuilder<'a>,
    globals: JitGlobals,
    variable_index: usize,
}

impl<'a> FunctionTranslator<'a> {
    fn translate_node(&mut self, node: &CompileNode, node_id: NodeId) {
        let false_value = self.false_value();
        self.store_buffer(self.globals.scheduled_data, node_id, false_value);

        match &node.state {
            Block::RedstoneRepeater { repeater } => {
                self.translate_repeater(node, repeater, node_id)
            }
            Block::RedstoneWire { .. } => self.translate_wire(node, node_id),

            _ => {}
        }
    }

    fn translate_repeater(
        &mut self,
        node: &CompileNode,
        repeater: &RedstoneRepeater,
        node_id: usize,
    ) {
        let power = self.load_buffer(self.globals.power_data, node_id);

        let [turn_on_block, turn_off_block, end_block] = self.create_blocks();

        self.builder.ins().brz(power, turn_on_block, &[]);
        self.builder.ins().jump(turn_off_block, &[]);

        self.switch_seal_block(turn_on_block);
        let one_value = self.one_value();
        self.builder.ins().jump(end_block, &[one_value]);

        self.switch_seal_block(turn_off_block);
        let zero_value = self.zero_value();
        self.builder.ins().jump(end_block, &[zero_value]);

        self.switch_seal_block(end_block);
        self.builder.append_block_param(end_block, types::I8);
        let power_value = self.builder.block_params(end_block)[0];
        self.set_power(node_id, power_value);

        self.schedule_tick(node_id, 1, CLTickPriority::High);
    }

    fn translate_wire(&mut self, node: &CompileNode, node_id: usize) {
        let power = self.load_buffer(self.globals.power_data, node_id);
        let one_value = self.one_value();
        let power = self.builder.ins().iadd(power, one_value);
        let _15 = self.builder.ins().iconst(types::I8, 15);
        let power = self.builder.ins().urem(power, _15);
        self.set_power(node_id, power);

        self.schedule_tick(node_id, 1, CLTickPriority::Normal);
    }

    fn create_blocks<const COUNT: usize>(&mut self) -> [CLBlock; COUNT] {
        array::from_fn(|_| self.builder.create_block())
    }

    fn switch_seal_block(&mut self, block: CLBlock) {
        self.builder.switch_to_block(block);
        self.builder.seal_block(block);
    }

    fn load_buffer(&mut self, buffer: usize, offset: usize) -> Value {
        let addr = self
            .builder
            .ins()
            .iconst(self.ptr_type, (buffer + offset) as i64);
        self.builder
            .ins()
            .load(types::I8, MemFlags::trusted(), addr, 0)
    }

    fn store_buffer(&mut self, buffer: usize, offset: usize, value: Value) {
        let addr = self
            .builder
            .ins()
            .iconst(self.ptr_type, (buffer + offset) as i64);
        self.builder
            .ins()
            .store(MemFlags::trusted(), value, addr, 0);
    }

    fn set_power(&mut self, node_id: usize, power: Value) {
        self.store_buffer(self.globals.power_data, node_id, power);
        let true_value = self.true_value();
        self.store_buffer(self.globals.changed_data, node_id, true_value);
    }

    fn schedule_tick(&mut self, node_id: NodeId, delay: usize, priority: CLTickPriority) {
        let mut sig = self.module.make_signature();
        sig.params = vec![
            AbiParam::new(self.ptr_type),
            AbiParam::new(self.ptr_type),
            AbiParam::new(self.ptr_type),
            AbiParam::new(types::I8),
        ];
        let sig = self.builder.import_signature(sig);

        let jit_schedule_tick_value = self
            .builder
            .ins()
            .iconst(self.ptr_type, self.globals.jit_schedule_tick as i64);
        let node_id_value = self.builder.ins().iconst(self.ptr_type, node_id as i64);
        let delay_value = self.builder.ins().iconst(self.ptr_type, delay as i64);
        let priority_value = self.builder.ins().iconst(types::I8, priority as i64);
        self.builder.ins().call_indirect(
            sig,
            jit_schedule_tick_value,
            &[self.scheduler, node_id_value, delay_value, priority_value],
        );

        let true_value = self.true_value();
        self.store_buffer(self.globals.scheduled_data, node_id, true_value);
    }

    fn true_value(&mut self) -> Value {
        self.builder.ins().iconst(types::I8, 1)
    }

    fn false_value(&mut self) -> Value {
        self.builder.ins().iconst(types::I8, 0)
    }

    fn zero_value(&mut self) -> Value {
        self.builder.ins().iconst(types::I8, 0)
    }

    fn one_value(&mut self) -> Value {
        self.builder.ins().iconst(types::I8, 1)
    }
}

#[derive(Default, Clone)]
struct Queues([Vec<NodeId>; TickScheduler::NUM_PRIORITIES]);

impl Queues {
    fn drain_iter(&mut self) -> impl Iterator<Item = NodeId> + '_ {
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
                    let pos = blocks[*node].0;
                    plot.schedule_tick(pos, delay as u32, priority.into());
                }
            }
        }
        self.queues_deque.clear();
    }

    fn schedule_tick(&mut self, node: NodeId, delay: usize, priority: CLTickPriority) {
        if delay >= self.queues_deque.len() {
            self.queues_deque.resize(delay + 1, Default::default());
        }

        self.queues_deque[delay].0[Self::priority_index(priority)].push(node);
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

unsafe extern "C" fn jit_schedule_tick(
    scheduler: *mut TickScheduler,
    node_id: NodeId,
    delay: usize,
    priority: CLTickPriority,
) {
    (&mut *scheduler).schedule_tick(node_id, delay, priority);
}
