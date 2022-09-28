use std::cell::UnsafeCell;
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::{array, mem};

use cranelift::codegen;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::prelude::{
    types, AbiParam, Block as CLBlock, InstBuilder, IntCC, MemFlags, Type, Value,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use itertools::Itertools;
use log::warn;

use mchprs_blocks::block_entities::BlockEntity;
use mchprs_blocks::BlockPos;
use mchprs_world::{TickEntry, TickPriority};

use crate::blocks::{Block, ComparatorMode, RedstoneComparator, RedstoneRepeater};
use crate::plot::PlotWorld;
use crate::redpiler::backend::JITBackend;
use crate::redpiler::{block_powered_mut, bool_to_ss, CompileNode, Link, LinkType, NodeId};
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
            let power = match &node.state {
                Block::RedstoneWire { wire } => wire.power,
                Block::Lever { lever } => bool_to_ss(lever.powered),
                Block::StoneButton { button } => bool_to_ss(button.powered),
                Block::RedstoneTorch { lit } => bool_to_ss(*lit),
                Block::RedstoneWallTorch { lit, .. } => bool_to_ss(*lit),
                Block::RedstoneRepeater { repeater } => bool_to_ss(repeater.powered),
                Block::RedstoneLamp { lit } => bool_to_ss(*lit),
                Block::RedstoneComparator { .. } => node.comparator_output,
                Block::RedstoneBlock {} => 15,
                Block::StonePressurePlate { powered } => bool_to_ss(*powered),
                Block::IronTrapdoor { powered, .. } => bool_to_ss(*powered),
                s if s.has_comparator_override() => node.comparator_output,
                _ => 0,
            };
            power_data.push(power);

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

        for tick_entry in ticks {
            let node_index = self.pos_map[&tick_entry.pos];
            self.schedule_tick(
                node_index.0,
                tick_entry.ticks_left as usize,
                tick_entry.tick_priority.into(),
            );
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
        let locked_data = unsafe { self.program.locked_data.as_slice() };

        for node_index in 0..self.num_nodes {
            let changed = &mut changed_data[node_index];
            let (pos, block) = &mut self.blocks[node_index];
            if *changed && (!io_only || self.is_io_block[node_index]) {
                let power = power_data[node_index];
                if let Some(powered) = block_powered_mut(block) {
                    *powered = power > 0;
                }

                match block {
                    Block::RedstoneWire { wire, .. } => wire.power = power,
                    Block::RedstoneRepeater { repeater, .. } => {
                        repeater.locked = locked_data[node_index]
                    }
                    _ => (),
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

        let builder = JITBuilder::new(default_libcall_names()).unwrap();
        let module = JITModule::new(builder);

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
            .map(|(node_id, node)| context.compile_node(nodes, node, node_id))
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
    fn compile_node(
        &mut self,
        nodes: &[CompileNode],
        node: &CompileNode,
        node_id: NodeId,
    ) -> FuncId {
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
            nodes,
            node,
            node_id,

            ptr_type: self.ptr_type,
            module: &mut self.module,
            scheduler: builder.block_params(entry_block)[0],
            argument: builder.block_params(entry_block)[1],
            builder,
            globals: self.globals,
        };

        translator.translate();

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
    nodes: &'a [CompileNode],
    node: &'a CompileNode,
    node_id: NodeId,

    ptr_type: Type,
    module: &'a mut JITModule,
    scheduler: Value,
    argument: Value,
    builder: FunctionBuilder<'a>,
    globals: JitGlobals,
}

impl<'a> FunctionTranslator<'a> {
    fn translate(&mut self) {
        let false_value = self.false_value();
        self.store_buffer(self.globals.scheduled_data, self.node_id, false_value);

        match &self.node.state {
            Block::RedstoneRepeater { .. } => self.tick_repeater(),
            Block::RedstoneTorch { .. } => self.tick_torch(),
            Block::RedstoneWallTorch { .. } => self.tick_torch(),
            Block::RedstoneComparator { comparator } => self.tick_comparator(comparator),
            Block::RedstoneLamp { .. } => self.tick_lamp(),
            Block::StoneButton { .. } => self.tick_button(),
            Block::StonePressurePlate { .. } => self.tick_pressure_plate(),
            Block::Lever { .. } => self.tick_lever(),
            _ => {}
        }
    }

    fn tick_repeater(&mut self) {
        let [calc_block, cond1_p2_block, else_block, set_power_block, end_block] =
            self.create_blocks();

        let zero_value = self.zero_value();
        let full_power = self.full_power_value();

        let can_be_locked = self
            .node
            .inputs
            .iter()
            .any(|link| link.ty == LinkType::Side);
        if can_be_locked {
            let is_locked = self.get_locked(self.node_id);
            self.builder.ins().brnz(is_locked, end_block, &[]);
            self.builder.ins().jump(calc_block, &[]);
        } else {
            self.builder.ins().jump(calc_block, &[]);
        }

        self.switch_seal_block(calc_block);
        let should_be_powered = self.get_input_present(self.node, LinkType::Default);
        let is_powered = self.get_power(self.node_id);
        self.builder.ins().brz(is_powered, else_block, &[]);
        self.builder.ins().jump(cond1_p2_block, &[]);

        self.switch_seal_block(cond1_p2_block);
        self.builder.ins().brnz(should_be_powered, else_block, &[]);
        self.builder.ins().jump(set_power_block, &[zero_value]);

        self.switch_seal_block(else_block);
        self.builder.ins().brnz(is_powered, end_block, &[]);
        self.builder.ins().jump(set_power_block, &[full_power]);

        self.switch_seal_block(set_power_block);
        self.builder.append_block_param(set_power_block, types::I8);
        let power = self.builder.block_params(set_power_block)[0];
        self.set_power_and_update(self.node_id, power);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn tick_torch(&mut self) {
        let [cond1_p2_block, else_block, cond2_p2_block, set_power_block, end_block] =
            self.create_blocks();

        let zero_value = self.zero_value();
        let full_power = self.full_power_value();

        let should_be_off = self.get_input_present(self.node, LinkType::Default);
        let lit = self.get_power(self.node_id);
        self.builder.ins().brz(lit, else_block, &[]);
        self.builder.ins().jump(cond1_p2_block, &[]);

        self.switch_seal_block(cond1_p2_block);
        self.builder.ins().brz(should_be_off, else_block, &[]);
        self.builder.ins().jump(set_power_block, &[zero_value]);

        self.switch_seal_block(else_block);
        self.builder.ins().brnz(lit, end_block, &[]);
        self.builder.ins().jump(cond2_p2_block, &[]);

        self.switch_seal_block(cond2_p2_block);
        self.builder.ins().brnz(should_be_off, end_block, &[]);
        self.builder.ins().jump(set_power_block, &[full_power]);

        self.switch_seal_block(set_power_block);
        self.builder.append_block_param(set_power_block, types::I8);
        let power = self.builder.block_params(set_power_block)[0];
        self.set_power_and_update(self.node_id, power);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn tick_comparator(&mut self, comparator: &RedstoneComparator) {
        let [change_block, end_block] = self.create_blocks();

        let new_strength = self.calculate_comparator(self.node_id, comparator);
        let old_strength = self.get_power(self.node_id);
        self.builder.ins().br_icmp(
            IntCC::NotEqual,
            new_strength,
            old_strength,
            change_block,
            &[],
        );
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(change_block);
        self.set_power_and_update(self.node_id, new_strength);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn tick_lamp(&mut self) {
        let [cond1_p2_block, change_block, end_block] = self.create_blocks();

        let should_be_lit = self.get_input_value(self.node, LinkType::Default);
        let is_lit = self.get_power(self.node_id);
        self.builder.ins().brz(is_lit, end_block, &[]);
        self.builder.ins().jump(cond1_p2_block, &[]);

        self.switch_seal_block(cond1_p2_block);
        self.builder.ins().brnz(should_be_lit, end_block, &[]);
        self.builder.ins().jump(change_block, &[]);

        self.switch_seal_block(change_block);
        let zero_value = self.zero_value();
        self.set_power_and_update(self.node_id, zero_value);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn tick_button(&mut self) {
        let [pressed_block, normal_block, normal_change_block, set_power_block, end_block] =
            self.create_blocks();

        let pressed = self.argument;
        self.builder.ins().brnz(pressed, pressed_block, &[]);
        self.builder.ins().jump(normal_block, &[]);

        self.switch_seal_block(pressed_block);
        let full_power = self.full_power_value();
        self.builder.ins().jump(set_power_block, &[full_power]);

        self.switch_seal_block(normal_block);
        let powered = self.get_power(self.node_id);
        self.builder.ins().brnz(powered, normal_change_block, &[]);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(normal_change_block);
        let zero_value = self.zero_value();
        self.builder.ins().jump(set_power_block, &[zero_value]);

        self.switch_seal_block(set_power_block);
        self.builder.append_block_param(set_power_block, types::I8);
        let power = self.builder.block_params(set_power_block)[0];
        self.set_power_and_update(self.node_id, power);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn tick_pressure_plate(&mut self) {
        let [end_block] = self.create_blocks();

        let zero_value = self.zero_value();
        let full_power = self.full_power_value();
        let power = self.get_power(self.node_id);
        self.builder.ins().brnz(power, end_block, &[zero_value]);
        self.builder.ins().jump(end_block, &[full_power]);

        self.switch_seal_block(end_block);
        self.builder.append_block_param(end_block, types::I8);
        let power = self.builder.block_params(end_block)[0];
        self.set_power_and_update(self.node_id, power);
    }

    fn tick_lever(&mut self) {
        self.tick_pressure_plate();
    }

    fn node_changed(&mut self, node_id: NodeId) {
        let node = &self.nodes[node_id];

        for update in &node.updates {
            self.update_node(*update);
        }
        self.update_node(node_id);
    }

    fn update_node(&mut self, node_id: NodeId) {
        match &self.nodes[node_id].state {
            Block::RedstoneRepeater { repeater } => self.update_repeater(node_id, repeater),
            Block::RedstoneTorch { .. } => self.update_torch(node_id),
            Block::RedstoneWallTorch { .. } => self.update_torch(node_id),
            Block::RedstoneComparator { comparator } => self.update_comparator(node_id, comparator),
            Block::RedstoneLamp { .. } => self.update_lamp(node_id),
            Block::IronTrapdoor { .. } => self.update_trapdoor(node_id),
            Block::RedstoneWire { .. } => self.update_wire(node_id),
            _ => {}
        }
    }

    fn update_repeater(&mut self, node_id: NodeId, repeater: &RedstoneRepeater) {
        let node = &self.nodes[node_id];

        let [test_locked_block, test_scheduled_block, calc_power_block, set_power_block, power_change_block, end_block] =
            self.create_blocks();

        let can_be_locked = node.inputs.iter().any(|link| link.ty == LinkType::Side);
        if can_be_locked {
            let [set_locked_block] = self.create_blocks();

            let should_be_locked = self.get_input_present(node, LinkType::Side);
            let is_locked = self.get_locked(node_id);

            self.builder.ins().br_icmp(
                IntCC::NotEqual,
                should_be_locked,
                is_locked,
                set_locked_block,
                &[],
            );
            self.builder.ins().jump(test_locked_block, &[]);

            self.switch_seal_block(set_locked_block);
            self.set_locked(node_id, should_be_locked);
            self.builder.ins().jump(test_locked_block, &[]);
        } else {
            self.builder.ins().jump(test_locked_block, &[]);
        }

        self.switch_seal_block(test_locked_block);
        let is_locked = self.get_locked(node_id);
        self.builder.ins().brnz(is_locked, end_block, &[]);
        self.builder.ins().jump(test_scheduled_block, &[]);

        self.switch_seal_block(test_scheduled_block);
        let is_scheduled = self.get_scheduled(node_id);
        self.builder.ins().brnz(is_scheduled, end_block, &[]);
        self.builder.ins().jump(calc_power_block, &[]);

        self.switch_seal_block(calc_power_block);
        let now_powered = self.get_input_present(node, LinkType::Default);
        let was_powered = self.get_power(node_id);
        let was_powered = self.int_to_bool(was_powered);
        self.builder.ins().br_icmp(
            IntCC::NotEqual,
            now_powered,
            was_powered,
            set_power_block,
            &[],
        );
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(set_power_block);
        if node.facing_diode {
            let priority = self
                .builder
                .ins()
                .iconst(types::I8, TickPriority::Highest as i64);
            self.builder.ins().jump(power_change_block, &[priority]);
        } else {
            let power_on_prio = self
                .builder
                .ins()
                .iconst(types::I8, TickPriority::High as i64);
            let power_off_prio = self
                .builder
                .ins()
                .iconst(types::I8, TickPriority::Higher as i64);
            self.builder
                .ins()
                .brnz(now_powered, power_change_block, &[power_on_prio]);
            self.builder
                .ins()
                .jump(power_change_block, &[power_off_prio]);
        };

        self.switch_seal_block(power_change_block);
        self.builder
            .append_block_param(power_change_block, types::I8);
        let priority = self.builder.block_params(power_change_block)[0];
        self.schedule_tick(node_id, repeater.delay as usize, priority);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn update_torch(&mut self, node_id: NodeId) {
        let node = &self.nodes[node_id];

        let [update_block, schedule_block, end_block] = self.create_blocks();

        let scheduled = self.get_scheduled(node_id);
        self.builder.ins().brnz(scheduled, end_block, &[]);
        self.builder.ins().jump(update_block, &[]);

        self.switch_seal_block(update_block);
        let should_be_off = self.get_input_present(node, LinkType::Default);
        let power = self.get_power(node_id);
        let lit = self.int_to_bool(power);
        self.builder
            .ins()
            .br_icmp(IntCC::Equal, should_be_off, lit, schedule_block, &[]);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(schedule_block);
        self.schedule_tick_static(node_id, 1, CLTickPriority::Normal);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn update_comparator(&mut self, node_id: NodeId, comparator: &RedstoneComparator) {
        let [update_block, schedule_block, end_block] = self.create_blocks();

        let scheduled = self.get_scheduled(node_id);
        self.builder.ins().brnz(scheduled, end_block, &[]);
        self.builder.ins().jump(update_block, &[]);

        self.switch_seal_block(update_block);
        let new_power = self.calculate_comparator(node_id, comparator);
        let old_power = self.get_power(node_id);
        self.builder
            .ins()
            .br_icmp(IntCC::NotEqual, old_power, new_power, schedule_block, &[]);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(schedule_block);
        let priority = if self.nodes[node_id].facing_diode {
            CLTickPriority::High
        } else {
            CLTickPriority::Normal
        };
        self.schedule_tick_static(node_id, 1, priority);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn update_lamp(&mut self, node_id: NodeId) {
        let node = &self.nodes[node_id];

        let [cond1_p2_block, schedule_off_block, else_block, cond2_p2_block, turn_on_block, end_block] =
            self.create_blocks();

        let should_be_lit = self.get_input_value(node, LinkType::Default);
        let lit = self.get_power(node_id);
        self.builder.ins().brz(lit, else_block, &[]);
        self.builder.ins().jump(cond1_p2_block, &[]);

        self.switch_seal_block(cond1_p2_block);
        self.builder.ins().brnz(should_be_lit, else_block, &[]);
        self.builder.ins().jump(schedule_off_block, &[]);

        self.switch_seal_block(schedule_off_block);
        self.schedule_tick_static(node_id, 2, CLTickPriority::Normal);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(else_block);
        self.builder.ins().brnz(lit, end_block, &[]);
        self.builder.ins().jump(cond2_p2_block, &[]);

        self.switch_seal_block(cond2_p2_block);
        self.builder.ins().brz(should_be_lit, end_block, &[]);
        self.builder.ins().jump(turn_on_block, &[]);

        self.switch_seal_block(turn_on_block);
        let one_value = self.one_value();
        self.set_power(node_id, one_value);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn update_trapdoor(&mut self, node_id: NodeId) {
        let node = &self.nodes[node_id];

        let [change_block, end_block] = self.create_blocks();

        let should_be_powered = self.get_input_value(node, LinkType::Default);
        let powered = self.get_power(node_id);
        self.builder.ins().br_icmp(
            IntCC::NotEqual,
            should_be_powered,
            powered,
            change_block,
            &[],
        );
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(change_block);
        self.set_power(node_id, should_be_powered);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn update_wire(&mut self, node_id: NodeId) {
        let node = &self.nodes[node_id];

        let new_power = self.get_input_value(node, LinkType::Default);
        let current_power = self.get_power(node_id);

        let [change_block, end_block] = self.create_blocks();

        self.builder
            .ins()
            .br_icmp(IntCC::NotEqual, current_power, new_power, change_block, &[]);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(change_block);
        self.set_power(node_id, new_power);
        self.builder.ins().jump(end_block, &[]);

        self.switch_seal_block(end_block);
    }

    fn calculate_comparator(&mut self, node_id: NodeId, comparator: &RedstoneComparator) -> Value {
        let node = &self.nodes[node_id];

        let [end_block] = self.create_blocks();

        let input_power = self.get_input_value(node, LinkType::Default);
        let side_input_power = self.get_input_value(node, LinkType::Side);

        if let Some(far_override) = node.comparator_far_input {
            let full_power = self.full_power_value();
            let override_power = self.builder.ins().iconst(types::I8, far_override as i64);
            self.builder.ins().br_icmp(
                IntCC::Equal,
                input_power,
                full_power,
                end_block,
                &[full_power],
            );
            self.builder.ins().jump(end_block, &[override_power]);
        } else {
            self.builder.ins().jump(end_block, &[input_power]);
        }

        self.switch_seal_block(end_block);
        self.builder.append_block_param(end_block, types::I8);
        let input_power = self.builder.block_params(end_block)[0];

        self.calculate_comparator_output(comparator.mode, input_power, side_input_power)
    }

    fn calculate_comparator_output(
        &mut self,
        mode: ComparatorMode,
        input_strength: Value,
        power_on_sides: Value,
    ) -> Value {
        let zero_value = self.zero_value();
        let [end_block] = self.create_blocks();

        match mode {
            ComparatorMode::Compare => {
                self.builder.ins().br_icmp(
                    IntCC::UnsignedGreaterThanOrEqual,
                    input_strength,
                    power_on_sides,
                    end_block,
                    &[input_strength],
                );
                self.builder.ins().jump(end_block, &[zero_value]);
            }
            ComparatorMode::Subtract => {
                let [sub_block] = self.create_blocks();
                self.builder.ins().br_icmp(
                    IntCC::UnsignedGreaterThanOrEqual,
                    power_on_sides,
                    input_strength,
                    end_block,
                    &[zero_value],
                );
                self.builder.ins().jump(sub_block, &[]);

                self.switch_seal_block(sub_block);
                let output_value = self.builder.ins().isub(input_strength, power_on_sides);
                self.builder.ins().jump(end_block, &[output_value]);
            }
        }

        self.switch_seal_block(end_block);
        self.builder.append_block_param(end_block, types::I8);
        self.builder.block_params(end_block)[0]
    }

    fn get_input_present(&mut self, node: &CompileNode, ty: LinkType) -> Value {
        let true_value = self.true_value();
        let false_value = self.false_value();
        let [end_block] = self.create_blocks();

        let mut next_block = self.builder.create_block();
        for link in node.inputs.iter().filter(|link| link.ty == ty) {
            let link_power = self.get_power(link.end);
            let link_weight = self.builder.ins().iconst(types::I8, link.weight as i64);
            self.builder.ins().br_icmp(
                IntCC::UnsignedGreaterThan,
                link_power,
                link_weight,
                end_block,
                &[true_value],
            );
            self.builder.ins().jump(next_block, &[]);
            self.switch_seal_block(next_block);
            next_block = self.builder.create_block();
        }

        self.builder.ins().jump(end_block, &[false_value]);

        self.switch_seal_block(end_block);
        self.builder.append_block_param(end_block, types::I8);
        self.builder.block_params(end_block)[0]
    }

    fn get_input_value(&mut self, node: &CompileNode, ty: LinkType) -> Value {
        node.inputs
            .iter()
            .filter(|link| link.ty == ty)
            .fold(self.zero_value(), |max, link| {
                self.calculate_strength(max, link)
            })
    }

    fn calculate_strength(&mut self, cur_max: Value, link: &Link) -> Value {
        let link_output = self.get_power(link.end);
        let link_weight = self.builder.ins().iconst(types::I8, link.weight as i64);
        let link_strength = self.builder.ins().isub(link_output, link_weight);
        self.builder.ins().imax(cur_max, link_strength)
    }

    fn int_to_bool(&mut self, int: Value) -> Value {
        let [end_block] = self.create_blocks();

        let false_value = self.false_value();
        let true_value = self.true_value();

        self.builder.ins().brz(int, end_block, &[false_value]);
        self.builder.ins().jump(end_block, &[true_value]);

        self.switch_seal_block(end_block);
        self.builder.append_block_param(end_block, types::I8);
        self.builder.block_params(end_block)[0]
    }

    fn load_buffer(&mut self, buffer: usize, offset: usize) -> Value {
        assert!(offset < self.nodes.len());
        let addr = self
            .builder
            .ins()
            .iconst(self.ptr_type, (buffer + offset) as i64);
        self.builder
            .ins()
            .load(types::I8, MemFlags::trusted(), addr, 0)
    }

    fn store_buffer(&mut self, buffer: usize, offset: usize, value: Value) {
        assert!(offset < self.nodes.len());
        let addr = self
            .builder
            .ins()
            .iconst(self.ptr_type, (buffer + offset) as i64);
        self.builder
            .ins()
            .store(MemFlags::trusted(), value, addr, 0);
    }

    fn get_power(&mut self, node_id: NodeId) -> Value {
        self.load_buffer(self.globals.power_data, node_id)
    }

    fn set_power_and_update(&mut self, node_id: usize, power: Value) {
        self.set_power(node_id, power);
        self.node_changed(node_id);
    }

    fn set_power(&mut self, node_id: NodeId, power: Value) {
        self.store_buffer(self.globals.power_data, node_id, power);
        let true_value = self.true_value();
        self.store_buffer(self.globals.changed_data, node_id, true_value);
    }

    fn get_scheduled(&mut self, node_id: NodeId) -> Value {
        self.load_buffer(self.globals.scheduled_data, node_id)
    }

    fn get_locked(&mut self, node_id: NodeId) -> Value {
        self.load_buffer(self.globals.locked_data, node_id)
    }

    fn set_locked(&mut self, node_id: NodeId, value: Value) {
        let value = self.int_to_bool(value);
        self.store_buffer(self.globals.locked_data, node_id, value);
        let true_value = self.true_value();
        self.store_buffer(self.globals.changed_data, node_id, true_value);
    }

    fn schedule_tick_static(&mut self, node_id: NodeId, delay: usize, priority: CLTickPriority) {
        let priority = self.builder.ins().iconst(types::I8, priority as i64);
        self.schedule_tick(node_id, delay, priority);
    }

    fn schedule_tick(&mut self, node_id: NodeId, delay: usize, priority: Value) {
        assert!(node_id < self.nodes.len());

        let true_value = self.true_value();
        self.store_buffer(self.globals.scheduled_data, node_id, true_value);

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
        let node_id = self.builder.ins().iconst(self.ptr_type, node_id as i64);
        let delay = self.builder.ins().iconst(self.ptr_type, delay as i64);
        self.builder.ins().call_indirect(
            sig,
            jit_schedule_tick_value,
            &[self.scheduler, node_id, delay, priority],
        );
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

    fn full_power_value(&mut self) -> Value {
        self.builder.ins().iconst(types::I8, 15)
    }

    fn create_blocks<const COUNT: usize>(&mut self) -> [CLBlock; COUNT] {
        array::from_fn(|_| self.builder.create_block())
    }

    fn switch_seal_block(&mut self, block: CLBlock) {
        self.builder.switch_to_block(block);
        self.builder.seal_block(block);
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
    // unsafe fn into_slice(self) -> Box<[T]> {
    //     Box::from_raw(self.ptr)
    // }

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
