#[cfg(feature = "jit_cranelift")]
pub mod cranelift;
pub mod direct;
// pub mod par_direct;

// pub mod cranelift_old;

use crate::plot::PlotWorld;
use mchprs_blocks::BlockPos;
use mchprs_world::TickEntry;

use super::CompileNode;

pub trait JITBackend {
    fn compile(&mut self, nodes: Vec<CompileNode>, ticks: Vec<TickEntry>);
    fn tick(&mut self, plot: &mut PlotWorld);
    fn on_use_block(&mut self, plot: &mut PlotWorld, pos: BlockPos);
    fn set_pressure_plate(&mut self, plot: &mut PlotWorld, pos: BlockPos, powered: bool);
    fn flush(&mut self, plot: &mut PlotWorld, io_only: bool);
    fn reset(&mut self, plot: &mut PlotWorld, io_only: bool);
}
