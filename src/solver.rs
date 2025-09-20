//! MLS-MPM solver orchestration translated from `shaders/mls-mpm.ts`.
//!
//! This module encapsulates creation of compute pipelines, GPU buffers and
//! dispatch order for one simulation step. It mirrors (subset of) the JS/TS
//! `MLSMPMSimulator` API so the render loop can simply call
//! `solver.execute(&mut encoder, params)` each frame.
//!
//! NOTE: The original TypeScript version dynamically spawns particles over
//! time and responds to mouse interaction. For this initial Rust port we keep
//! the structure but stub user input (mouse) handling. Follow-up work can wire
//! real input sources.

use encase::{ShaderType, UniformBuffer, internal::WriteInto};

// Re-export particle type expected by rendering (copyPosition writes into the
// sphere PosVel buffer). The rendering pipeline already owns a STORAGE buffer
// of `sphere::types::PosVel` (position, velocity, density).

// Convenience constants (mirroring TS defaults)
const MAX_X_GRIDS: u32 = 80;
const MAX_Y_GRIDS: u32 = 80;
const MAX_Z_GRIDS: u32 = 80;
const CELL_STRUCT_SIZE: u64 = std::mem::size_of::<clearGrid::types::Cell>() as u64; // 16 bytes

// A single MPM particle as used by compute shaders (layout matches generated code)
#[repr(C)]
#[derive(Debug, Copy, Clone, ShaderType)]
pub struct Particle {
	pub position: glam::Vec3,
	pub v: glam::Vec3,
	pub C: glam::Mat3,
}

// Internal view of simulation uniform scalars we pass as separate buffers in WGSL.
// (Each single value buffer is represented as its own `wgpu::Buffer` for parity with generated layouts.)

pub struct MLSMPMSolver {
	// Pipelines
	clear_grid_pipeline: wgpu::ComputePipeline,
	spawn_particles_pipeline: wgpu::ComputePipeline,
	p2g1_pipeline: wgpu::ComputePipeline,
	p2g2_pipeline: wgpu::ComputePipeline,
	update_grid_pipeline: wgpu::ComputePipeline,
	g2p_pipeline: wgpu::ComputePipeline,
	copy_position_pipeline: wgpu::ComputePipeline,

	// Buffers
	particle_buffer: wgpu::Buffer,    // storage array<Particle>
	cell_buffer: wgpu::Buffer,        // storage array<Cell/AtomCell>
	density_buffer: wgpu::Buffer,     // storage f32 densities per particle
	real_box_size_buffer: wgpu::Buffer, // uniform vec3f
	init_box_size_buffer: wgpu::Buffer, // uniform vec3f
	num_particles_buffer: wgpu::Buffer, // uniform u32 / i32 depending on shader usage
	sphere_radius_buffer: wgpu::Buffer, // uniform f32

	// Bind groups (created fresh when any underlying buffer replaced)
	clear_grid_bg: clearGrid::bind_groups::BindGroup0,
	spawn_particles_bg: spawnParticles::bind_groups::BindGroup0,
	p2g1_bg: p2g_1::bind_groups::BindGroup0,
	p2g2_bg: p2g_2::bind_groups::BindGroup0,
	update_grid_bg: updateGrid::bind_groups::BindGroup0,
	g2p_bg: g2p::bind_groups::BindGroup0,
	copy_position_bg: copyPosition::bind_groups::BindGroup0,

	// Simulation parameters / counters
	pub num_particles: u32,
	frame_count: u32,
	grid_count: u32,
	// Cached dimensions
	init_box_size: glam::Vec3,
	real_box_size: glam::Vec3,
	// Constants
	fixed_point_multiplier: f32,
	stiffness: f32,
	rest_density: f32,
	dynamic_viscosity: f32,
	dt: f32,
}

impl MLSMPMSolver {
	pub fn new(
		device: &wgpu::Device,
		// Buffer that the renderer uses (pos + density + velocity) updated by copyPosition pass
		render_posvel_buffer: &wgpu::Buffer,
		max_particles: u32,
	) -> Self {
		// Pipelines
		let clear_grid_pipeline = clearGrid::compute::create_clearGrid_pipeline(device);
		let spawn_particles_pipeline = spawnParticles::compute::create_spawn_pipeline(device);
		let p2g1_pipeline = p2g_1::compute::create_p2g_1_pipeline(device);
		let p2g2_pipeline = p2g_2::compute::create_p2g_2_pipeline(device);
		let update_grid_pipeline = updateGrid::compute::create_updateGrid_pipeline(device);
		let g2p_pipeline = g2p::compute::create_g2p_pipeline(device);
		let copy_position_pipeline = copyPosition::compute::create_copyPosition_pipeline(device);

		// Allocate maximum sized buffers similar to JS side
		let max_grid_count = MAX_X_GRIDS * MAX_Y_GRIDS * MAX_Z_GRIDS;
		let cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("cells buffer"),
			size: CELL_STRUCT_SIZE * max_grid_count as u64,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let particle_stride = std::mem::size_of::<Particle>() as u64;
		let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("mpm particles"),
			size: particle_stride * max_particles as u64,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("density buffer"),
			size: 4 * max_particles as u64,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});

		// Helper to create small uniform buffers (vec3f or scalar). We use encase to serialize.
		fn create_uniform<T: ShaderType + Copy + WriteInto>(
			device: &wgpu::Device,
			value: &T,
			label: &str,
		) -> wgpu::Buffer {
			let mut ubuf = UniformBuffer::new(Vec::new());
			ubuf.write(value).expect("serialize uniform");
			device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some(label),
				contents: ubuf.as_ref(),
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			})
		}

		let init_box_size = glam::Vec3::new(12.0, 12.0, 12.0);
		let real_box_size = init_box_size;
		let num_particles_u32: u32 = 0;
		let sphere_radius: f32 = 3.0;

		let real_box_size_buffer = create_uniform(device, &real_box_size, "real-box-size");
		let init_box_size_buffer = create_uniform(device, &init_box_size, "init-box-size");
		let num_particles_buffer = create_uniform(device, &num_particles_u32, "num-particles");
		let sphere_radius_buffer = create_uniform(device, &sphere_radius, "sphere-radius");

		// Build bind groups with current buffers
		let clear_grid_bg = clearGrid::bind_groups::BindGroup0::from_bindings(
			device,
			clearGrid::bind_groups::BindGroupLayout0 {
				cells: cell_buffer.as_entire_buffer_binding(),
			},
		);
		let spawn_particles_bg = spawnParticles::bind_groups::BindGroup0::from_bindings(
			device,
			spawnParticles::bind_groups::BindGroupLayout0 {
				particles: particle_buffer.as_entire_buffer_binding(),
				init_box_size: init_box_size_buffer.as_entire_buffer_binding(),
				numParticles: num_particles_buffer.as_entire_buffer_binding(),
			},
		);
		let p2g1_bg = p2g_1::bind_groups::BindGroup0::from_bindings(
			device,
			p2g_1::bind_groups::BindGroupLayout0 {
				particles: particle_buffer.as_entire_buffer_binding(),
				cells: cell_buffer.as_entire_buffer_binding(),
				init_box_size: init_box_size_buffer.as_entire_buffer_binding(),
				numParticles: num_particles_buffer.as_entire_buffer_binding(),
			},
		);
		let p2g2_bg = p2g_2::bind_groups::BindGroup0::from_bindings(
			device,
			p2g_2::bind_groups::BindGroupLayout0 {
				particles: particle_buffer.as_entire_buffer_binding(),
				cells: cell_buffer.as_entire_buffer_binding(),
				init_box_size: init_box_size_buffer.as_entire_buffer_binding(),
				numParticles: num_particles_buffer.as_entire_buffer_binding(),
				densities: density_buffer.as_entire_buffer_binding(),
			},
		);
		let update_grid_bg = updateGrid::bind_groups::BindGroup0::from_bindings(
			device,
			updateGrid::bind_groups::BindGroupLayout0 {
				cells: cell_buffer.as_entire_buffer_binding(),
				real_box_size: real_box_size_buffer.as_entire_buffer_binding(),
				init_box_size: init_box_size_buffer.as_entire_buffer_binding(),
			},
		);
		let g2p_bg = g2p::bind_groups::BindGroup0::from_bindings(
			device,
			g2p::bind_groups::BindGroupLayout0 {
				particles: particle_buffer.as_entire_buffer_binding(),
				cells: cell_buffer.as_entire_buffer_binding(),
				real_box_size: real_box_size_buffer.as_entire_buffer_binding(),
				init_box_size: init_box_size_buffer.as_entire_buffer_binding(),
				numParticles: num_particles_buffer.as_entire_buffer_binding(),
				sphereRadius: sphere_radius_buffer.as_entire_buffer_binding(),
			},
		);
		let copy_position_bg = copyPosition::bind_groups::BindGroup0::from_bindings(
			device,
			copyPosition::bind_groups::BindGroupLayout0 {
				particles: particle_buffer.as_entire_buffer_binding(),
				posvel: render_posvel_buffer.as_entire_buffer_binding(),
				numParticles: num_particles_buffer.as_entire_buffer_binding(),
				densities: density_buffer.as_entire_buffer_binding(),
			},
		);

		Self {
			clear_grid_pipeline,
			spawn_particles_pipeline,
			p2g1_pipeline,
			p2g2_pipeline,
			update_grid_pipeline,
			g2p_pipeline,
			copy_position_pipeline,
			particle_buffer,
			cell_buffer,
			density_buffer,
			real_box_size_buffer,
			init_box_size_buffer,
			num_particles_buffer,
			sphere_radius_buffer,
			clear_grid_bg,
			spawn_particles_bg,
			p2g1_bg,
			p2g2_bg,
			update_grid_bg,
			g2p_bg,
			copy_position_bg,
			num_particles: 0,
			frame_count: 0,
			grid_count: 0,
			init_box_size,
			real_box_size,
			fixed_point_multiplier: 1e7,
			stiffness: 3.0,
			rest_density: 4.0,
			dynamic_viscosity: 0.1,
			dt: 0.20,
		}
	}

	/// Execute one simulation step: (optionally spawn) -> clear -> p2g passes -> updateGrid -> g2p -> copyPosition.
	pub fn execute(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, target_num_particles: u32) {
		// First pass: (optional) spawn new particles (simple strategy every other frame like TS)
		if self.frame_count % 2 == 0 && self.num_particles < target_num_particles {
			// Increase particle count in-place (here we just jump by 100 like TS shape 10x10)
			let spawn_batch = 100.min(target_num_particles - self.num_particles);
			self.num_particles += spawn_batch;
			self.write_num_particles(queue);
			// Dispatch spawn compute to initialize last N particles
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("spawn"), timestamp_writes: None });
			cpass.set_pipeline(&self.spawn_particles_pipeline);
			spawnParticles::set_bind_groups(&mut cpass, &self.spawn_particles_bg);
			cpass.dispatch_workgroups(1, 1, 1); // workgroup_size(1)
		}

		// Clear grid
		{
			let total_cells = (self.init_box_size.x * self.init_box_size.y * self.init_box_size.z) as u32;
			let workgroups = (total_cells + 63) / 64; // workgroup_size(64)
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("clearGrid"), timestamp_writes: None });
			cpass.set_pipeline(&self.clear_grid_pipeline);
			clearGrid::set_bind_groups(&mut cpass, &self.clear_grid_bg);
			cpass.dispatch_workgroups(workgroups, 1, 1);
		}

		// p2g1
		{
			let workgroups = (self.num_particles + 63) / 64;
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("p2g_1"), timestamp_writes: None });
			cpass.set_pipeline(&self.p2g1_pipeline);
			p2g_1::set_bind_groups(&mut cpass, &self.p2g1_bg);
			cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
		}
		// p2g2
		{
			let workgroups = (self.num_particles + 63) / 64;
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("p2g_2"), timestamp_writes: None });
			cpass.set_pipeline(&self.p2g2_pipeline);
			p2g_2::set_bind_groups(&mut cpass, &self.p2g2_bg);
			cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
		}
		// updateGrid
		{
			let total_cells = (self.init_box_size.x * self.init_box_size.y * self.init_box_size.z) as u32;
			let workgroups = (total_cells + 63) / 64;
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("updateGrid"), timestamp_writes: None });
			cpass.set_pipeline(&self.update_grid_pipeline);
			updateGrid::set_bind_groups(&mut cpass, &self.update_grid_bg);
			cpass.dispatch_workgroups(workgroups, 1, 1);
		}
		// g2p
		{
			let workgroups = (self.num_particles + 63) / 64;
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("g2p"), timestamp_writes: None });
			cpass.set_pipeline(&self.g2p_pipeline);
			g2p::set_bind_groups(&mut cpass, &self.g2p_bg);
			cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
		}
		// copyPosition (write into renderer posvel buffer and densities)
		{
			let workgroups = (self.num_particles + 63) / 64;
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("copyPosition"), timestamp_writes: None });
			cpass.set_pipeline(&self.copy_position_pipeline);
			copyPosition::set_bind_groups(&mut cpass, &self.copy_position_bg);
			cpass.dispatch_workgroups(workgroups.max(1), 1, 1);
		}

		self.frame_count += 1;
	}

	fn write_num_particles(&self, queue: &wgpu::Queue) {
		let mut ubuf = UniformBuffer::new(Vec::new());
		ubuf
			.write(&self.num_particles)
			.expect("serialize num particles");
		queue.write_buffer(&self.num_particles_buffer, 0, ubuf.as_ref());
	}
}

// Required modules from sibling generated code
use crate::{clearGrid, copyPosition, g2p, p2g_1, p2g_2, spawnParticles, updateGrid};
use wgpu::util::DeviceExt; // for create_buffer_init

