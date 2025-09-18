#[allow(dead_code)]
mod compute;
#[allow(dead_code)]
mod hello_triangle;
mod render;

use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    render::main();
    // //region System
    // // Initialize logger (env_logger like original)
    // env_logger::init();
    //
    // // Instance
    // let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    //
    // // Adapter
    // let adapter =
    //     pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    //         .expect("Failed to create adapter");
    // println!("Running on Adapter: {:#?}", adapter.get_info());
    //
    // let downlevel_capabilities = adapter.get_downlevel_capabilities();
    // if !downlevel_capabilities
    //     .flags
    //     .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    // {
    //     panic!("Adapter does not support compute shaders");
    // }
    //
    // // Device + Queue (mirroring original required limits/features)
    // let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
    //     label: None,
    //     required_features: wgpu::Features::empty(),
    //     required_limits: wgpu::Limits::downlevel_defaults(),
    //     memory_hints: wgpu::MemoryHints::MemoryUsage,
    //     trace: wgpu::Trace::Off,
    // }))
    // .expect("Failed to create device");
    // //endregion
    //
    // //region prepare
    // // Use generated shader module + pipeline layout helpers
    // let pipeline = compute::compute::create_doubleMe_pipeline(&device);
    //
    // // Single storage buffer (shader does in-place modification); original sample had separate input/output.
    // let arguments: Vec<compute::types::Cell> = vec![
    //     compute::types::Cell{
    //         vx: 1,
    //         vy: 2,
    //         vz: 3,
    //         mass: 4,
    //     }
    // ];
    // let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: Some("storage"),
    //     contents: bytemuck::cast_slice(&arguments),
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    // });
    // let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("download"),
    //     size: storage_buffer.size(),
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    //     mapped_at_creation: false,
    // });
    //
    // // Bind group via generated helper (encapsulates layout creation)
    // let bind_group0 = compute::bind_groups::BindGroup0::from_bindings(
    //     &device,
    //     compute::bind_groups::BindGroupLayout0 {
    //         input: storage_buffer.as_entire_buffer_binding(),
    //     },
    // );
    // //endregion
    //
    // //region compute
    // // Command encoding
    // let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //     label: Some("encoder"),
    // });
    // {
    //     let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //         label: Some("compute_pass"),
    //         timestamp_writes: None,
    //     });
    //     compute_pass.set_pipeline(&pipeline);
    //     // Using helper to set bind group(s)
    //     compute::set_bind_groups(&mut compute_pass, &bind_group0);
    //
    //     // Workgroup sizing same as original logic
    //     let workgroup_size = compute::compute::DOUBLEME_WORKGROUP_SIZE[0] as usize; // 64
    //     let workgroup_count = arguments.len().div_ceil(workgroup_size);
    //     compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    // }
    //
    // // Copy GPU output to download buffer
    // encoder.copy_buffer_to_buffer(
    //     &storage_buffer,
    //     0,
    //     &download_buffer,
    //     0,
    //     storage_buffer.size(),
    // );
    // let command_buffer = encoder.finish();
    // queue.submit([command_buffer]);
    //
    // // Map for reading (async like original)
    // let buffer_slice = download_buffer.slice(..);
    // let notify = Arc::new(Notify::new());
    // let notify_clone = notify.clone();
    // buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
    //     if let Err(e) = result {
    //         eprintln!("Failed to map buffer: {:?}", e);
    //     }
    //     notify_clone.notify_one();
    // });
    // device.poll(PollType::Wait).unwrap();
    // notify.notified().await;
    //
    // let data = buffer_slice.get_mapped_range();
    // let result: &[f32] = bytemuck::cast_slice(&data);
    // println!("Result: {:?}", result);
    // //endregion
}
