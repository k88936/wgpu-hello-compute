mod shader; // make the generated shader module visible
use std::sync::Arc;
use tokio::sync::Notify;

// Re-export wgpu utilities if needed
use wgpu::util::DeviceExt;
use wgpu::wgt::PollType;
// We'll use the generated shader module in this crate (`crate::shader`).
// The original example logic is preserved as closely as possible.

#[tokio::main]
async fn main() {
    // Original: parse arguments -> here we just hardcode same demo vector
    let arguments: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    println!("Parsed {} arguments", arguments.len());

    // Initialize logger (env_logger like original)
    env_logger::init();

    // Instance
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    // Adapter
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to create adapter");
    println!("Running on Adapter: {:#?}", adapter.get_info());

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders");
    }

    // Device + Queue (mirroring original required limits/features)
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    // Use generated shader module + pipeline layout helpers
    let module = crate::shader::create_shader_module(&device);
    let pipeline_layout = crate::shader::create_pipeline_layout(&device);
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline doubleMe (rewritten)"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(crate::shader::ENTRY_DOUBLEME),
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    // Buffers (same as original logic)
    let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&arguments),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download"),
        size: input_data_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    // We still must guarantee min_binding_size like original example. The auto layout doesn't set it, so if strict size needed we can keep a manual layout.
    // For adherence to generated layout, proceed without explicit min_binding_size (wgpu validation allows None if shader uses it correctly).

    // Create bind group using generated helper struct
    let bind_group0 = crate::shader::bind_groups::BindGroup0::from_bindings(
        &device,
        crate::shader::bind_groups::BindGroupLayout0 {
            input: wgpu::BufferBinding {
                buffer: &input_data_buffer,
                offset: 0,
                size: None,
            },
        },
    );

    // Command encoding
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        // Using helper to set bind group(s)
        crate::shader::set_bind_groups(&mut compute_pass, &bind_group0);

        // Workgroup sizing same as original logic
        let workgroup_size = crate::shader::compute::DOUBLEME_WORKGROUP_SIZE[0] as usize; // 64
        let workgroup_count = arguments.len().div_ceil(workgroup_size);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    // Copy GPU output to download buffer
    encoder.copy_buffer_to_buffer(
        &input_data_buffer,
        0,
        &download_buffer,
        0,
        input_data_buffer.size(),
    );
    let command_buffer = encoder.finish();
    queue.submit([command_buffer]);

    // Map for reading (async like original)
    let buffer_slice = download_buffer.slice(..);
    let notify = Arc::new(Notify::new());
    let notify_clone = notify.clone();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        if let Err(e) = result {
            eprintln!("Failed to map buffer: {:?}", e);
        }
        notify_clone.notify_one();
    });
    device.poll(PollType::Wait).unwrap();
    notify.notified().await;

    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);
    println!("Result: {:?}", result);
}
