mod sphere;
mod updateGrid;
mod spawnParticles;
mod p2g_1;
mod p2g_2;
mod g2p;
mod copyPosition;
mod clearGrid;
mod solver;

const TARGET_NUM_PARTICLES: u32 = 1_000_000; // TS medium parameter set (mlsmpmNumParticleParams[1])

use encase::{UniformBuffer, StorageBuffer};
use wgpu::util::DeviceExt;
use wgpu::BufferUsages;
use wgpu::wgt::BufferDescriptor;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

#[derive(Default)]
struct App {
    window: Option<&'static Window>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    pipeline: Option<wgpu::RenderPipeline>,
    // GPU resources that must outlive passes
    bind_group: Option<sphere::bind_groups::BindGroup0>,
    particles_buffer: Option<wgpu::Buffer>,
    uniforms_buffer: Option<wgpu::Buffer>,
    stretch_buffer: Option<wgpu::Buffer>,
    // Second color attachment (for depth-as-color output)
    aux_rt_texture: Option<wgpu::Texture>,
    aux_rt_view: Option<wgpu::TextureView>,
    start_time: Option<std::time::Instant>,
    // Depth buffer
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    // MLS-MPM solver
    solver: Option<solver::MLSMPMSolver>,
}

impl App {
    fn new() -> Self {
        Self::default()
    }
    fn update_uniforms(&self, size: winit::dpi::PhysicalSize<u32>, time: f32) {
        if let (Some(queue), Some(uniforms_buffer)) =
            (self.queue.as_ref(), self.uniforms_buffer.as_ref())
        {
            // Create perspective projection and view matrices
            let aspect = size.width as f32 / size.height as f32;
            // Use WebGPU-friendly depth range [0,1]
            let projection_matrix = glam::Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4, // 45 degree FOV
                aspect,
                0.1,   // near plane
                100.0, // far plane
            );

            // Auto-rotate camera around TS medium box (0..60) with center (30,30,30)
            let box_center = glam::Vec3::new(30.0, 30.0, 30.0);
            // Distance ~ init distance param mlsmpmInitDistances[1] = 70
            let radius = 70.0;
            let angular_time = time * 0.05;
            let camera_x = box_center.x + radius * angular_time.cos();
            let camera_z = box_center.z + radius * angular_time.sin();
            // Slight elevated angle so we see depth and layering
            let camera_y = box_center.y + 6.0; // tilt downward a bit
            let camera_position = glam::Vec3::new(camera_x, camera_y, camera_z);

            let view_matrix = glam::Mat4::look_at_rh(
                camera_position,
                box_center, // look at center of unshifted simulation box
                glam::Vec3::Y,
            );

            // Create uniform buffer for render uniforms
            let uniforms = sphere::RenderUniforms {
                inv_projection_matrix: projection_matrix.inverse(),
                projection_matrix,
                view_matrix,
                inv_view_matrix: view_matrix.inverse(),
                texel_size: glam::Vec2::new(1.0 / size.width as f32, 1.0 / size.height as f32),
                sphere_size: 1.0,
            };

            // Update the uniforms buffer using encase
            let mut ubuf = UniformBuffer::new(Vec::new());
            ubuf.write(&uniforms).expect("serialize updated uniforms");
            queue.write_buffer(uniforms_buffer, 0, ubuf.as_ref());
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create a window if we don't have one yet (mobile may resume multiple times)
        if self.window.is_none() {
            let window = event_loop
                .create_window(WindowAttributes::default())
                .expect("failed to create window");
            let window: &'static Window = Box::leak(Box::new(window));
            let size = window.inner_size();

            // WGPU setup
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
            let surface = instance.create_surface(window).expect("create surface");

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                }))
                .expect("Failed to find an appropriate adapter");

            let required_features = wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY;
            let required_limits = wgpu::Limits::default();

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
                    required_limits,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                }))
                .expect("Failed to create device");

            // let points: Vec<sphere::types::PosVel> = {
            //     let mut particles = Vec::with_capacity(1000);
            //     for x in 0..10 {
            //         for y in 0..10 {
            //             for z in 0..10 {
            //                 particles.push(sphere::types::PosVel {
            //                     position: glam::Vec3::new(
            //                         (x as f32 - 4.5) * 0.1,
            //                         (y as f32 - 4.5) * 0.1,
            //                         (z as f32 - 4.5) * 0.1,
            //                     ),
            //                     v: glam::Vec3::new(0.0, 0.0, 0.0),
            //                     density: 1.0,
            //                 });
            //             }
            //         }
            //     }
            //     particles
            // };
            // // Serialize particle data using a single slice write so encase applies the correct stride (48 bytes per PosVel)
            // // Writing elements one-by-one can result in a tightly packed 32-byte layout (Rust repr(C)) instead of the WGSL std430 stride (48),
            // // causing all but the first particle to appear with zero size. A single slice write preserves proper padding per element.
            // let mut particle_storage = StorageBuffer::new(Vec::new());
            // particle_storage
            //     .write(&points[..])
            //     .expect("serialize particles slice");
            // let particles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //     label: Some("particles"),
            //     contents: particle_storage.as_ref(),
            //     usage: wgpu::BufferUsages::STORAGE,
            // });
            let particles_buffer = device.create_buffer(&BufferDescriptor{
                label: Some("particles"),
                size: 80*TARGET_NUM_PARTICLES as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Create initial uniform buffer with placeholder data
            let uniforms = sphere::RenderUniforms {
                inv_projection_matrix: glam::Mat4::IDENTITY,
                projection_matrix: glam::Mat4::IDENTITY,
                view_matrix: glam::Mat4::IDENTITY,
                inv_view_matrix: glam::Mat4::IDENTITY,
                texel_size: glam::Vec2::new(1.0 / size.width as f32, 1.0 / size.height as f32),
                sphere_size: 1.0,
            };
            // Use encase to create initial uniform buffer contents
            let mut ubuf = UniformBuffer::new(Vec::new());
            ubuf.write(&uniforms).expect("serialize uniforms");
            let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: ubuf.as_ref(),
                usage: wgpu::BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

            // Initialize start time
            self.start_time = Some(std::time::Instant::now());

            // Update uniforms with proper perspective matrices
            if let Some(start_time) = self.start_time {
                let elapsed = start_time.elapsed().as_secs_f32();
                self.update_uniforms(size, elapsed);
            }

            // Create stretch strength uniform buffer
            let stretch_strength = 2.5f32;
            // Single f32 uniform (stretch strength) serialized via encase UniformBuffer for consistency
            let mut stretch_encase = UniformBuffer::new(Vec::new());
            stretch_encase.write(&stretch_strength).expect("serialize stretch strength");
            let stretch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stretch-strength"),
                contents: stretch_encase.as_ref(),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Create bind group using the generated helper
            let bind_group = sphere::bind_groups::BindGroup0::from_bindings(
                &device,
                sphere::bind_groups::BindGroupLayout0 {
                    particles: particles_buffer.as_entire_buffer_binding(),
                    uniforms: uniforms_buffer.as_entire_buffer_binding(),
                    stretchStrength: stretch_buffer.as_entire_buffer_binding(),
                },
            );

            // Use generated hello_triangle helpers
            let shader = sphere::create_shader_module(&device);
            let pipeline_layout = sphere::create_pipeline_layout(&device);

            let caps = surface.get_capabilities(&adapter);
            let format = caps.formats[0];

            let vs = sphere::vs_entry();
            let fs = sphere::fs_entry(
                [
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // Blendable format for depth buffer
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            );

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sphere-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: sphere::vertex_state(&shader, &vs),
                fragment: Some(sphere::fragment_state(&shader, &fs)),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let config = surface
                .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                .expect("surface config");
            surface.configure(&device, &config);

            // Create auxiliary render target for second color attachment (Rgba16Float)
            let aux_rt_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("aux-rt-texture"),
                size: wgpu::Extent3d {
                    width: config.width.max(1),
                    height: config.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let aux_rt_view = aux_rt_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create depth texture
            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth-texture"),
                size: wgpu::Extent3d {
                    width: config.width.max(1),
                    height: config.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            self.pipeline = Some(pipeline);
            self.config = Some(config);
            // Create MLS-MPM solver before moving device/queue into self
            let solver = solver::MLSMPMSolver::new(&device, &particles_buffer, 10_000);
            self.solver = Some(solver);
            self.queue = Some(queue.clone());
            self.device = Some(device.clone());
            self.adapter = Some(adapter);
            self.surface = Some(surface);
            self.window = Some(window);
            // Persist resources
            self.bind_group = Some(bind_group);
            self.particles_buffer = Some(particles_buffer);
            self.uniforms_buffer = Some(uniforms_buffer);
            self.stretch_buffer = Some(stretch_buffer);
            self.aux_rt_view = Some(aux_rt_view);
            self.aux_rt_texture = Some(aux_rt_texture);
            self.depth_view = Some(depth_view);
            self.depth_texture = Some(depth_texture);
            // Create MLS-MPM solver (reuse particles buffer for rendering as destination of copyPosition)
            // solver already created above
        }

        // Kick off first frame
        if let Some(w) = self.window {
            w.request_redraw()
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(surface), Some(device), Some(config), Some(window)) = (
                    self.surface.as_ref(),
                    self.device.as_ref(),
                    self.config.as_mut(),
                    self.window.as_ref(),
                ) {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(device, config);
                    // Recreate auxiliary render target to match new size
                    let aux_rt_texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("aux-rt-texture"),
                        size: wgpu::Extent3d {
                            width: config.width.max(1),
                            height: config.height.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let aux_rt_view =
                        aux_rt_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    self.aux_rt_view = Some(aux_rt_view);
                    self.aux_rt_texture = Some(aux_rt_texture);
                    // Recreate depth texture to match new size
                    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("depth-texture"),
                        size: wgpu::Extent3d {
                            width: config.width.max(1),
                            height: config.height.max(1),
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    });
                    let depth_view =
                        depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    self.depth_view = Some(depth_view);
                    self.depth_texture = Some(depth_texture);
                    // Update uniforms with new aspect ratio and rotation
                    if let Some(start_time) = self.start_time {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        self.update_uniforms(new_size, elapsed);
                    }
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Update uniforms with rotation
                if let (Some(config), Some(start_time)) = (self.config.as_ref(), self.start_time) {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    self.update_uniforms(
                        winit::dpi::PhysicalSize::new(config.width, config.height),
                        elapsed,
                    );
                }

                // Early return if we don't have all required components
                let (surface, device, queue, pipeline, bind_group, aux_rt_view, depth_view) = match (
                    self.surface.as_ref(),
                    self.device.as_ref(),
                    self.queue.as_ref(),
                    self.pipeline.as_ref(),
                    self.bind_group.as_ref(),
                    self.aux_rt_view.as_ref(),
                    self.depth_view.as_ref(),
                ) {
                    (Some(s), Some(d), Some(q), Some(p), Some(bg), Some(av), Some(dv)) => {
                        (s, d, q, p, bg, av, dv)
                    }
                    _ => return,
                };

                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                        // On Outdated/Lost errors, reconfigure and skip frame
                        if let (Some(device), Some(config)) =
                            (self.device.as_ref(), self.config.as_ref())
                        {
                            surface.configure(device, config);
                        }
                        return;
                    }
                    Err(_) => return,
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render-encoder"),
                });

                // Run simulation step (advance physics & write into renderer particle buffer)
                if let (Some(solver), Some(queue)) = (self.solver.as_mut(), self.queue.as_ref()) {
                    solver.execute(device, queue, &mut encoder, TARGET_NUM_PARTICLES);
                }

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("rpass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: aux_rt_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rpass.set_pipeline(pipeline);
                    // Set required bind group(s)
                    sphere::set_bind_groups(&mut rpass, bind_group);
                    // Draw all 1000 particles as quads (6 vertices per instance)
                    rpass.draw(0..6, 0..TARGET_NUM_PARTICLES);
                }

                queue.submit(Some(encoder.finish()));
                #[cfg(windows)]
                if let Some(w) = self.window {
                    w.pre_present_notify()
                }
                frame.present();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuously redraw
        if let Some(w) = self.window {
            w.request_redraw()
        }
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
