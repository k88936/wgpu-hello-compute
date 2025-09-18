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
}


impl App {
    fn new() -> Self {
        Self::default()
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

            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            }))
            .expect("Failed to find an appropriate adapter");

            let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            }))
            .expect("Failed to create device");

            // Use generated hello_triangle helpers
            let shader = crate::hello_triangle::create_shader_module(&device);
            let pipeline_layout = crate::hello_triangle::create_pipeline_layout(&device);

            let caps = surface.get_capabilities(&adapter);
            let format = caps.formats[0];

            let vs = crate::hello_triangle::vs_main_entry();
            let fs = crate::hello_triangle::fs_main_entry([Some(format.into())]);

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hello-triangle"),
                layout: Some(&pipeline_layout),
                vertex: crate::hello_triangle::vertex_state(&shader, &vs),
                fragment: Some(crate::hello_triangle::fragment_state(&shader, &fs)),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let config = surface
                .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                .expect("surface config");
            surface.configure(&device, &config);

            self.pipeline = Some(pipeline);
            self.config = Some(config);
            self.queue = Some(queue);
            self.device = Some(device);
            self.adapter = Some(adapter);
            self.surface = Some(surface);
            self.window = Some(window);
        }

        // Kick off first frame
        if let Some(w) = self.window { w.request_redraw() }
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
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Early return if we don't have all required components
                let (surface, device, queue, pipeline) = match (
                    self.surface.as_ref(),
                    self.device.as_ref(),
                    self.queue.as_ref(),
                    self.pipeline.as_ref(),
                ) {
                    (Some(s), Some(d), Some(q), Some(p)) => (s, d, q, p),
                    _ => return,
                };

                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                        // On Outdated/Lost errors, reconfigure and skip frame
                        if let (Some(device), Some(config)) = (self.device.as_ref(), self.config.as_ref()) {
                            surface.configure(device, config);
                        }
                        return;
                    }
                    Err(_) => return,
                };

                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render-encoder"),
                });

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("rpass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rpass.set_pipeline(pipeline);
                    rpass.draw(0..3, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                #[cfg(windows)]
                if let Some(w) = self.window { w.pre_present_notify() }
                frame.present();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuously redraw
        if let Some(w) = self.window { w.request_redraw() }
    }
}

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
