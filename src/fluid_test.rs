// Minimal standalone fluid simulation test with visualization
// Run with: cargo run --bin fluid_test

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{PhysicalKey, KeyCode},
};

const GRID_SIZE: u32 = 128;
const GRID_CELLS: usize = (GRID_SIZE * GRID_SIZE) as usize;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FluidParams {
    time: f32,
    dt: f32,
    decay: f32,
    grid_size: u32,
    mouse: [f32; 4],
    splat: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VisualizeParams {
    grid_size: u32,
    mode: u32,
    _pad: [u32; 2],
    scale: [f32; 4],
}

struct FluidState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    // Buffers
    velocity_a: wgpu::Buffer,
    velocity_b: wgpu::Buffer,
    forces: wgpu::Buffer,
    dye_a: wgpu::Buffer,
    dye_b: wgpu::Buffer,
    pressure_a: wgpu::Buffer,
    pressure_b: wgpu::Buffer,
    divergence: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    visualize_params_buffer: wgpu::Buffer,
    
    // Pipelines
    generate_forces_pipeline: wgpu::ComputePipeline,
    add_forces_pipeline: wgpu::ComputePipeline,
    advect_velocity_pipeline: wgpu::ComputePipeline,
    enforce_boundaries_pipeline: wgpu::ComputePipeline,
    advect_dye_pipeline: wgpu::ComputePipeline,
    divergence_pipeline: wgpu::ComputePipeline,
    vorticity_confinement_pipeline: wgpu::ComputePipeline,
    clear_pressure_pipeline: wgpu::ComputePipeline,
    jacobi_pressure_pipeline: wgpu::ComputePipeline,
    subtract_gradient_pipeline: wgpu::ComputePipeline,
    copy_pipeline: wgpu::ComputePipeline,
    clear_velocity_pipeline: wgpu::ComputePipeline,
    clear_dye_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    
    // Bind groups
    bind_group_ab: wgpu::BindGroup, // velocity_a -> velocity_b, pressure_a -> pressure_b
    bind_group_ba: wgpu::BindGroup, // velocity_b -> velocity_a, pressure_b -> pressure_a
    dye_bind_group_ab: wgpu::BindGroup, // dye_a -> dye_b (uses velocity_a)
    dye_bind_group_ba: wgpu::BindGroup, // dye_b -> dye_a (uses velocity_a)
    render_bind_group_a: wgpu::BindGroup,
    render_bind_group_b: wgpu::BindGroup,
    
    // State
    dye_ping: bool, // true = dye_a is current
    time: f32,
    render_mode: u32,
    mouse_down: bool,
    mouse_pos: [f32; 2],
    prev_mouse_pos: [f32; 2],
    
    // Performance tracking
    frame_count: u32,
    frame_time_sum: f32,
    last_fps_update: std::time::Instant,
}

impl FluidState {
    async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        // Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(window.clone()).unwrap();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Fluid Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        
        // Create buffers
        let velocity_size = (GRID_CELLS * std::mem::size_of::<[f32; 2]>()) as u64;
        let dye_size = (GRID_CELLS * std::mem::size_of::<[f32; 4]>()) as u64;
        let scalar_size = (GRID_CELLS * std::mem::size_of::<f32>()) as u64;
        
        let velocity_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity A"),
            size: velocity_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let velocity_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity B"),
            size: velocity_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let forces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Forces"),
            size: velocity_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let dye_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dye A"),
            size: dye_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dye_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dye B"),
            size: dye_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pressure_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure A"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pressure_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure B"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let divergence = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Divergence"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let params = FluidParams {
            time: 0.0,
            dt: 0.016,
            decay: 0.995,
            grid_size: GRID_SIZE,
            mouse: [0.0; 4],
            splat: [0.0; 4],
        };
        
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let visualize_params = VisualizeParams {
            grid_size: GRID_SIZE,
            mode: 0,
            _pad: [0, 0],
            // velocity, pressure, divergence, vorticity scales
            scale: [40.0, 0.5, 2.0, 0.25],
        };
        
        let visualize_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Visualize Params"),
            contents: bytemuck::cast_slice(&[visualize_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Configure surface
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate, // Uncapped framerate
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
        
        // Load shader
        let shader_source = std::fs::read_to_string("shaders/fluid.wgsl").unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind groups (velocity always advects A->B then projects B->A)
        let bind_group_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group AB"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: divergence.as_entire_binding() },
            ],
        });

        let bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group BA"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                // unused for velocity projection, but required by layout
                wgpu::BindGroupEntry { binding: 4, resource: dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: divergence.as_entire_binding() },
            ],
        });



        // Dye ping-pong bind groups (always use velocity_a as the advecting field)
        let dye_bind_group_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dye Bind Group AB"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: divergence.as_entire_binding() },
            ],
        });

        let dye_bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dye Bind Group BA"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: divergence.as_entire_binding() },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipelines
        let generate_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generate Forces"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "generate_test_forces",
            compilation_options: Default::default(),
            cache: None,
        });
        
        let add_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Forces"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "add_forces",
            compilation_options: Default::default(),
            cache: None,
        });

        let advect_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Advect Velocity"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "advect_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let enforce_boundaries_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Enforce Boundaries"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "enforce_boundaries",
            compilation_options: Default::default(),
            cache: None,
        });        let advect_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Advect Dye"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "advect_dye",
            compilation_options: Default::default(),
            cache: None,
        });

        let divergence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Divergence"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "compute_divergence",
            compilation_options: Default::default(),
            cache: None,
        });

        let vorticity_confinement_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vorticity Confinement"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "vorticity_confinement",
            compilation_options: Default::default(),
            cache: None,
        });

        let clear_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Pressure"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_pressure",
            compilation_options: Default::default(),
            cache: None,
        });

        let jacobi_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Jacobi Pressure"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "jacobi_pressure",
            compilation_options: Default::default(),
            cache: None,
        });

        let subtract_gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Subtract Gradient"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "subtract_gradient",
            compilation_options: Default::default(),
            cache: None,
        });
        
        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Copy"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "copy_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let clear_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Velocity"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let clear_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Dye"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_dye",
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Load render shader
        let render_shader_source = std::fs::read_to_string("shaders/fluid_visualize.wgsl").unwrap();
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(render_shader_source.into()),
        });
        
        // Create render bind group layout
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Render bind groups: velocity_a is always current; dye ping-pongs.
        let render_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group (dye_a)"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: visualize_params_buffer.as_entire_binding() },
            ],
        });

        let render_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group (dye_b)"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: visualize_params_buffer.as_entire_binding() },
            ],
        });
        
        // Create render pipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        // Clear buffers to a known state.
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fluid Init Clear"),
            });
            let workgroups = (GRID_SIZE + 15) / 16;

            for bg in [&bind_group_ab, &bind_group_ba] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Clear Velocity Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&clear_velocity_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups, workgroups, 1);
            }

            for bg in [&dye_bind_group_ab, &dye_bind_group_ba] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Clear Dye Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&clear_dye_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups, workgroups, 1);
            }

            for bg in [&bind_group_ab, &bind_group_ba] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Clear Pressure Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&clear_pressure_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(workgroups, workgroups, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        Self {
            device,
            queue,
            surface,
            surface_config,
            velocity_a,
            velocity_b,
            forces,
            dye_a,
            dye_b,
            pressure_a,
            pressure_b,
            divergence,
            params_buffer,
            visualize_params_buffer,
            generate_forces_pipeline,
            add_forces_pipeline,
            advect_velocity_pipeline,
            enforce_boundaries_pipeline,
            advect_dye_pipeline,
            divergence_pipeline,
            vorticity_confinement_pipeline,
            clear_pressure_pipeline,
            jacobi_pressure_pipeline,
            subtract_gradient_pipeline,
            copy_pipeline,
            clear_velocity_pipeline,
            clear_dye_pipeline,
            render_pipeline,
            bind_group_ab,
            bind_group_ba,
            dye_bind_group_ab,
            dye_bind_group_ba,
            render_bind_group_a,
            render_bind_group_b,
            dye_ping: true,
            time: 0.0,
            render_mode: 0,
            mouse_down: false,
            mouse_pos: [f32::NAN; 2],
            prev_mouse_pos: [f32::NAN; 2],
            frame_count: 0,
            frame_time_sum: 0.0,
            last_fps_update: std::time::Instant::now(),
        }
    }

    fn set_mouse(&mut self, down: bool, pos_grid: [f32; 2]) {
        self.mouse_down = down;
        self.mouse_pos = pos_grid;
        if !self.prev_mouse_pos[0].is_finite() {
            self.prev_mouse_pos = pos_grid;
        }
    }

    fn set_render_mode(&mut self, mode: u32) {
        self.render_mode = mode;
    }
    
    fn get_fps_and_frame_time(&mut self) -> Option<(f32, f32)> {
        let elapsed = self.last_fps_update.elapsed();
        if elapsed.as_secs_f32() >= 0.5 {
            if self.frame_count > 0 {
                let fps = self.frame_count as f32 / elapsed.as_secs_f32();
                let avg_frame_time_ms = (self.frame_time_sum / self.frame_count as f32) * 1000.0;
                self.frame_count = 0;
                self.frame_time_sum = 0.0;
                self.last_fps_update = std::time::Instant::now();
                return Some((fps, avg_frame_time_ms));
            }
        }
        None
    }
    
    fn update_window_title(&mut self, window: &winit::window::Window) {
        if let Some((fps, frame_time_ms)) = self.get_fps_and_frame_time() {
            let title = format!(
                "Fluid Simulation Test - 128x128 | {:.0} FPS | {:.2} ms/frame",
                fps, frame_time_ms
            );
            window.set_title(&title);
        }
    }
    
    fn update(&mut self, dt: f32) {
        self.time += dt;

        // Track performance
        self.frame_count += 1;
        self.frame_time_sum += dt;

        let dt = dt.clamp(1.0 / 240.0, 1.0 / 30.0);
        let mut mouse_vel = [0.0f32, 0.0f32];
        if self.mouse_down && self.prev_mouse_pos[0].is_finite() {
            mouse_vel[0] = (self.mouse_pos[0] - self.prev_mouse_pos[0]) / dt;
            mouse_vel[1] = (self.mouse_pos[1] - self.prev_mouse_pos[1]) / dt;
        }
        self.prev_mouse_pos = self.mouse_pos;
        
        let params = FluidParams {
            time: self.time,
            dt,
            decay: 0.995,
            grid_size: GRID_SIZE,
            mouse: [self.mouse_pos[0], self.mouse_pos[1], mouse_vel[0], mouse_vel[1]],
            // radius, force, dye amount, mouse_down
            splat: [10.0, 1.5, 45.0, if self.mouse_down { 1.0 } else { 0.0 }],
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        let visualize_params = VisualizeParams {
            grid_size: GRID_SIZE,
            mode: self.render_mode,
            _pad: [0, 0],
            scale: [40.0, 0.5, 2.0, 0.25],
        };
        self.queue
            .write_buffer(&self.visualize_params_buffer, 0, bytemuck::cast_slice(&[visualize_params]));
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fluid Update"),
        });
        
        let workgroups = (GRID_SIZE + 15) / 16;
        let bg_ab = &self.bind_group_ab;
        let bg_ba = &self.bind_group_ba;
        let dye_bg = if self.dye_ping { &self.dye_bind_group_ab } else { &self.dye_bind_group_ba };
        
        {
            // 1. Generate test forces
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generate Forces"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate_forces_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 2. Advect velocity (A -> B)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Advect Velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.advect_velocity_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 3. Add forces (B -> A)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Add Forces"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add_forces_pipeline);
            pass.set_bind_group(0, bg_ba, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 4. Vorticity confinement (A -> B)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vorticity Confinement"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vorticity_confinement_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 5. Enforce boundaries (B -> A)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Enforce Boundaries"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.enforce_boundaries_pipeline);
            pass.set_bind_group(0, bg_ba, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 6. Divergence of the velocity field (reads velocity_a)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Divergence"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divergence_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 7. Clear both pressure buffers
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Pressure A"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pressure_pipeline);
            pass.set_bind_group(0, bg_ba, &[]); // clears pressure_a
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Pressure B"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pressure_pipeline);
            pass.set_bind_group(0, bg_ab, &[]); // clears pressure_b
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        // 8. Jacobi iterations for pressure (odd count so final ends up in pressure_a)
        const JACOBI_ITERS: u32 = 31;
        for i in 0..JACOBI_ITERS {
            let bg = if (i & 1) == 0 { bg_ab } else { bg_ba };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Jacobi Pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.jacobi_pressure_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 9. Subtract pressure gradient (A->B, using pressure_a)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.subtract_gradient_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 10. Copy result back to A so rendering always uses velocity_a.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy B to A"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, bg_ba, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 11. Advect dye using the PROJECTED velocity field (velocity_a)
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Advect Dye"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.advect_dye_pipeline);
            pass.set_bind_group(0, dye_bg, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Toggle dye ping-pong only.
        self.dye_ping = !self.dye_ping;
    }
    
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            // If dye_ping is true, dye_a is current; otherwise dye_b is current.
            let bg = if self.dye_ping { &self.render_bind_group_a } else { &self.render_bind_group_b };
            render_pass.set_bind_group(0, bg, &[]);
            render_pass.draw(0..3, 0..1); // Full-screen triangle
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
    
    async fn read_velocity_field(&self) -> Vec<[f32; 2]> {
        // Velocity is always projected into `velocity_a` each frame.
        let current_buffer = &self.velocity_a;
        
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (GRID_CELLS * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read"),
        });
        encoder.copy_buffer_to_buffer(
            current_buffer,
            0,
            &staging_buffer,
            0,
            (GRID_CELLS * std::mem::size_of::<[f32; 2]>()) as u64,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.unwrap().unwrap();
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<[f32; 2]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        
        result
    }
}

fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let window = std::sync::Arc::new(event_loop.create_window(
        winit::window::WindowAttributes::default()
            .with_title("Fluid Simulation Test - 128x128")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 800)))
        .unwrap());
    
    let mut fluid = pollster::block_on(FluidState::new(window.clone()));
    let mut last_update = std::time::Instant::now();
    let mut cursor_pos_px: Option<(f32, f32)> = None;
    let mut mouse_down = false;
    
    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => control_flow.exit(),
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                    ..
                } => control_flow.exit(),
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(key),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    // Render modes like common fluid sims:
                    // 1 dye, 2 velocity, 3 pressure, 4 divergence, 5 vorticity
                    match key {
                        KeyCode::Digit1 => fluid.set_render_mode(0),
                        KeyCode::Digit2 => fluid.set_render_mode(1),
                        KeyCode::Digit3 => fluid.set_render_mode(2),
                        KeyCode::Digit4 => fluid.set_render_mode(3),
                        KeyCode::Digit5 => fluid.set_render_mode(4),
                        _ => {}
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    cursor_pos_px = Some((position.x as f32, position.y as f32));
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    mouse_down = *state == ElementState::Pressed;
                }
                WindowEvent::Resized(physical_size) => {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        fluid.surface_config.width = physical_size.width;
                        fluid.surface_config.height = physical_size.height;
                        fluid.surface.configure(&fluid.device, &fluid.surface_config);
                    }
                }
                WindowEvent::RedrawRequested => {
                    // Update simulation
                    let now = std::time::Instant::now();
                    let dt = (now - last_update).as_secs_f32();
                    last_update = now;

                    // Update mouse state in grid coordinates.
                    if let Some((mx, my)) = cursor_pos_px {
                        let w = fluid.surface_config.width.max(1) as f32;
                        let h = fluid.surface_config.height.max(1) as f32;
                        let gx = (mx / w) * GRID_SIZE as f32;
                        let gy = (my / h) * GRID_SIZE as f32;
                        fluid.set_mouse(mouse_down, [gx, gy]);
                    } else {
                        // If we don't have cursor info yet, keep mouse "up".
                        fluid.set_mouse(false, [GRID_SIZE as f32 * 0.5, GRID_SIZE as f32 * 0.5]);
                    }
                    
                    fluid.update(dt);
                    fluid.update_window_title(&window);
                    
                    // Render
                    match fluid.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            fluid.surface.configure(&fluid.device, &fluid.surface_config);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            control_flow.exit();
                        }
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
