// Minimal standalone fluid simulation test with visualization
// Run with: cargo run --bin fluid_test

use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{PhysicalKey, KeyCode},
};

const GRID_SIZE: u32 = 128;
const GRID_CELLS: usize = (GRID_SIZE * GRID_SIZE) as usize;

fn rgba8_to_rgba16f_bytes(rgba8: &[u8]) -> Vec<u8> {
    debug_assert!(rgba8.len() % 4 == 0);
    let mut out = Vec::with_capacity(rgba8.len() * 2);
    for &b in rgba8 {
        let v = (b as f32) * (1.0 / 255.0);
        let h = f16::from_f32(v).to_bits().to_le_bytes();
        out.push(h[0]);
        out.push(h[1]);
    }
    out
}

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
    pressure_a: wgpu::Buffer,
    pressure_b: wgpu::Buffer,
    divergence: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    visualize_params_buffer: wgpu::Buffer,

    // Display/feedback texture ping-pong (advected each frame)
    display_texture_a: wgpu::Texture,
    display_texture_a_view: wgpu::TextureView,
    display_texture_b: wgpu::Texture,
    display_texture_b_view: wgpu::TextureView,
    display_sampler_linear: wgpu::Sampler,
    display_sampler_nearest: wgpu::Sampler,

    // Pipelines
    generate_forces_pipeline: wgpu::ComputePipeline,
    add_forces_pipeline: wgpu::ComputePipeline,
    advect_velocity_pipeline: wgpu::ComputePipeline,
    diffuse_velocity_pipeline: wgpu::ComputePipeline,
    advect_display_texture_pipeline: wgpu::ComputePipeline,
    enforce_boundaries_pipeline: wgpu::ComputePipeline,
    divergence_pipeline: wgpu::ComputePipeline,
    vorticity_confinement_pipeline: wgpu::ComputePipeline,
    clear_pressure_pipeline: wgpu::ComputePipeline,
    jacobi_pressure_pipeline: wgpu::ComputePipeline,
    subtract_gradient_pipeline: wgpu::ComputePipeline,
    copy_pipeline: wgpu::ComputePipeline,
    clear_velocity_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    // Bind groups
    bind_group_ab: wgpu::BindGroup, // velocity_a -> velocity_b, pressure_a -> pressure_b
    bind_group_ba: wgpu::BindGroup, // velocity_b -> velocity_a, pressure_b -> pressure_a
    // group(1) bind groups for display texture advection (A->B, B->A)
    display_tex_advect_ab: wgpu::BindGroup,
    display_tex_advect_ba: wgpu::BindGroup,

    // Render bind groups (sample either A or B)
    render_bind_group_a: wgpu::BindGroup,
    render_bind_group_b: wgpu::BindGroup,

    // State
    time: f32,
    render_mode: u32,
    mouse_down: bool,
    mouse_pos: [f32; 2],
    prev_mouse_pos: [f32; 2],

    // Which display texture currently contains the latest advected frame.
    display_texture_is_a: bool,

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
            decay: 0.999,
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
                wgpu::BindGroupEntry { binding: 4, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: divergence.as_entire_binding() },
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
                wgpu::BindGroupEntry { binding: 4, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: divergence.as_entire_binding() },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Display texture bind group layout (group 1)
        let display_tex_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layout for display texture advection: uses group(0) buffers + group(1) textures.
        let advect_display_tex_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Advect Display Texture Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &display_tex_bind_group_layout],
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

        let diffuse_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse Velocity"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "diffuse_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let advect_display_texture_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Advect Display Texture"),
            layout: Some(&advect_display_tex_pipeline_layout),
            module: &shader,
            entry_point: "advect_display_texture",
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Seed display texture A from assets/texture.png (fallback: generated checkerboard), resized to GRID_SIZE.
        let seed_rgba8: Vec<u8> = match image::open("assets/texture.png") {
            Ok(img) => {
                let resized = img.resize_exact(
                    GRID_SIZE,
                    GRID_SIZE,
                    image::imageops::FilterType::Triangle,
                );
                resized.to_rgba8().into_raw()
            }
            Err(_) => {
                let w = GRID_SIZE;
                let h = GRID_SIZE;
                let mut data = vec![0u8; (w * h * 4) as usize];
                for y in 0..h {
                    for x in 0..w {
                        let check = ((x / 8) ^ (y / 8)) & 1;
                        let v = if check == 0 { 40u8 } else { 220u8 };
                        let i = ((y * w + x) * 4) as usize;
                        data[i + 0] = v;
                        data[i + 1] = v;
                        data[i + 2] = v;
                        data[i + 3] = 255;
                    }
                }
                data
            }
        };

        let seed_rgba16f = rgba8_to_rgba16f_bytes(&seed_rgba8);

        let display_tex_desc = wgpu::TextureDescriptor {
            label: Some("Display Texture (Feedback)"),
            size: wgpu::Extent3d {
                width: GRID_SIZE,
                height: GRID_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let display_texture_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Display Texture A"),
            ..display_tex_desc
        });
        let display_texture_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Display Texture B"),
            ..display_tex_desc
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &display_texture_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &seed_rgba16f,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8 * GRID_SIZE),
                rows_per_image: Some(GRID_SIZE),
            },
            wgpu::Extent3d {
                width: GRID_SIZE,
                height: GRID_SIZE,
                depth_or_array_layers: 1,
            },
        );

        // Initialize B to black.
        let black = vec![0u8; (GRID_SIZE * GRID_SIZE * 8) as usize];
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &display_texture_b,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &black,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8 * GRID_SIZE),
                rows_per_image: Some(GRID_SIZE),
            },
            wgpu::Extent3d {
                width: GRID_SIZE,
                height: GRID_SIZE,
                depth_or_array_layers: 1,
            },
        );

        let display_texture_a_view = display_texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let display_texture_b_view = display_texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        let display_sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler (Linear)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let display_sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler (Nearest)"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Bind groups for display texture advection (group 1)
        let display_tex_advect_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Texture Advect A->B"),
            layout: &display_tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_texture_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&display_sampler_nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&display_texture_b_view),
                },
            ],
        });
        let display_tex_advect_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Texture Advect B->A"),
            layout: &display_tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_texture_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&display_sampler_nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&display_texture_a_view),
                },
            ],
        });

        // Render bind groups: velocity_a is always current; texture varies (A vs B).
        let render_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group (Tex A)"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: visualize_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&display_texture_a_view),
                },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&display_sampler_linear) },
            ],
        });
        let render_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group (Tex B)"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: visualize_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&display_texture_b_view),
                },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&display_sampler_linear) },
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
            pressure_a,
            pressure_b,
            divergence,
            params_buffer,
            visualize_params_buffer,
            display_texture_a,
            display_texture_a_view,
            display_texture_b,
            display_texture_b_view,
            display_sampler_linear,
            display_sampler_nearest,
            generate_forces_pipeline,
            add_forces_pipeline,
            advect_velocity_pipeline,
            diffuse_velocity_pipeline,
            advect_display_texture_pipeline,
            enforce_boundaries_pipeline,
            divergence_pipeline,
            vorticity_confinement_pipeline,
            clear_pressure_pipeline,
            jacobi_pressure_pipeline,
            subtract_gradient_pipeline,
            copy_pipeline,
            clear_velocity_pipeline,
            render_pipeline,
            bind_group_ab,
            bind_group_ba,
            display_tex_advect_ab,
            display_tex_advect_ba,
            render_bind_group_a,
            render_bind_group_b,
            time: 0.0,
            render_mode: 0,
            mouse_down: false,
            mouse_pos: [f32::NAN; 2],
            prev_mouse_pos: [f32::NAN; 2],
            display_texture_is_a: true,
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

    fn update(&mut self, dt_real: f32) {
        // Frame-based simulation: fixed dt per frame (ignores real elapsed time).
        // This makes the simulation deterministic with respect to frame count.
        let dt_sim: f32 = 1.0 / 60.0;
        self.time += dt_sim;

        // Track performance using real time between frames.
        self.frame_count += 1;
        self.frame_time_sum += dt_real;
        // Mouse velocity should feel consistent in *real time*; otherwise at high FPS
        // the per-frame cursor delta becomes tiny and force injection effectively vanishes.
        let dt_input = dt_real.clamp(1.0 / 240.0, 1.0 / 30.0);
        let mut mouse_vel = [0.0f32, 0.0f32];
        if self.mouse_down && self.prev_mouse_pos[0].is_finite() {
            mouse_vel[0] = (self.mouse_pos[0] - self.prev_mouse_pos[0]) / dt_input;
            mouse_vel[1] = (self.mouse_pos[1] - self.prev_mouse_pos[1]) / dt_input;
        }
        self.prev_mouse_pos = self.mouse_pos;

        let params = FluidParams {
            time: self.time,
            dt: dt_sim,
            // Higher = less damping per frame.
            decay: 0.9995,
            grid_size: GRID_SIZE,
            mouse: [self.mouse_pos[0], self.mouse_pos[1], mouse_vel[0], mouse_vel[1]],
            // splat = [radius_cells, force_scale, vorticity_strength, mouse_down]
            // Keep this small; high values can re-introduce grid-scale static.
            splat: [10.0, 15.0, 4.0, if self.mouse_down { 1.0 } else { 0.0 }],
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
            // 5b. Diffuse velocity (A -> B) to damp 1-cell noise
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Diffuse Velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.diffuse_velocity_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 5c. Copy back so the rest of the pipeline reads velocity_a
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Copy Velocity (Post-Diffuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
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
                label: Some("Enforce Boundaries (Post-Project)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.enforce_boundaries_pipeline);
            pass.set_bind_group(0, bg_ba, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        {
            // 11. Advect the display texture using the final velocity field (feedback loop).
            // Always bind group(0) as AB so velocity_in is velocity_a.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Advect Display Texture"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.advect_display_texture_pipeline);
            pass.set_bind_group(0, bg_ab, &[]);
            let tex_bg = if self.display_texture_is_a {
                &self.display_tex_advect_ab
            } else {
                &self.display_tex_advect_ba
            };
            pass.set_bind_group(1, tex_bg, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        // Flip which texture is current (A->B or B->A).
        self.display_texture_is_a = !self.display_texture_is_a;

        self.queue.submit(std::iter::once(encoder.finish()));
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
            let bg = if self.display_texture_is_a {
                &self.render_bind_group_a
            } else {
                &self.render_bind_group_b
            };
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
                    // Render modes:
                    // 1=texture (feedback), 2=velocity Y, 3=speed, 4=pressure, 5=divergence, 6=vorticity
                    match key {
                        KeyCode::Digit1 => fluid.set_render_mode(0),
                        KeyCode::Digit2 => fluid.set_render_mode(1),
                        KeyCode::Digit3 => fluid.set_render_mode(2),
                        KeyCode::Digit4 => fluid.set_render_mode(3),
                        KeyCode::Digit5 => fluid.set_render_mode(4),
                        KeyCode::Digit6 => fluid.set_render_mode(5),
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
