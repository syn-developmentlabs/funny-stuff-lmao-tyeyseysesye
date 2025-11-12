// put this shit in ur cargo.toml:
// [dependencies]
// wgpu = "27.0"
// pollster = "0.4.0"
// bytemuck = { version = "1.14", features = ["derive"] }

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SearchParams {
    start_seed: u32,
    end_seed: u32,
    target_float: f32,
    tolerance: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SearchResult {
    found: u32,
    seed: u32,
    actual_value: f32,
    _padding: u32,
}

// kernel operation time babbbyy i hope ur pc explodes
const SHADER: &str = r#"
struct SearchParams {
    start_seed: u32,
    end_seed: u32,
    target_float: f32,
    tolerance: f32,
}

struct SearchResult {
    found: u32,
    seed: u32,
    actual_value: f32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: SearchParams;
@group(0) @binding(1) var<storage, read_write> result: SearchResult;

fn mt19937_init(seed: u32) -> array<u32, 624> {
    var mt: array<u32, 624>;
    mt[0] = seed;
    for (var i = 1u; i < 624u; i++) {
        let prev = mt[i - 1u]; 
        mt[i] = (1812433253u * (prev ^ (prev >> 30u)) + i);
    }
    return mt;
}

fn mt19937_twist(mt: ptr<function, array<u32, 624>>) {
    for (var i = 0u; i < 624u; i++) {
        let y = ((*mt)[i] & 0x80000000u) | ((*mt)[(i + 1u) % 624u] & 0x7fffffffu);
        (*mt)[i] = (*mt)[(i + 397u) % 624u] ^ (y >> 1u);
        if ((y & 1u) != 0u) {
            (*mt)[i] ^= 0x9908B0DFu;
        }
    }
}

fn mt19937_extract(mt: ptr<function, array<u32, 624>>, index: ptr<function, u32>) -> u32 {
    if (*index >= 624u) {
        mt19937_twist(mt);
        *index = 0u;
    }
    var y = (*mt)[*index];
    *index = *index + 1u;
    
    // xor ops
    y ^= (y >> 11u);
    y ^= (y << 7u) & 0x9D2C5680u;
    y ^= (y << 15u) & 0xEFC60000u;
    y ^= (y >> 18u);
    return y;
}

fn next_number(mt: ptr<function, array<u32, 624>>, index: ptr<function, u32>) -> f32 {
    let r = mt19937_extract(mt, index);
    return f32(r) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let seed = params.start_seed + global_id.x;
    
    if (seed > params.end_seed) {
        return;
    }
    
    // check if already exists
    if (result.found != 0u) {
        return;
    }
    
    var mt = mt19937_init(seed);
    var index = 624u;
    let value = next_number(&mt, &index);
    
    let diff = abs(value - params.target_float); // get abs or it'd fail
    if (diff <= params.tolerance) {
        // sort type shit
        if (result.found == 0u) {
            result.found = 1u;
            result.seed = seed;
            result.actual_value = value;
        }
    }
}
"#;

pub async fn bruteforce_seed(
    target: f32,
    tolerance: f32,
    start_seed: u32,
    end_seed: u32,
) -> Option<(u32, f32)> {
    const BATCH_SIZE: u32 = 256 * 65535;
    
    let mut current_start = start_seed;
    
    while current_start <= end_seed {
        let current_end = (current_start + BATCH_SIZE - 1).min(end_seed);
                
        if let Some(result) = bruteforce_seed_batch(target, tolerance, current_start, current_end).await { 
            return Some(result);
        }
        
        current_start = current_end + 1;
    }
    
    None
}

async fn bruteforce_seed_batch(
    target: f32,
    tolerance: f32,
    start_seed: u32,
    end_seed: u32,
) -> Option<(u32, f32)> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok()?;
    
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default()) // yes please israel gpt
        .await
        .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MT19937 Shader"), // it errors if i do not use Pascal Case, PascalCase lmao fucking nigger
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let params = SearchParams {
        start_seed,
        end_seed,
        target_float: target,
        tolerance,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_init = SearchResult {
        found: 0,
        seed: 0,
        actual_value: 0.0,
        _padding: 0,
    };

    let result_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Result Buffer"),
        contents: bytemuck::cast_slice(&[result_init]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: std::mem::size_of::<SearchResult>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // ???
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout], // borrow since memory shtuff
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        let workgroups = (((end_seed - start_seed + 1) + 255) / 256).min(65535);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &result_buffer,
        0,
        &staging_buffer,
        0,
        std::mem::size_of::<SearchResult>() as u64,
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    
    let mapping_result = Arc::new(Mutex::new(None));
    let mapping_result_clone = Arc::clone(&mapping_result);
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        *mapping_result_clone.lock().unwrap() = Some(result);
    });
    
    // PIM GOING OT FUJDFVVCBBBBBBBBBBBBBBBBBBFCJUNG JUKK NYSEKF IF THIS DOESNT WORK
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    }).ok()?;
    
    mapping_result.lock().unwrap().take()?.ok()?;

    let data = buffer_slice.get_mapped_range();
    let result: SearchResult = *bytemuck::from_bytes(&data); // rust type shit
    drop(data);
    staging_buffer.unmap();

    if result.found != 0 {
        Some((result.seed, result.actual_value))
    } else {
        None
    }
}

fn main() {
    let target = 0.9413137700091349;
    let tolerance = 0.0;
    let start_seed = 0;
    let end_seed = 100_000_000;
    
    println!("looking for -> {}", target);
    println!("r: ({} ... {})", start_seed, end_seed);
    
    match pollster::block_on(bruteforce_seed(target, tolerance, start_seed, end_seed)) {
        Some((seed, value)) => {
            println!("seed: {}", seed);
            println!("- actual eval value: {}", value);
        }
        None => {
            println!("invalid range");
        }
    }
}