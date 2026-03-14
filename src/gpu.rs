//! GPU compute backend support for swarm inference.
//!
//! This module provides a pluggable backend abstraction for GPU-backed tensor
//! computation. It currently supports a `wgpu` backend (cross-platform) and an
//! optional CUDA backend (when built with `--features cuda`).

use anyhow::{Context, Result};
use tokio::sync::oneshot;
use wgpu::util::DeviceExt;

/// Matrix-vector multiplication interface for GPU compute.
pub trait GpuCompute: Send + Sync {
    /// Computes `out = weights * input + bias`.
    ///
    /// - `weights` is stored row-major as `out_dim x in_dim`.
    /// - `input` is length `in_dim`.
    /// - `bias` is length `out_dim`.
    fn matmul(
        &self,
        weights: &[f32],
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
        bias: &[f32],
    ) -> Result<Vec<f32>>;
}

/// Selects the active GPU backend.
///
/// Default is `Wgpu` unless `SWARM_GPU_BACKEND=cuda` is set and CUDA feature is enabled.
pub fn create_gpu_backend() -> Result<Box<dyn GpuCompute>> {
    let backend = std::env::var("SWARM_GPU_BACKEND").unwrap_or_else(|_| "wgpu".to_string());

    match backend.as_str() {
        "wgpu" => Ok(Box::new(WgpuBackend::new()?)),
        "cuda" => {
            if cfg!(feature = "cuda") {
                Ok(Box::new(CudaBackend::new()?))
            } else {
                anyhow::bail!(
                    "CUDA backend was requested but crate was not built with --features cuda"
                )
            }
        }
        other => anyhow::bail!("Unsupported GPU backend: {}", other),
    }
}

// ---------------------------------------------------------
// WGPU Backend
// ---------------------------------------------------------

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuBackend {
    pub fn new() -> Result<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .context("Failed to request wgpu adapter")?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("swarm_gpu_device"),
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::downlevel_defaults(),
                    },
                    None,
                )
                .await
                .context("Failed to request wgpu device")?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "matmul.wgsl"
                ))),
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("matmul_bind_group_layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                bind_group_layout,
            })
        })
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulUniforms {
    in_dim: u32,
    out_dim: u32,
}

impl GpuCompute for WgpuBackend {
    fn matmul(
        &self,
        weights: &[f32],
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
        bias: &[f32],
    ) -> Result<Vec<f32>> {
        let weight_bytes = bytemuck::cast_slice(weights);
        let input_bytes = bytemuck::cast_slice(input);
        let bias_bytes = bytemuck::cast_slice(bias);

        let weight_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weights_buffer"),
                contents: weight_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input_buffer"),
                contents: input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bias_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bias_buffer"),
                contents: bias_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: (out_dim * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create a readback buffer that can be mapped for reading on the CPU.
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_readback"),
            size: (out_dim * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = MatMulUniforms {
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("matmul_bind_group"),
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = ((out_dim as f32) / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy the output buffer into the readback buffer so it can be mapped.
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &readback_buffer,
            0,
            (out_dim * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read back results from the readback buffer
        let buffer_slice = readback_buffer.slice(..);
        type BufferResult = Result<(), wgpu::BufferAsyncError>;
        let (tx, rx): (
            oneshot::Sender<BufferResult>,
            oneshot::Receiver<BufferResult>,
        ) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        pollster::block_on(rx).context("wgpu map_async channel closed")??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buffer.unmap();

        Ok(result)
    }
}

// ---------------------------------------------------------
// CUDA backend (optional)
// ---------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use cust::prelude::*;

    pub struct CudaBackend {
        _context: Context,
        module: Module,
        stream: Stream,
    }

    impl CudaBackend {
        pub fn new() -> Result<Self> {
            let _context = cust::quick_init().context("Failed to initialize CUDA")?;
            let ptx = include_str!("cuda_matmul.ptx");
            let module = Module::from_ptx(ptx, &[])?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            Ok(Self {
                _context,
                module,
                stream,
            })
        }
    }

    impl GpuCompute for CudaBackend {
        fn matmul(
            &self,
            weights: &[f32],
            input: &[f32],
            in_dim: usize,
            out_dim: usize,
            bias: &[f32],
        ) -> Result<Vec<f32>> {
            let mut output = vec![0.0f32; out_dim];

            let weights_buf = DeviceBuffer::from_slice(weights)?;
            let input_buf = DeviceBuffer::from_slice(input)?;
            let bias_buf = DeviceBuffer::from_slice(bias)?;
            let mut output_buf = DeviceBuffer::from_slice(&output)?;

            let func = self
                .module
                .get_function("matmul_vector")
                .context("CUDA kernel missing")?;
            let args = (
                weights_buf.as_device_ptr(),
                input_buf.as_device_ptr(),
                bias_buf.as_device_ptr(),
                output_buf.as_device_ptr(),
                in_dim as i32,
                out_dim as i32,
            );

            unsafe {
                launch!(func<<<out_dim as u32, 1, 0, self.stream>>> (args))?;
            }

            self.stream.synchronize()?;
            output_buf.copy_to(&mut output)?;
            Ok(output)
        }
    }

    pub use CudaBackend;
}

#[cfg(not(feature = "cuda"))]
mod cuda {
    use super::*;

    pub struct CudaBackend;

    impl CudaBackend {
        pub fn new() -> Result<Self> {
            anyhow::bail!("CUDA backend is not enabled. Build with --features cuda")
        }
    }

    impl GpuCompute for CudaBackend {
        fn matmul(
            &self,
            _weights: &[f32],
            _input: &[f32],
            _in_dim: usize,
            _out_dim: usize,
            _bias: &[f32],
        ) -> Result<Vec<f32>> {
            anyhow::bail!("CUDA backend is not enabled. Build with --features cuda")
        }
    }
}

use cuda::CudaBackend;

// Keep `WgpuBackend` publicly available.
