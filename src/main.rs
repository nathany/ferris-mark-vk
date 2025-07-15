use anyhow::{Result, anyhow};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use std::collections::HashSet;
use std::env;
use std::mem::size_of;
use std::time::Instant;
use vulkanalia::prelude::v1_4::*;
use vulkanalia::vk::{KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{
    Device, Entry, Instance,
    loader::{LIBRARY, LibloadingLoader},
    window as vk_window,
};
use vulkanalia_vma::{self as vma, Alloc};
use winit::dpi::PhysicalSize;
use winit::event::{Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder};

// Error handling wrapper functions
fn map_entry_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow!("{}", error)
}

fn map_shader_error(error: shaderc::Error) -> anyhow::Error {
    anyhow!("Shader compilation failed: {}", error)
}

// Vulkan extensions required for rendering to a window surface (always required)
const REQUIRED_DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name, // Enables rendering to window surfaces
    vk::EXT_MEMORY_PRIORITY_EXTENSION.name, // Allows specifying memory allocation priorities
    vk::EXT_PAGEABLE_DEVICE_LOCAL_MEMORY_EXTENSION.name, // Enables memory paging optimizations
];

// Minimum Vulkan versions for our feature requirements
const MIN_VULKAN_VERSION: u32 = vk::make_version(1, 1, 0); // For VK_KHR_dynamic_rendering extension support
const PREFERRED_VULKAN_VERSION: u32 = vk::make_version(1, 3, 0); // VK_KHR_dynamic_rendering is core
const OPTIMAL_VULKAN_VERSION: u32 = vk::make_version(1, 4, 0); // VK_KHR_synchronization2 is core

// Number of frames we can work on simultaneously (prevents CPU waiting for GPU)
// Modern GPUs can process multiple frames concurrently while displaying others
const MAX_FRAMES_IN_FLIGHT: usize = 2;

// Logical resolution constants - our "virtual" resolution that sprites are positioned in
// This gets scaled to fit the actual window size while maintaining aspect ratio
const LOGICAL_WIDTH: f32 = 1920.0;
const LOGICAL_HEIGHT: f32 = 1080.0;

// Initial window size - the actual pixel dimensions when the window first opens
const INITIAL_WINDOW_WIDTH: u32 = 1920;
const INITIAL_WINDOW_HEIGHT: u32 = 1080;

// Physics simulation constants
const GRAVITY: f32 = 0.5; // Downward acceleration per frame
const BOUNCE_DAMPING: f32 = 0.90; // Energy loss when sprites hit the ground (0.0-1.0)

// Sprite rendering constants
const SPRITE_WIDTH: f32 = 99.0; // Width of each Ferris sprite in logical pixels
const SPRITE_HEIGHT: f32 = 70.0; // Height of each Ferris sprite in logical pixels

// Image and timing constants
const RGBA_BYTES_PER_PIXEL: u32 = 4; // 4 bytes per pixel for RGBA format
const TARGET_FPS: f32 = 60.0; // Target frame rate for physics scaling

// Sprite velocity constants for randomized movement
const MAX_INITIAL_HORIZONTAL_VELOCITY: f32 = 10.0; // Maximum initial horizontal velocity
const MIN_INITIAL_VERTICAL_VELOCITY: f32 = 5.0; // Minimum initial upward velocity
const MAX_INITIAL_VERTICAL_VELOCITY: f32 = 10.0; // Maximum initial upward velocity
const MAX_BOUNCE_BOOST: f32 = 9.0; // Maximum random upward boost on bounce

// Swapchain configuration constants
const SWAPCHAIN_IMAGE_ARRAY_LAYERS: u32 = 1; // Always 1 unless doing stereoscopic 3D
const MIN_SWAPCHAIN_IMAGE_COUNT_OFFSET: u32 = 1; // Request this many more than minimum for better performance

// Sprite spawning configuration
const SPAWN_SPRITES_IMMEDIATELY: bool = true; // If true, sprites spawn at startup. If false, wait for spacebar.

// Represents a single sprite in our physics simulation
#[derive(Clone, Debug)]
struct Sprite {
    pos: Vec2, // Current position in logical coordinates
    vel: Vec2, // Current velocity (pixels per frame)
}

// Creates the initial sprites with random positions and upward velocities
fn generate_sprites(count: usize) -> Vec<Sprite> {
    let mut sprites = Vec::new();

    for _i in 0..count {
        sprites.push(Sprite {
            pos: Vec2::new(
                fastrand::f32() * (LOGICAL_WIDTH - SPRITE_WIDTH),
                fastrand::f32() * (LOGICAL_HEIGHT - SPRITE_HEIGHT),
            ),
            vel: Vec2::new(
                (fastrand::f32() - 0.5) * MAX_INITIAL_HORIZONTAL_VELOCITY, // Random horizontal velocity
                fastrand::f32() * (MAX_INITIAL_VERTICAL_VELOCITY - MIN_INITIAL_VERTICAL_VELOCITY)
                    + MIN_INITIAL_VERTICAL_VELOCITY, // Random upward velocity
            ),
        });
    }

    sprites
}

// Convert sprites to sprite commands for GPU rendering
impl App {
    // Converts our sprite physics data into rendering commands for the GPU
    fn sprites_to_commands(&self) -> Vec<SpriteCommand> {
        let mut commands = Vec::new();

        for sprite in &self.data.sprites {
            // Create a 4x4 matrix that positions and scales the sprite for the GPU
            let transform = Mat4::from_translation(Vec3::new(sprite.pos.x, sprite.pos.y, 0.0))
                * Mat4::from_scale(Vec3::new(SPRITE_WIDTH, SPRITE_HEIGHT, 1.0));

            commands.push(SpriteCommand {
                transform: transform.to_cols_array_2d(),
                color: [1.0, 1.0, 1.0, 1.0], // White tint (no modification)
                uv_min: [0.0, 0.0],          // Top-left UV coordinate
                uv_max: [1.0, 1.0],          // Bottom-right UV coordinate
            });
        }

        commands
    }
}

// Sprite command structure sent to GPU - must be compatible with bytemuck for memcpy
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SpriteCommand {
    transform: [[f32; 4]; 4], // 4x4 transformation matrix as array for bytemuck compatibility
    color: [f32; 4],          // RGBA color multiplier (1.0 = no change)
    uv_min: [f32; 2],         // Minimum texture coordinates (top-left)
    uv_max: [f32; 2],         // Maximum texture coordinates (bottom-right)
}

// Push constants structure - small data sent directly to shaders without buffers
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PushConstants {
    view_proj: [[f32; 4]; 4], // View-projection matrix as array for bytemuck compatibility
}

// Vulkan queue families - different types of operations need different queues
#[derive(Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32, // Queue that can run graphics commands (drawing triangles)
    present: u32,  // Queue that can present images to the window surface
}

impl QueueFamilyIndices {
    // Finds suitable queue families on the given physical device
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        unsafe {
            // Get all available queue families on this device
            let properties = instance.get_physical_device_queue_family_properties(physical_device);

            // Find a queue family that supports graphics operations
            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);

            // Find a queue family that can present to our window surface
            let mut present = None;
            for (index, _) in properties.iter().enumerate() {
                if instance.get_physical_device_surface_support_khr(
                    physical_device,
                    index as u32,
                    data.surface,
                )? {
                    present = Some(index as u32);
                    break;
                }
            }

            if let (Some(graphics), Some(present)) = (graphics, present) {
                Ok(Self { graphics, present })
            } else {
                Err(anyhow!("Missing required queue families."))
            }
        }
    }
}

// Information about what the swapchain (rendering target) can support
#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR, // Image count limits, size limits, etc.
    formats: Vec<vk::SurfaceFormatKHR>,       // Available pixel formats (RGBA, BGRA, etc.)
    present_modes: Vec<vk::PresentModeKHR>,   // Timing modes (VSync, immediate, mailbox, etc.)
}

// Configuration for swapchain creation using builder pattern
#[derive(Debug)]
struct SwapchainConfig {
    surface_format: vk::SurfaceFormatKHR, // Pixel format and color space
    present_mode: vk::PresentModeKHR,     // VSync mode
    extent: vk::Extent2D,                 // Image dimensions
    image_count: u32,                     // Number of images in swapchain
    image_sharing_mode: vk::SharingMode,  // How queues share access to images
    queue_family_indices: Vec<u32>,       // Which queue families need access
}

impl SwapchainConfig {
    fn new(
        window: &Window,
        support: &SwapchainSupport,
        indices: &QueueFamilyIndices,
        vsync_enabled: bool,
    ) -> Self {
        let surface_format = get_swapchain_surface_format(&support.formats);
        let present_mode = get_swapchain_present_mode(&support.present_modes, vsync_enabled);
        let extent = get_swapchain_extent(window, support.capabilities);

        // Request one more image than minimum for better performance
        let mut image_count =
            support.capabilities.min_image_count + MIN_SWAPCHAIN_IMAGE_COUNT_OFFSET;
        if support.capabilities.max_image_count != 0
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        // Handle queue family sharing for the images
        let (image_sharing_mode, queue_family_indices) = if indices.graphics != indices.present {
            // Different queues need concurrent access (slightly slower)
            (
                vk::SharingMode::CONCURRENT,
                vec![indices.graphics, indices.present],
            )
        } else {
            // Same queue family can use exclusive access (better performance)
            (vk::SharingMode::EXCLUSIVE, vec![])
        };

        Self {
            surface_format,
            present_mode,
            extent,
            image_count,
            image_sharing_mode,
            queue_family_indices,
        }
    }

    fn create_info(
        &self,
        surface: vk::SurfaceKHR,
        capabilities: &vk::SurfaceCapabilitiesKHR,
    ) -> vk::SwapchainCreateInfoKHR {
        vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(self.image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(self.extent)
            .image_array_layers(SWAPCHAIN_IMAGE_ARRAY_LAYERS) // Always 1 unless doing stereoscopic 3D
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) // We'll render into these images
            .image_sharing_mode(self.image_sharing_mode)
            .queue_family_indices(&self.queue_family_indices)
            .pre_transform(capabilities.current_transform) // Don't transform images
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // Don't blend with other windows
            .present_mode(self.present_mode)
            .clipped(true) // Don't care about obscured pixels (performance optimization)
            .old_swapchain(vk::SwapchainKHR::null()) // Not recreating an existing swapchain
            .build()
    }
}

impl SwapchainSupport {
    // Queries the surface to determine what swapchain features are supported
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        unsafe {
            Ok(Self {
                capabilities: instance
                    .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
                formats: instance
                    .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
                present_modes: instance
                    .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
            })
        }
    }
}

// Contains all Vulkan objects and application state
// Order matters for cleanup - resources must be destroyed in reverse creation order
struct AppData {
    // Window surface and device selection (destroyed last)
    surface: vk::SurfaceKHR,             // The window surface we render to
    physical_device: vk::PhysicalDevice, // GPU we selected to use
    graphics_queue: vk::Queue,           // Queue for graphics commands
    present_queue: vk::Queue,            // Queue for presenting to screen

    // Swapchain - double/triple buffering for smooth rendering (must be destroyed before device)
    swapchain_format: vk::Format,     // Pixel format (RGBA, BGRA, etc.)
    swapchain_extent: vk::Extent2D,   // Current window size in pixels
    swapchain: vk::SwapchainKHR,      // The swapchain itself
    swapchain_images: Vec<vk::Image>, // Images we render into (owned by swapchain)
    swapchain_image_views: Vec<vk::ImageView>, // Views for accessing the images (must destroy before swapchain)

    // Graphics pipeline - how to draw our sprites (must be destroyed before device)
    pipeline_layout: vk::PipelineLayout, // Describes resources the pipeline uses
    pipeline: vk::Pipeline,              // The actual graphics pipeline

    // Sprite data buffer - uploaded to GPU each frame (VMA manages cleanup order)
    sprite_command_buffer: vk::Buffer, // GPU buffer containing sprite transformation data
    sprite_command_buffer_allocation: vma::Allocation, // VMA allocation for sprite buffer
    sprite_command_buffer_mapped: *mut SpriteCommand, // CPU pointer to GPU memory (persistently mapped)

    // Descriptors - how shaders access resources (must be destroyed before device)
    descriptor_set: vk::DescriptorSet, // Binds texture and sprite buffer to shaders
    descriptor_pool: vk::DescriptorPool, // Pool to allocate descriptor sets from
    descriptor_set_layout: vk::DescriptorSetLayout, // Layout of descriptor set

    // Texture resources - the Ferris sprite image (VMA manages cleanup order)
    texture_image: vk::Image, // GPU image containing sprite texture
    texture_image_allocation: vma::Allocation, // VMA allocation for texture image
    texture_image_view: vk::ImageView, // View for accessing the texture (must destroy before image)
    texture_sampler: vk::Sampler, // How to sample/filter the texture (must destroy before device)

    // Command recording - how we tell the GPU what to draw (must be destroyed before device)
    command_pool: vk::CommandPool, // Pool to allocate command buffers from
    command_buffers: Vec<vk::CommandBuffer>, // Buffers containing GPU commands (freed with pool)

    // Synchronization - coordinating CPU and GPU work (must be destroyed before device)
    image_available_semaphores: Vec<vk::Semaphore>, // Signals when swapchain image is ready
    render_finished_semaphores: Vec<vk::Semaphore>, // Signals when rendering is done
    in_flight_fences: Vec<vk::Fence>,               // Synchronization for frames in flight
    frame: usize, // Current frame index (0 to MAX_FRAMES_IN_FLIGHT-1)

    // Application state
    sprite_count: usize,   // Number of sprites being rendered
    sprites: Vec<Sprite>,  // Physics simulation state
    sprites_spawned: bool, // Whether sprites have been spawned
    last_update: Instant,  // Time of last physics update

    // Performance metrics
    frame_count: u32,            // Frames rendered this second
    last_metrics_time: Instant,  // When we last printed metrics
    accumulated_frame_time: f32, // Total frame time this second

    // Benchmark support
    total_frame_count: u32,   // Total frames rendered since start
    frame_limit: Option<u32>, // Exit after this many frames (for benchmarking)
}

// Main application state - owns all Vulkan objects
// Destruction order: VMA allocations -> device -> instance -> entry
struct App {
    #[allow(dead_code)]
    entry: Entry, // Vulkan library entry point (destroyed last)
    instance: Instance, // Vulkan instance - connection to Vulkan (destroyed second-to-last)
    device: Device,     // Logical device - interface to the GPU (destroyed after VMA)
    allocator: Option<vma::Allocator>, // VMA allocator for GPU memory management (destroyed before device)
    data: AppData,                     // All other Vulkan objects and app state
}

impl App {
    // Creates and initializes the entire Vulkan application
    unsafe fn create(
        window: &Window,
        sprite_count: usize,
        vsync_enabled: bool,
        frame_limit: Option<u32>,
    ) -> Result<Self> {
        let mut data = AppData {
            // Initialize all Vulkan handles as null - will be created in order
            surface: vk::SurfaceKHR::null(),
            physical_device: vk::PhysicalDevice::null(),
            graphics_queue: vk::Queue::null(),
            present_queue: vk::Queue::null(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            sprite_command_buffer: vk::Buffer::null(),
            sprite_command_buffer_allocation: unsafe { std::mem::zeroed() },
            sprite_command_buffer_mapped: std::ptr::null_mut(),
            descriptor_set: vk::DescriptorSet::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            texture_image: vk::Image::null(),
            texture_image_allocation: unsafe { std::mem::zeroed() },
            texture_image_view: vk::ImageView::null(),
            texture_sampler: vk::Sampler::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),
            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            frame: 0,
            sprite_count,
            sprites: if SPAWN_SPRITES_IMMEDIATELY {
                generate_sprites(sprite_count)
            } else {
                Vec::new()
            },
            sprites_spawned: SPAWN_SPRITES_IMMEDIATELY,
            last_update: Instant::now(),
            frame_count: 0,
            last_metrics_time: Instant::now(),
            accumulated_frame_time: 0.0,
            total_frame_count: 0,
            frame_limit,
        };

        unsafe {
            // Step 1: Load Vulkan library and create entry point
            let loader = LibloadingLoader::new(LIBRARY)?;
            let entry = Entry::new(loader).map_err(map_entry_error)?;

            // Step 2: Create Vulkan instance and window surface
            let instance = create_instance(window, &entry, &mut data)?;
            data.surface = vk_window::create_surface(&instance, window, window)?;

            // Step 3: Select GPU and create logical device
            pick_physical_device(&instance, &mut data)?;
            log_gpu_info(&instance, &mut data);
            let device = create_logical_device(&instance, &mut data)?;

            // Step 4: Create VMA allocator with memory priority support
            let mut allocator_options =
                vma::AllocatorOptions::new(&instance, &device, data.physical_device);
            allocator_options.flags |= vma::AllocatorCreateFlags::EXT_MEMORY_PRIORITY;
            let allocator = Some(vma::Allocator::new(&allocator_options)?);

            // Step 5: Create swapchain for rendering to window
            create_swapchain(window, &instance, &device, &mut data, vsync_enabled)?;
            create_swapchain_image_views(&instance, &device, &mut data)?;

            // Step 6: Create command infrastructure
            create_command_pool(&instance, &device, &mut data)?;

            // Step 7: Load and setup sprite texture
            create_texture_image(&instance, &device, &mut data, allocator.as_ref().unwrap())?;
            create_texture_image_view(&instance, &device, &mut data)?;
            create_texture_sampler(&instance, &device, &mut data)?;

            // Step 8: Setup sprite rendering pipeline
            create_sprite_command_buffer(
                &instance,
                &device,
                &mut data,
                allocator.as_ref().unwrap(),
                sprite_count,
            )?;
            create_pipeline(&instance, &device, &mut data)?;
            create_descriptor_sets(&instance, &device, &mut data)?;

            // Step 9: Create command buffers and synchronization objects
            create_command_buffers(&instance, &device, &mut data)?;
            create_sync_objects(&instance, &device, &mut data)?;

            Ok(Self {
                entry,
                instance,
                device,
                allocator,
                data,
            })
        }
    }

    // Renders one frame - updates physics and draws all sprites
    unsafe fn render(&mut self, window: &Window) -> Result<bool> {
        let frame_start = Instant::now();

        // Update physics simulation with delta time
        let now = Instant::now();
        let dt = now.duration_since(self.data.last_update).as_secs_f32();
        self.data.last_update = now;
        self.update_sprites(dt);

        // Update sprite command buffer with new positions
        unsafe {
            self.update_sprite_command_buffer()?;
        }

        // Calculate view-projection matrix for current window size
        let window_size = window.inner_size();
        let view_proj =
            create_sprite_transform(window_size.width as f32, window_size.height as f32);

        // Get synchronization objects for current frame
        let in_flight_fence = self.data.in_flight_fences[self.data.frame];
        let current_frame = self.data.frame;

        unsafe {
            // Wait for the previous frame using this fence to finish
            // This prevents the CPU from getting too far ahead of the GPU
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

            // Get the next image from the swapchain to render into
            let result = self.device.acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores
                    [current_frame % self.data.image_available_semaphores.len()],
                vk::Fence::null(),
            );

            let image_index = match result {
                Ok((image_index, _)) => image_index as usize,
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                    // Swapchain is out of date (window resized), recreate it
                    self.handle_swapchain_recreation(window)?;
                    return Ok(true);
                }
                Err(e) => return Err(anyhow!("{}", e)),
            };

            // Reset the fence so we can use it again
            self.device.reset_fences(&[in_flight_fence])?;

            // Record commands into this frame's command buffer
            let command_buffer = self.data.command_buffers[current_frame];
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            record_command_buffer(
                &self.device,
                command_buffer,
                image_index,
                &self.data,
                &view_proj,
            )?;

            // Setup synchronization: wait for image to be available, signal when rendering is done
            let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(
                    self.data.image_available_semaphores
                        [current_frame % self.data.image_available_semaphores.len()],
                )
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT); // Wait before writing to framebuffer

            let command_buffer_submit_info =
                vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);

            let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(self.data.render_finished_semaphores[image_index])
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS); // Signal after all graphics work

            let submit_info = vk::SubmitInfo2::builder()
                .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
                .command_buffer_infos(std::slice::from_ref(&command_buffer_submit_info))
                .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info));

            // Submit the command buffer to the GPU for execution
            self.device
                .queue_submit2(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

            // Present the rendered image to the screen
            let swapchains = &[self.data.swapchain];
            let image_indices = &[image_index as u32];
            let signal_semaphores = &[self.data.render_finished_semaphores[image_index]];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(signal_semaphores) // Wait for rendering to complete
                .swapchains(swapchains)
                .image_indices(image_indices);

            let result = self
                .device
                .queue_present_khr(self.data.present_queue, &present_info);
            match result {
                Ok(_) => {}
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                    // Swapchain became out of date during presentation
                    self.handle_swapchain_recreation(window)?;
                }
                Err(e) => return Err(anyhow!("{}", e)),
            }

            // Move to next frame (cycles through 0 to MAX_FRAMES_IN_FLIGHT-1)
            self.data.frame = (self.data.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        let frame_end = Instant::now();
        let frame_time = frame_end.duration_since(frame_start).as_secs_f32();
        // Update performance metrics and check if we should exit
        let should_continue = self.update_metrics(frame_time);

        Ok(should_continue)
    }

    // Tracks and displays performance metrics every second
    fn update_metrics(&mut self, frame_time: f32) -> bool {
        self.data.frame_count += 1;
        self.data.total_frame_count += 1;
        self.data.accumulated_frame_time += frame_time;

        let now = Instant::now();
        let elapsed = now
            .duration_since(self.data.last_metrics_time)
            .as_secs_f32();

        // Check if we should exit after frame limit (benchmark mode)
        if let Some(limit) = self.data.frame_limit {
            if self.data.total_frame_count >= limit {
                // Print final benchmark summary before exiting
                let final_fps = if elapsed > 0.0 {
                    self.data.frame_count as f32 / elapsed
                } else {
                    0.0
                };
                let sprites_per_sec = final_fps * self.data.sprites.len() as f32;
                println!(
                    "BENCHMARK_RESULT: {} sprites, {:.1} FPS, {:.0} sprites/sec",
                    self.data.sprites.len(),
                    final_fps,
                    sprites_per_sec
                );
                return false; // Signal to exit
            }
        }

        // Print performance stats every second
        if elapsed >= 1.0 {
            let fps = self.data.frame_count as f32 / elapsed;
            let avg_frame_time =
                (self.data.accumulated_frame_time / self.data.frame_count as f32) * 1000.0;
            let sprites_per_second = fps * self.data.sprites.len() as f32;

            let window_size = format!(
                "{}x{}",
                self.data.swapchain_extent.width, self.data.swapchain_extent.height
            );

            println!(
                "FPS: {fps:.1} | Frame time: {avg_frame_time:.2}ms | Sprites: {} | Sprites/sec: {sprites_per_second:.0} | Resolution: {window_size}",
                self.data.sprites.len()
            );

            // Reset counters for next second
            self.data.frame_count = 0;
            self.data.accumulated_frame_time = 0.0;
            self.data.last_metrics_time = now;
        }

        true // Continue running
    }

    // Updates physics simulation for all sprites
    fn update_sprites(&mut self, dt: f32) {
        let sprite_size = Vec2::new(SPRITE_WIDTH, SPRITE_HEIGHT);
        let logical_bounds = Vec2::new(LOGICAL_WIDTH, LOGICAL_HEIGHT);
        let scaled_dt = dt * TARGET_FPS; // Scale delta time for consistent physics regardless of framerate
        let gravity = Vec2::new(0.0, GRAVITY * scaled_dt);

        for sprite in &mut self.data.sprites {
            // Basic physics: position += velocity, velocity += gravity
            sprite.pos += sprite.vel * scaled_dt;
            sprite.vel += gravity;

            // Bounce off left and right walls
            if sprite.pos.x + sprite_size.x > logical_bounds.x {
                sprite.vel.x *= -1.0;
                sprite.pos.x = logical_bounds.x - sprite_size.x;
            }
            if sprite.pos.x < 0.0 {
                sprite.vel.x *= -1.0;
                sprite.pos.x = 0.0;
            }

            // Bounce off ground with damping and random boost
            if sprite.pos.y + sprite_size.y > logical_bounds.y {
                sprite.vel.y *= -BOUNCE_DAMPING;
                sprite.pos.y = logical_bounds.y - sprite_size.y;

                // Sometimes add a random upward boost for variety
                if fastrand::f32() < 0.5 {
                    sprite.vel.y -= fastrand::f32() * MAX_BOUNCE_BOOST;
                }
            }

            // Stop at ceiling
            if sprite.pos.y < 0.0 {
                sprite.vel.y = 0.0;
                sprite.pos.y = 0.0;
            }
        }
    }

    // Spawns sprites if they haven't been spawned yet
    fn spawn_sprites(&mut self) {
        if !self.data.sprites_spawned {
            self.data.sprites = generate_sprites(self.data.sprite_count);
            self.data.sprites_spawned = true;
        }
    }

    // Uploads updated sprite data to GPU memory
    unsafe fn update_sprite_command_buffer(&mut self) -> Result<()> {
        unsafe {
            let sprite_commands = self.sprites_to_commands();

            // Copy sprite data directly to GPU-visible memory
            // This is efficient because the buffer is persistently mapped
            std::ptr::copy_nonoverlapping(
                sprite_commands.as_ptr(),
                self.data.sprite_command_buffer_mapped,
                sprite_commands.len(),
            );

            Ok(())
        }
    }

    // Handles swapchain recreation with consolidated logic
    unsafe fn handle_swapchain_recreation(&mut self, window: &Window) -> Result<bool> {
        log::debug!("Recreating swapchain due to window changes or out-of-date swapchain");
        unsafe { self.recreate_swapchain(window) }
    }

    // Recreates swapchain when window is resized or other changes occur
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<bool> {
        unsafe {
            // Wait for all operations to complete before recreating
            self.device.device_wait_idle()?;

            // Clean up old swapchain resources
            self.destroy_swapchain();

            // Create new swapchain and associated resources
            self.create_swapchain_and_views(window, true)?;

            log::debug!("Swapchain recreated successfully");
            Ok(true)
        }
    }

    // Creates swapchain and image views together for better organization
    unsafe fn create_swapchain_and_views(
        &mut self,
        window: &Window,
        vsync_enabled: bool,
    ) -> Result<()> {
        unsafe {
            create_swapchain(
                window,
                &self.instance,
                &self.device,
                &mut self.data,
                vsync_enabled,
            )?;
            create_swapchain_image_views(&self.instance, &self.device, &mut self.data)?;
            Ok(())
        }
    }

    // Cleans up swapchain resources in proper order
    unsafe fn destroy_swapchain(&mut self) {
        unsafe {
            // Destroy all image views first (dependent on swapchain images)
            for &image_view in &self.data.swapchain_image_views {
                self.device.destroy_image_view(image_view, None);
            }
            self.data.swapchain_image_views.clear();

            // Then destroy the swapchain itself (images are owned by swapchain)
            if self.data.swapchain != vk::SwapchainKHR::null() {
                self.device.destroy_swapchain_khr(self.data.swapchain, None);
                self.data.swapchain = vk::SwapchainKHR::null();
            }

            log::debug!("Swapchain resources destroyed");
        }
    }

    // Destroys all Vulkan resources in reverse creation order
    // This is critical for proper cleanup - resources must be destroyed before their dependencies
    unsafe fn destroy(&mut self) {
        unsafe {
            // Wait for all operations to complete before destroying anything
            self.device.device_wait_idle().unwrap();

            // 1. Destroy swapchain and related resources first
            self.destroy_swapchain();

            // 2. Destroy synchronization objects (semaphores and fences)
            for i in 0..self.data.image_available_semaphores.len() {
                self.device
                    .destroy_semaphore(self.data.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.data.render_finished_semaphores[i], None);
            }

            // Destroy fences (one per frame in flight)
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_fence(self.data.in_flight_fences[i], None);
            }

            // 3. Destroy command pool (this automatically frees all command buffers)
            self.device
                .destroy_command_pool(self.data.command_pool, None);

            // 4. Destroy texture resources (views before images)
            self.device.destroy_sampler(self.data.texture_sampler, None);
            self.device
                .destroy_image_view(self.data.texture_image_view, None);

            // 5. Destroy descriptor resources
            self.device
                .destroy_descriptor_pool(self.data.descriptor_pool, None);

            // 6. Destroy pipeline and layout
            self.device.destroy_pipeline(self.data.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.data.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);

            // 7. Destroy all VMA allocations before destroying device
            // VMA manages the actual GPU memory and must be cleaned up before device
            if let Some(allocator) = &self.allocator {
                allocator
                    .destroy_image(self.data.texture_image, self.data.texture_image_allocation);
                allocator.destroy_buffer(
                    self.data.sprite_command_buffer,
                    self.data.sprite_command_buffer_allocation,
                );
            }

            // 8. Explicitly drop VMA allocator before destroying device
            self.allocator.take();

            // 9. Destroy logical device (this invalidates all device-created objects)
            self.device.destroy_device(None);

            // 10. Destroy surface (window-specific resource)
            self.instance.destroy_surface_khr(self.data.surface, None);

            // 11. Destroy instance last (this cleans up the Vulkan connection)
            self.instance.destroy_instance(None);
        }
    }
}

// Creates the Vulkan instance - the connection between our app and the Vulkan library
unsafe fn create_instance(window: &Window, entry: &Entry, _data: &mut AppData) -> Result<Instance> {
    // Check what Vulkan version is available on this system
    let loader_version = unsafe {
        entry
            .enumerate_instance_version()
            .unwrap_or(vk::make_version(1, 0, 0))
    };

    log::info!(
        "Vulkan loader version: {}.{}.{}",
        vk::version_major(loader_version),
        vk::version_minor(loader_version),
        vk::version_patch(loader_version)
    );

    // Request the best available version, but require at least our minimum
    let requested_version = if loader_version >= OPTIMAL_VULKAN_VERSION {
        OPTIMAL_VULKAN_VERSION
    } else if loader_version >= PREFERRED_VULKAN_VERSION {
        PREFERRED_VULKAN_VERSION
    } else if loader_version >= MIN_VULKAN_VERSION {
        MIN_VULKAN_VERSION
    } else {
        return Err(anyhow!(
            "Vulkan loader version {}.{}.{} is too old. Minimum required: 1.1.0",
            vk::version_major(loader_version),
            vk::version_minor(loader_version),
            vk::version_patch(loader_version)
        ));
    };

    // Application metadata for debugging and optimization
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Ferris Mark VK\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(requested_version);

    // Get extensions needed to render to our window surface
    let extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_extension_names(&extensions);

    // Create the Vulkan instance (validation layers are controlled by vkconfig)
    let instance = unsafe { entry.create_instance(&info, None)? };

    Ok(instance)
}

// Selects a GPU that supports the features we need
unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    // Try each available GPU until we find one that works
    for physical_device in physical_devices {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_version = properties.api_version;
        log::info!(
            "Physical device `{}` supports Vulkan {}.{}.{}",
            properties.device_name,
            vk::version_major(device_version),
            vk::version_minor(device_version),
            vk::version_patch(device_version)
        );

        // Check if this GPU meets our requirements
        if let Err(error) = unsafe { check_physical_device(instance, data, physical_device) } {
            log::warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name,
                error
            );
        } else {
            log::info!("Selected physical device (`{}`).", properties.device_name);
            if vk::version_major(device_version) >= 1 && vk::version_minor(device_version) >= 4 {
                log::info!("Device supports Vulkan 1.4 - all features available as core");
            } else {
                log::warn!(
                    "Device does not support Vulkan 1.4 - some features may not be available"
                );
            }

            data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

// Checks if a GPU supports all the features we need for rendering
unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    unsafe {
        // Check if GPU has the queue families we need (graphics and present)
        QueueFamilyIndices::get(instance, data, physical_device)?;

        // Check if GPU supports the extensions we need (swapchain, etc.)
        check_physical_device_extensions(instance, physical_device)?;

        // Check if swapchain has formats and present modes we can use
        let support = SwapchainSupport::get(instance, data, physical_device)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(anyhow!("Insufficient swapchain support."));
        }

        Ok(())
    }
}

// Verifies that the GPU supports all the Vulkan extensions we need
unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    unsafe {
        // Get list of extensions this GPU supports
        let extensions = instance
            .enumerate_device_extension_properties(physical_device, None)?
            .iter()
            .map(|e| e.extension_name)
            .collect::<HashSet<_>>();

        // Check if all required extensions are supported
        if REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .all(|e| extensions.contains(e))
        {
            Ok(())
        } else {
            Err(anyhow!("Missing required device extensions."))
        }
    }
}

// Creates a logical device - our interface to the GPU
unsafe fn create_logical_device(instance: &Instance, data: &mut AppData) -> Result<Device> {
    let indices = unsafe { QueueFamilyIndices::get(instance, data, data.physical_device)? };

    // Create queues for graphics and present (may be same queue family)
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0]; // All queues get equal priority
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };

    // Determine device capabilities and required extensions
    let device_version = properties.api_version;
    let major = vk::version_major(device_version);
    let minor = vk::version_minor(device_version);

    log::info!(
        "Device Vulkan version: {}.{}.{}",
        major,
        minor,
        vk::version_patch(device_version)
    );

    // Check version compatibility
    if device_version < MIN_VULKAN_VERSION {
        return Err(anyhow!(
            "Device does not support minimum required Vulkan version 1.1.0"
        ));
    }

    // Determine which extensions we need based on device version
    let mut required_extensions = REQUIRED_DEVICE_EXTENSIONS.to_vec();
    let is_vulkan_13_plus = device_version >= vk::make_version(1, 3, 0);

    // Add extensions that aren't core in older Vulkan versions
    if !is_vulkan_13_plus {
        // VK_KHR_synchronization2 provides enhanced synchronization (core in 1.3+)
        required_extensions.push(vk::KHR_SYNCHRONIZATION2_EXTENSION.name);
        // VK_KHR_dynamic_rendering eliminates render passes (core in 1.3+)
        required_extensions.push(vk::KHR_DYNAMIC_RENDERING_EXTENSION.name);
    }

    // Convert extension names to the format expected by Vulkan
    let extensions: Vec<*const std::os::raw::c_char> = required_extensions
        .iter()
        .map(|name| name.as_ptr())
        .collect();

    // We don't need any special Vulkan 1.0 features for 2D sprite rendering
    let features = vk::PhysicalDeviceFeatures::builder();

    let device = if is_vulkan_13_plus {
        // Use Vulkan 1.3+ core features
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true) // No render passes needed
            .synchronization2(true); // Enhanced synchronization

        let mut memory_priority_features =
            vk::PhysicalDeviceMemoryPriorityFeaturesEXT::builder().memory_priority(true);

        let mut pageable_device_local_memory_features =
            vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::builder()
                .pageable_device_local_memory(true);

        log::info!("Using Vulkan 1.3+ with core dynamic rendering and synchronization2");

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut vulkan13_features)
            .push_next(&mut memory_priority_features)
            .push_next(&mut pageable_device_local_memory_features);

        unsafe { instance.create_device(data.physical_device, &device_create_info, None)? }
    } else {
        // Fallback for Vulkan 1.1/1.2 devices using extensions
        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::builder().dynamic_rendering(true);

        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);

        let mut memory_priority_features =
            vk::PhysicalDeviceMemoryPriorityFeaturesEXT::builder().memory_priority(true);

        let mut pageable_device_local_memory_features =
            vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::builder()
                .pageable_device_local_memory(true);

        log::info!("Using Vulkan 1.1/1.2 with dynamic rendering and synchronization2 extensions");

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut sync2_features)
            .push_next(&mut memory_priority_features)
            .push_next(&mut pageable_device_local_memory_features);

        unsafe { instance.create_device(data.physical_device, &device_create_info, None)? }
    };

    unsafe {
        // Get the actual queue handles we'll use for submitting work
        data.graphics_queue = device.get_device_queue(indices.graphics, 0);
        data.present_queue = device.get_device_queue(indices.present, 0);

        Ok(device)
    }
}

// Creates the swapchain - multiple images we can render into for smooth display
unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    vsync_enabled: bool,
) -> Result<()> {
    unsafe {
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
        let support = SwapchainSupport::get(instance, data, data.physical_device)?;

        // Create swapchain configuration using builder pattern
        let config = SwapchainConfig::new(window, &support, &indices, vsync_enabled);
        log::debug!(
            "Swapchain config: {}x{}, format: {:?}, present_mode: {:?}, images: {}",
            config.extent.width,
            config.extent.height,
            config.surface_format.format,
            config.present_mode,
            config.image_count
        );
        let info = config.create_info(data.surface, &support.capabilities);

        data.swapchain = device.create_swapchain_khr(&info, None)?;
        data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
        data.swapchain_format = config.surface_format.format;
        data.swapchain_extent = config.extent;

        Ok(())
    }
}

// Chooses the best pixel format for our swapchain images
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|f| {
            // Prefer BGRA 8-bit with sRGB color space (most common and efficient)
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .cloned()
        .unwrap_or(formats[0]) // Fall back to first available format
}

// Chooses the best present mode based on VSync preference
fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
    vsync_enabled: bool,
) -> vk::PresentModeKHR {
    if vsync_enabled {
        // VSync enabled - use FIFO (guaranteed available, synchronized to refresh rate)
        vk::PresentModeKHR::FIFO
    } else {
        // VSync disabled - prefer IMMEDIATE (no sync), fallback to MAILBOX (triple buffering), then FIFO
        present_modes
            .iter()
            .cloned()
            .find(|m| *m == vk::PresentModeKHR::IMMEDIATE)
            .or_else(|| {
                present_modes
                    .iter()
                    .cloned()
                    .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            })
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }
}

// Determines the swapchain image resolution based on window size and surface capabilities
fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        // Most platforms provide the extent directly
        capabilities.current_extent
    } else {
        // Some platforms require us to calculate it ourselves
        let PhysicalSize { width, height } = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                height,
            ))
            .build()
    }
}

// Creates image views for swapchain images so shaders can access them
unsafe fn create_swapchain_image_views(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        // Create subresource range for color images (used by all image views)
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR) // Color data (not depth/stencil)
            .base_mip_level(0) // Start at mip level 0 (full resolution)
            .level_count(1) // Use 1 mip level (no mipmapping)
            .base_array_layer(0) // Start at array layer 0
            .layer_count(1) // Use 1 array layer (not an array texture)
            .build();

        data.swapchain_image_views = data
            .swapchain_images
            .iter()
            .map(|&image| {
                let info = vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::_2D) // 2D texture (not 1D, 3D, or cubemap)
                    .format(data.swapchain_format)
                    .subresource_range(subresource_range)
                    .build();

                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        log::debug!(
            "Created {} swapchain image views",
            data.swapchain_image_views.len()
        );
        Ok(())
    }
}

// Creates the graphics pipeline - defines how to render our sprites
unsafe fn create_pipeline(_instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Load and compile shaders at runtime using shaderc
    let compiler = shaderc::Compiler::new().unwrap();
    let vert_shader_source = std::fs::read_to_string("shaders/sprite.vert")?;
    let frag_shader_source = std::fs::read_to_string("shaders/sprite.frag")?;

    let vert_compiled = compiler
        .compile_into_spirv(
            &vert_shader_source,
            shaderc::ShaderKind::Vertex,
            "sprite.vert",
            "main",
            None,
        )
        .map_err(map_shader_error)?;

    let frag_compiled = compiler
        .compile_into_spirv(
            &frag_shader_source,
            shaderc::ShaderKind::Fragment,
            "sprite.frag",
            "main",
            None,
        )
        .map_err(map_shader_error)?;

    let vert_shader_code = vert_compiled.as_binary_u8();
    let frag_shader_code = frag_compiled.as_binary_u8();

    // No vertex input needed - vertices are generated procedurally in the vertex shader
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    // How to assemble vertices into triangles
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST) // Every 3 vertices = 1 triangle
        .primitive_restart_enable(false);

    // Viewport and scissor are set dynamically, just specify counts
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1) // One viewport covering the whole window
        .scissor_count(1); // One scissor rectangle (no clipping)

    // How to rasterize triangles into pixels
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false) // Don't clamp depth values
        .rasterizer_discard_enable(false) // Actually render pixels
        .polygon_mode(vk::PolygonMode::FILL) // Fill triangles (not wireframe)
        .line_width(1.0) // Line width (only matters for wireframe)
        .cull_mode(vk::CullModeFlags::BACK) // Don't render back-facing triangles
        .front_face(vk::FrontFace::CLOCKWISE) // Clockwise triangles face forward
        .depth_bias_enable(false); // No depth bias

    // No multisampling (anti-aliasing)
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1); // 1 sample per pixel

    // Alpha blending setup for transparency
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all()) // Write all color channels
        .blend_enable(true) // Enable alpha blending
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA) // New color * alpha
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA) // Old color * (1-alpha)
        .color_blend_op(vk::BlendOp::ADD) // Add the two together
        .src_alpha_blend_factor(vk::BlendFactor::ONE) // Keep new alpha
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO) // Ignore old alpha
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false) // No bitwise operations
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]); // Blend constant colors (unused)

    // These states can be changed without recreating the pipeline
    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

    // Describe what resources our shaders need access to
    let bindings = &[
        // Binding 0: Texture sampler for fragment shader
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        // Binding 1: Sprite data buffer for vertex shader
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build(),
    ];

    let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    // Push constants let us send small amounts of data directly to shaders
    let push_constant_ranges = &[vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX) // Only vertex shader needs this
        .offset(0)
        .size(size_of::<PushConstants>() as u32) // Size of our view-projection matrix
        .build()];

    unsafe {
        // Create shader modules from compiled bytecode
        let vert_shader_module = create_shader_module(device, vert_shader_code)?;
        let frag_shader_module = create_shader_module(device, frag_shader_code)?;

        // Define shader stages for the pipeline
        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0"); // Entry point function name

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0"); // Entry point function name

        // Create descriptor set layout (describes shader resources)
        data.descriptor_set_layout =
            device.create_descriptor_set_layout(&descriptor_layout_info, None)?;

        // Create pipeline layout (combines descriptor sets and push constants)
        let set_layouts = &[data.descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);
        data.pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

        // Setup for dynamic rendering (no render passes needed)
        let stages = &[vert_stage, frag_stage];
        let color_attachment_formats = &[data.swapchain_format];
        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(color_attachment_formats);

        // Create the complete graphics pipeline
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(data.pipeline_layout)
            .push_next(&mut pipeline_rendering_create_info);

        let pipelines =
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?;
        data.pipeline = pipelines.0[0];

        // Clean up temporary shader modules
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);

        Ok(())
    }
}

// Creates a Vulkan shader module from SPIR-V bytecode
unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    unsafe {
        // Convert byte array to u32 array (SPIR-V is 32-bit words)
        let bytecode =
            std::slice::from_raw_parts(bytecode.as_ptr().cast(), bytecode.len() / size_of::<u32>());
        let info = vk::ShaderModuleCreateInfo::builder()
            .code_size(std::mem::size_of_val(bytecode))
            .code(bytecode);

        Ok(device.create_shader_module(&info, None)?)
    }
}

// Creates command pool for allocating command buffers
unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // Allow individual command buffer reset
            .queue_family_index(indices.graphics); // Pool for graphics queue

        data.command_pool = device.create_command_pool(&info, None)?;

        Ok(())
    }
}

// Allocates command buffers for recording GPU commands
unsafe fn create_command_buffers(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY) // Primary buffers can be submitted to queues
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32); // One per frame in flight

        data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

        Ok(())
    }
}

// Records drawing commands into a command buffer for one frame
unsafe fn record_command_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
    data: &AppData,
    view_proj: &[[f32; 4]; 4],
) -> Result<()> {
    unsafe {
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0], // Black background - optimal for GPU compression with sRGB
            },
        };

        // Transition image from UNDEFINED to COLOR_ATTACHMENT_OPTIMAL
        let barrier = vk::ImageMemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(data.swapchain_images[image_index])
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );

        let dependency_info =
            vk::DependencyInfo::builder().image_memory_barriers(std::slice::from_ref(&barrier));

        device.cmd_pipeline_barrier2(command_buffer, &dependency_info);

        // Start dynamic rendering (no render pass needed)
        let color_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(data.swapchain_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR) // Clear to background color
            .store_op(vk::AttachmentStoreOp::STORE) // Save the rendered result
            .clear_value(color_clear_value);

        let color_attachments = &[color_attachment];
        let rendering_info = vk::RenderingInfo::builder()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(color_attachments);

        device.cmd_begin_rendering(command_buffer, &rendering_info);

        // Bind our graphics pipeline
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline,
        );

        // Set up viewport to cover the entire window
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(data.swapchain_extent.width as f32)
            .height(data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        device.cmd_set_viewport(command_buffer, 0, &[viewport]);

        // Set up scissor rectangle (no clipping)
        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(data.swapchain_extent);

        device.cmd_set_scissor(command_buffer, 0, &[scissor]);

        // Bind descriptor set with texture and sprite buffer
        let descriptor_sets = &[data.descriptor_set];
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline_layout,
            0,
            descriptor_sets,
            &[],
        );

        // Push constants with view-projection matrix
        let push_constants = PushConstants {
            view_proj: *view_proj,
        };
        device.cmd_push_constants(
            command_buffer,
            data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            std::slice::from_raw_parts(
                (&raw const push_constants).cast::<u8>(),
                size_of::<PushConstants>(),
            ),
        );

        // Instanced draw: 6 vertices per quad (2 triangles), data.sprite_count instances
        device.cmd_draw(command_buffer, 6, data.sprite_count as u32, 0, 0);

        device.cmd_end_rendering(command_buffer);

        // Transition image from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
        let barrier = vk::ImageMemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(data.swapchain_images[image_index])
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );

        let dependency_info =
            vk::DependencyInfo::builder().image_memory_barriers(std::slice::from_ref(&barrier));

        device.cmd_pipeline_barrier2(command_buffer, &dependency_info);

        device.end_command_buffer(command_buffer)?;

        Ok(())
    }
}

// Loads the Ferris sprite texture from disk and uploads it to GPU memory
unsafe fn create_texture_image(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
    allocator: &vma::Allocator,
) -> Result<()> {
    // Load image from disk and convert to RGBA format
    let img = image::open("ferris.png")?.to_rgba8();
    let (width, height) = (img.width(), img.height());
    let size = u64::from(width * height * RGBA_BYTES_PER_PIXEL);

    // Create staging buffer to temporarily hold image data in CPU-accessible memory
    let staging_buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC) // Source for transfer operations
        .sharing_mode(vk::SharingMode::EXCLUSIVE); // Only one queue accesses it

    // Create GPU image that will hold the final texture
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D) // 2D texture
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1, // Always 1 for 2D images
        })
        .mip_levels(1) // No mipmapping
        .array_layers(1) // Single image, not an array
        .format(vk::Format::R8G8B8A8_SRGB) // RGBA 8-bit with sRGB color space
        .tiling(vk::ImageTiling::OPTIMAL) // GPU-optimized memory layout
        .initial_layout(vk::ImageLayout::UNDEFINED) // We don't care about initial contents
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED) // Transfer target and shader sampling
        .sharing_mode(vk::SharingMode::EXCLUSIVE) // Only graphics queue uses it
        .samples(vk::SampleCountFlags::_1); // No multisampling

    unsafe {
        // Create staging buffer using VMA for CPU-visible memory
        let staging_allocation_options = vma::AllocationOptions {
            usage: vma::MemoryUsage::AutoPreferHost,
            priority: 0.3, // Lower priority - temporary staging buffer
            flags: vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let (staging_buffer, staging_allocation) =
            allocator.create_buffer(staging_buffer_info, &staging_allocation_options)?;

        // Copy image pixel data from CPU to staging buffer
        let memory_ptr = allocator.map_memory(staging_allocation)?;
        std::ptr::copy_nonoverlapping(img.as_raw().as_ptr(), memory_ptr.cast(), size as usize);
        allocator.unmap_memory(staging_allocation);

        // Create the actual GPU texture image using VMA with high priority
        let image_allocation_options = vma::AllocationOptions {
            usage: vma::MemoryUsage::AutoPreferDevice, // GPU-only memory (faster)
            priority: 1.0, // Highest priority - textures are frequently accessed by GPU
            ..Default::default()
        };

        let (texture_image, texture_allocation) =
            allocator.create_image(image_info, &image_allocation_options)?;

        data.texture_image = texture_image;
        data.texture_image_allocation = texture_allocation;

        // Transition image to receive data, copy from staging buffer, then optimize for shader reads
        transition_image_layout(
            device,
            data,
            data.texture_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, // Optimal for receiving data
        )?;
        copy_buffer_to_image(
            device,
            data,
            staging_buffer,
            data.texture_image,
            width,
            height,
        )?;
        transition_image_layout(
            device,
            data,
            data.texture_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, // Optimal for shader sampling
        )?;

        // Clean up staging resources using VMA
        allocator.destroy_buffer(staging_buffer, staging_allocation);

        Ok(())
    }
}

// Creates an image view for the texture so shaders can access it
unsafe fn create_texture_image_view(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        let info = vk::ImageViewCreateInfo::builder()
            .image(data.texture_image)
            .view_type(vk::ImageViewType::_2D) // View as 2D texture
            .format(vk::Format::R8G8B8A8_SRGB) // Same format as the image
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR) // Color data (not depth/stencil)
                    .base_mip_level(0) // Start at mip level 0
                    .level_count(1) // Use 1 mip level
                    .base_array_layer(0) // Start at array layer 0
                    .layer_count(1) // Use 1 array layer
                    .build(),
            );

        data.texture_image_view = device.create_image_view(&info, None)?;

        Ok(())
    }
}

// Creates a sampler that defines how the texture is filtered and wrapped
unsafe fn create_texture_sampler(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR) // Smooth interpolation when zoomed in
            .min_filter(vk::Filter::LINEAR) // Smooth interpolation when zoomed out
            .address_mode_u(vk::SamplerAddressMode::REPEAT) // Repeat texture horizontally
            .address_mode_v(vk::SamplerAddressMode::REPEAT) // Repeat texture vertically
            .address_mode_w(vk::SamplerAddressMode::REPEAT) // Repeat texture in depth (unused for 2D)
            .anisotropy_enable(false) // No anisotropic filtering (would improve quality at angles)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK) // Border color (unused with REPEAT)
            .unnormalized_coordinates(false) // Use normalized coordinates (0.0 to 1.0)
            .compare_enable(false) // No shadow mapping comparison
            .compare_op(vk::CompareOp::ALWAYS) // Comparison operator (unused)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR) // How to blend between mip levels
            .mip_lod_bias(0.0) // Bias for mip level selection
            .min_lod(0.0) // Minimum mip level
            .max_lod(0.0); // Maximum mip level (0 = no mipmapping)

        data.texture_sampler = device.create_sampler(&info, None)?;

        Ok(())
    }
}

// Creates a buffer that holds sprite data for GPU rendering
unsafe fn create_sprite_command_buffer(
    _instance: &Instance,
    _device: &Device,
    data: &mut AppData,
    allocator: &vma::Allocator,
    sprite_count: usize,
) -> Result<()> {
    unsafe {
        let size = (sprite_count * size_of::<SpriteCommand>()) as u64;

        // Create buffer in CPU-visible memory for frequent updates using VMA
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER) // Storage buffer for shader access
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_options = vma::AllocationOptions {
            usage: vma::MemoryUsage::Auto, // CPU-visible memory for frequent updates
            priority: 0.7,                 // High-medium priority - frequently updated sprite data
            flags: vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vma::AllocationCreateFlags::MAPPED, // Keep persistently mapped
            ..Default::default()
        };

        let (sprite_command_buffer, sprite_command_buffer_allocation) =
            allocator.create_buffer(buffer_create_info, &allocation_options)?;

        data.sprite_command_buffer = sprite_command_buffer;
        data.sprite_command_buffer_allocation = sprite_command_buffer_allocation;

        // Get the mapped pointer from VMA allocation
        data.sprite_command_buffer_mapped = allocator
            .get_allocation_info(sprite_command_buffer_allocation)
            .pMappedData
            .cast();

        Ok(())
    }
}

// Creates descriptor sets that bind resources to shaders
unsafe fn create_descriptor_sets(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        // Create descriptor pool - allocates memory for descriptor sets
        let pool_sizes = &[
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER) // For texture + sampler
                .descriptor_count(1) // We need 1 texture binding
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER) // For sprite data buffer
                .descriptor_count(1) // We need 1 buffer binding
                .build(),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(1); // Only allocating 1 descriptor set

        data.descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

        // Allocate descriptor set from the pool
        let layouts = &[data.descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(data.descriptor_pool)
            .set_layouts(layouts);

        let descriptor_sets = device.allocate_descriptor_sets(&allocate_info)?;
        data.descriptor_set = descriptor_sets[0];

        // Bind our actual resources to the descriptor set
        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) // Image is ready for shader reads
            .image_view(data.texture_image_view) // Our Ferris texture
            .sampler(data.texture_sampler); // How to sample the texture

        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.sprite_command_buffer) // Our sprite data buffer
            .offset(0) // Use the whole buffer
            .range(vk::WHOLE_SIZE as u64); // From offset to end of buffer

        let descriptor_writes = &[
            // Bind texture to binding 0 (fragment shader)
            vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_set)
                .dst_binding(0) // binding = 0 in shader
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info))
                .build(),
            // Bind sprite buffer to binding 1 (vertex shader)
            vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_set)
                .dst_binding(1) // binding = 1 in shader
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info))
                .build(),
        ];

        // Actually update the descriptor set with our resources
        device.update_descriptor_sets(descriptor_writes, &[] as &[vk::CopyDescriptorSet]);

        Ok(())
    }
}

// Copies data from a buffer to an image using a one-time command buffer
unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    unsafe {
        // Allocate a temporary command buffer for this operation
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(data.command_pool)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

        // Define the region to copy
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0) // Start of buffer
            .buffer_row_length(0) // Tightly packed
            .buffer_image_height(0) // No padding between rows
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        device.end_command_buffer(command_buffer)?;

        // Submit and wait for completion
        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;

        // Clean up temporary command buffer
        device.free_command_buffers(data.command_pool, &[command_buffer]);

        Ok(())
    }
}

// Transitions an image from one layout to another using a pipeline barrier
unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    unsafe {
        // Allocate a temporary command buffer for this operation
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(data.command_pool)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

        // Determine the correct access masks and pipeline stages for the transition
        let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                _ => return Err(anyhow!("Unsupported layout transition!")),
            };

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED) // No queue ownership transfer
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        device.end_command_buffer(command_buffer)?;

        // Submit and wait for completion
        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;

        // Clean up temporary command buffer
        device.free_command_buffers(data.command_pool, &[command_buffer]);

        Ok(())
    }
}

// Creates synchronization objects for coordinating CPU and GPU work
unsafe fn create_sync_objects(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED); // Start signaled

        // Create semaphores per swapchain image for proper synchronization
        for _ in 0..data.swapchain_images.len() {
            // Signals when swapchain image becomes available for rendering
            data.image_available_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            // Signals when rendering to this image is complete
            data.render_finished_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
        }

        // Create fences per frame in flight to prevent CPU from getting too far ahead
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            data.in_flight_fences
                .push(device.create_fence(&fence_info, None)?);
        }

        Ok(())
    }
}

// Calculates scaling and offset for maintaining aspect ratio
fn calculate_scaling_and_offset(window_width: f32, window_height: f32) -> (f32, f32, f32) {
    let window_aspect = window_width / window_height;
    let logical_aspect = LOGICAL_WIDTH / LOGICAL_HEIGHT;

    let (scale, viewport_width, viewport_height) = if window_aspect > logical_aspect {
        // Window is wider than logical aspect ratio - add pillarboxes (black bars on sides)
        let scale = window_height / LOGICAL_HEIGHT;
        let viewport_width = LOGICAL_WIDTH * scale;
        (scale, viewport_width, window_height)
    } else {
        // Window is taller than logical aspect ratio - add letterboxes (black bars on top/bottom)
        let scale = window_width / LOGICAL_WIDTH;
        let viewport_height = LOGICAL_HEIGHT * scale;
        (scale, window_width, viewport_height)
    };

    let offset_x = (window_width - viewport_width) * 0.5;
    let offset_y = (window_height - viewport_height) * 0.5;

    (scale, offset_x, offset_y)
}

// Creates a view-projection matrix that maps logical coordinates to screen coordinates
fn create_sprite_transform(window_width: f32, window_height: f32) -> [[f32; 4]; 4] {
    let (scale, offset_x, offset_y) = calculate_scaling_and_offset(window_width, window_height);

    // Use glam for cleaner matrix math
    let logical_size = glam::vec2(LOGICAL_WIDTH, LOGICAL_HEIGHT);
    let window_size = glam::vec2(window_width, window_height);
    let offset = glam::vec2(offset_x, offset_y);

    // Calculate the actual viewport in normalized device coordinates (-1 to 1)
    let viewport_min = (offset / window_size) * 2.0 - 1.0;
    let viewport_max = ((offset + logical_size * scale) / window_size) * 2.0 - 1.0;

    // Create orthographic projection matrix using glam
    let left = 0.0;
    let right = LOGICAL_WIDTH;
    let bottom = LOGICAL_HEIGHT; // Flip Y coordinate so (0,0) is top-left
    let top = 0.0;

    // Map logical coordinates to the calculated viewport
    let viewport_size = viewport_max - viewport_min;
    let logical_size_2d = glam::vec2(right - left, bottom - top);

    let transform = glam::Mat4::from_cols(
        glam::vec4(viewport_size.x / logical_size_2d.x, 0.0, 0.0, 0.0),
        glam::vec4(0.0, viewport_size.y / logical_size_2d.y, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 1.0, 0.0),
        glam::vec4(
            viewport_min.x + viewport_size.x * (-left / logical_size_2d.x),
            viewport_min.y + viewport_size.y * (-top / logical_size_2d.y),
            0.0,
            1.0,
        ),
    );

    transform.to_cols_array_2d()
}

// Prints information about the selected GPU
unsafe fn log_gpu_info(instance: &Instance, data: &mut AppData) {
    if data.physical_device != vk::PhysicalDevice::null() {
        let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };
        let device_name =
            unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };
        println!("GPU: {device_name}");

        let api_version = properties.api_version;
        println!(
            "Vulkan API: {}.{}.{}",
            vk::version_major(api_version),
            vk::version_minor(api_version),
            vk::version_patch(api_version)
        );
    }
}

// Main entry point - sets up window and runs the render loop
fn main() -> Result<()> {
    // Initialize logging
    pretty_env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // Show help if requested
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("=== Ferris Mark VK - Vulkan Sprite Benchmark ===");
        println!("Usage: {} [SPRITE_COUNT] [OPTIONS]", args[0]);
        println!("Arguments:");
        println!("  SPRITE_COUNT    Number of sprites to render (default: 1)");
        println!("Options:");
        println!("  --vsync         Enable VSync");
        println!("  --frames N      Run for N frames then exit");
        println!("  --help, -h      Show this help message");
        return Ok(());
    }

    let sprite_count = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1)
    } else {
        1 // Default to 1 sprite if no argument provided
    };

    let vsync_enabled = args.iter().any(|arg| arg == "--vsync");

    // Parse frame limit for benchmark mode
    let frame_limit = args
        .iter()
        .position(|arg| arg == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u32>().ok());

    // Print startup information
    println!("=== Ferris Mark VK - Vulkan Sprite Benchmark ===");
    println!("Rendering {sprite_count} sprites");
    println!("Logical resolution: {LOGICAL_WIDTH}x{LOGICAL_HEIGHT}");
    println!("Initial window size: {INITIAL_WINDOW_WIDTH}x{INITIAL_WINDOW_HEIGHT}");
    println!("Physics: gravity={GRAVITY}, bounce_damping={BOUNCE_DAMPING}");
    println!(
        "VSync: {}",
        if vsync_enabled { "Enabled" } else { "Disabled" }
    );
    if let Some(frames) = frame_limit {
        println!("Frame limit: {frames} frames");
    }
    println!("Validation layers: Controlled by vkconfig");
    println!("Performance metrics will be logged every second...\n");

    // Create window and event loop
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Ferris Mark VK")
        // PhysicalSize creates a window with the specified dimensions, even if Windows scaling is more than 100%
        .with_inner_size(winit::dpi::PhysicalSize::new(
            INITIAL_WINDOW_WIDTH,
            INITIAL_WINDOW_HEIGHT,
        ))
        .build(&event_loop)?;

    // Initialize Vulkan application
    let mut app = unsafe { App::create(&window, sprite_count, vsync_enabled, frame_limit)? };

    // Run the main event loop
    event_loop.run(move |event, target| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            // Clean up Vulkan resources before exiting
            unsafe {
                app.destroy();
            }
            target.exit();
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Space),
                            state: winit::event::ElementState::Pressed,
                            ..
                        },
                    ..
                },
            ..
        } => {
            // Spawn sprites when spacebar is pressed
            app.spawn_sprites();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => unsafe {
            // Render one frame and check if we should continue
            match app.render(&window) {
                Ok(should_continue) => {
                    if !should_continue {
                        // Benchmark finished, clean up and exit
                        app.destroy();
                        target.exit();
                    }
                }
                Err(e) => {
                    eprintln!("Render error: {e}");
                    app.destroy();
                    target.exit();
                }
            }
        },
        Event::AboutToWait => {
            // Request a new frame to be drawn
            window.request_redraw();
        }
        _ => {}
    })?;
    Ok(())
}
