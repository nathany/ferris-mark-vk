use anyhow::{Result, anyhow};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use std::collections::HashSet;
use std::env;
use std::mem::size_of;
use std::time::Instant;
use vulkanalia::prelude::v1_3::*;
use vulkanalia::vk::{KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{
    Device, Entry, Instance,
    loader::{LIBRARY, LibloadingLoader},
    window as vk_window,
};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

// Error handling wrapper functions
fn map_entry_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow!("{}", error)
}

fn map_shader_error(error: shaderc::Error) -> anyhow::Error {
    anyhow!("Shader compilation failed: {}", error)
}

// Vulkan extensions required for rendering to a window surface
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
// Number of frames we can work on simultaneously (prevents CPU waiting for GPU)
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
const BOUNCE_DAMPING: f32 = 0.90; // Energy loss when sprites hit the ground

// Sprite rendering constants
const SPRITE_WIDTH: f32 = 99.0; // Width of each Ferris sprite
const SPRITE_HEIGHT: f32 = 70.0; // Height of each Ferris sprite

// Image and timing constants
const RGBA_BYTES_PER_PIXEL: u32 = 4; // 4 bytes per pixel for RGBA format
const TARGET_FPS: f32 = 60.0; // Target frame rate for physics scaling

// Sprite velocity constants
const MAX_INITIAL_HORIZONTAL_VELOCITY: f32 = 10.0; // Maximum initial horizontal velocity
const MIN_INITIAL_VERTICAL_VELOCITY: f32 = 5.0; // Minimum initial upward velocity
const MAX_INITIAL_VERTICAL_VELOCITY: f32 = 10.0; // Maximum initial upward velocity
const MAX_BOUNCE_BOOST: f32 = 9.0; // Maximum random upward boost on bounce

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
                uv_min: [0.0, 0.0],          // Top-left UV
                uv_max: [1.0, 1.0],          // Bottom-right UV
            });
        }

        commands
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SpriteCommand {
    transform: [[f32; 4]; 4], // Mat4 as array for bytemuck compatibility
    color: [f32; 4],
    uv_min: [f32; 2],
    uv_max: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PushConstants {
    view_proj: [[f32; 4]; 4], // Mat4 as array for bytemuck compatibility
}

#[derive(Clone, Debug)]
// Vulkan queue families - different types of operations need different queues
struct QueueFamilyIndices {
    graphics: u32, // Queue that can run graphics commands (drawing)
    present: u32,  // Queue that can present images to the window surface
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        unsafe {
            let properties = instance.get_physical_device_queue_family_properties(physical_device);

            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);

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

#[derive(Clone, Debug)]
// Information about what the swapchain (rendering target) can support
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR, // Image count, size limits, etc.
    formats: Vec<vk::SurfaceFormatKHR>,       // Available pixel formats (RGBA, BGRA, etc.)
    present_modes: Vec<vk::PresentModeKHR>,   // Timing modes (VSync, immediate, etc.)
}

impl SwapchainSupport {
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
struct AppData {
    // Window surface and device selection
    surface: vk::SurfaceKHR,             // The window surface we render to
    physical_device: vk::PhysicalDevice, // GPU we selected to use
    graphics_queue: vk::Queue,           // Queue for graphics commands
    present_queue: vk::Queue,            // Queue for presenting to screen

    // Swapchain - double/triple buffering for smooth rendering
    swapchain_format: vk::Format,     // Pixel format (RGBA, BGRA, etc.)
    swapchain_extent: vk::Extent2D,   // Current window size in pixels
    swapchain: vk::SwapchainKHR,      // The swapchain itself
    swapchain_images: Vec<vk::Image>, // Images we render into
    swapchain_image_views: Vec<vk::ImageView>, // Views for accessing the images

    // Graphics pipeline - how to draw our sprites
    pipeline_layout: vk::PipelineLayout, // Describes resources the pipeline uses
    pipeline: vk::Pipeline,              // The actual graphics pipeline

    // Sprite data buffer - uploaded to GPU each frame
    sprite_command_buffer: vk::Buffer, // GPU buffer containing sprite data
    sprite_command_buffer_memory: vk::DeviceMemory, // Memory backing the buffer
    sprite_command_buffer_mapped: *mut SpriteCommand, // CPU pointer to GPU memory

    // Descriptors - how shaders access resources
    descriptor_set: vk::DescriptorSet, // Binds texture and sprite buffer to shaders
    descriptor_pool: vk::DescriptorPool, // Pool to allocate descriptor sets from
    descriptor_set_layout: vk::DescriptorSetLayout, // Layout of descriptor set

    // Texture resources - the Ferris sprite image
    texture_image: vk::Image, // GPU image containing sprite texture
    texture_image_memory: vk::DeviceMemory, // Memory backing the texture
    texture_image_view: vk::ImageView, // View for accessing the texture
    texture_sampler: vk::Sampler, // How to sample/filter the texture

    // Command recording - how we tell the GPU what to draw
    command_pool: vk::CommandPool, // Pool to allocate command buffers from
    command_buffers: Vec<vk::CommandBuffer>, // Buffers containing GPU commands

    // Synchronization - coordinating CPU and GPU work
    image_available_semaphores: Vec<vk::Semaphore>, // Signals when swapchain image is ready
    render_finished_semaphores: Vec<vk::Semaphore>, // Signals when rendering is done
    in_flight_fences: Vec<vk::Fence>,               // CPU waits for GPU to finish frame
    frame: usize, // Current frame index (0 to MAX_FRAMES_IN_FLIGHT-1)

    // Application state
    sprite_count: usize,  // Number of sprites being rendered
    sprites: Vec<Sprite>, // Physics simulation state
    last_update: Instant, // Time of last physics update

    // Performance metrics
    frame_count: u32,            // Frames rendered this second
    last_metrics_time: Instant,  // When we last printed metrics
    accumulated_frame_time: f32, // Total frame time this second
}

// Main application state - owns all Vulkan objects
struct App {
    #[allow(dead_code)]
    entry: Entry, // Vulkan library entry point
    instance: Instance, // Vulkan instance - connection to Vulkan
    device: Device,     // Logical device - interface to the GPU
    data: AppData,      // All other Vulkan objects and app state
}

impl App {
    // Creates and initializes the entire Vulkan application
    unsafe fn create(window: &Window, sprite_count: usize, vsync_enabled: bool) -> Result<Self> {
        let mut data = AppData {
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
            sprite_command_buffer_memory: vk::DeviceMemory::null(),
            sprite_command_buffer_mapped: std::ptr::null_mut(),
            descriptor_set: vk::DescriptorSet::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            texture_image: vk::Image::null(),
            texture_image_memory: vk::DeviceMemory::null(),
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
            sprites: generate_sprites(sprite_count),
            last_update: Instant::now(),
            frame_count: 0,
            last_metrics_time: Instant::now(),
            accumulated_frame_time: 0.0,
        };

        unsafe {
            // Load Vulkan library and create entry point
            let loader = LibloadingLoader::new(LIBRARY)?;
            let entry = Entry::new(loader).map_err(map_entry_error)?;

            // Create Vulkan instance and window surface
            let instance = create_instance(window, &entry, &mut data)?;
            data.surface = vk_window::create_surface(&instance, window, window)?;

            // Select GPU and create logical device
            pick_physical_device(&instance, &mut data)?;
            log_gpu_info(&instance, &mut data);
            let device = create_logical_device(&instance, &mut data)?;

            // Create swapchain for rendering to window
            create_swapchain(window, &instance, &device, &mut data, vsync_enabled)?;
            create_swapchain_image_views(&instance, &device, &mut data)?;

            // Create command infrastructure
            create_command_pool(&instance, &device, &mut data)?;

            // Load and setup sprite texture
            create_texture_image(&instance, &device, &mut data)?;
            create_texture_image_view(&instance, &device, &mut data)?;
            create_texture_sampler(&instance, &device, &mut data)?;

            // Setup sprite rendering pipeline
            create_sprite_command_buffer(&instance, &device, &mut data, sprite_count)?;
            create_pipeline(&instance, &device, &mut data)?;
            create_descriptor_sets(&instance, &device, &mut data)?;

            // Create command buffers and synchronization objects
            create_command_buffers(&instance, &device, &mut data)?;
            create_sync_objects(&instance, &device, &mut data)?;

            Ok(Self {
                entry,
                instance,
                device,
                data,
            })
        }
    }

    // Renders one frame - updates physics and draws all sprites
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let frame_start = Instant::now();

        // Update physics simulation
        let now = Instant::now();
        let dt = now.duration_since(self.data.last_update).as_secs_f32();
        self.data.last_update = now;
        self.update_sprites(dt);

        // Calculate view-projection matrix for current window size
        let window_size = window.inner_size();
        let view_proj =
            create_sprite_transform(window_size.width as f32, window_size.height as f32);

        // Get synchronization objects for current frame
        let in_flight_fence = self.data.in_flight_fences[self.data.frame];
        let current_frame = self.data.frame;

        unsafe {
            // Wait for the previous frame using this fence to finish
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
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
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

            // Setup synchronization: wait for image, signal when rendering done
            let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(
                    self.data.image_available_semaphores
                        [current_frame % self.data.image_available_semaphores.len()],
                )
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

            let command_buffer_submit_info =
                vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);

            let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(self.data.render_finished_semaphores[image_index])
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);

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
                .wait_semaphores(signal_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            let result = self
                .device
                .queue_present_khr(self.data.present_queue, &present_info);
            match result {
                Ok(_) => {}
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window)?;
                }
                Err(e) => return Err(anyhow!("{}", e)),
            }

            // Move to next frame (cycles through 0 to MAX_FRAMES_IN_FLIGHT-1)
            self.data.frame = (self.data.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        let frame_end = Instant::now();
        let frame_time = frame_end.duration_since(frame_start).as_secs_f32();
        self.update_metrics(frame_time);

        Ok(())
    }

    // Tracks and displays performance metrics every second
    fn update_metrics(&mut self, frame_time: f32) {
        self.data.frame_count += 1;
        self.data.accumulated_frame_time += frame_time;

        let now = Instant::now();
        let elapsed = now
            .duration_since(self.data.last_metrics_time)
            .as_secs_f32();

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
    }

    // Updates physics simulation for all sprites
    fn update_sprites(&mut self, dt: f32) {
        let sprite_size = Vec2::new(SPRITE_WIDTH, SPRITE_HEIGHT);
        let logical_bounds = Vec2::new(LOGICAL_WIDTH, LOGICAL_HEIGHT);
        let scaled_dt = dt * TARGET_FPS; // Scale for consistent feel regardless of framerate
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

        // Upload new sprite positions to GPU
        unsafe {
            self.update_sprite_command_buffer().unwrap();
        }
    }

    // Uploads updated sprite data to GPU memory
    unsafe fn update_sprite_command_buffer(&mut self) -> Result<()> {
        unsafe {
            let sprite_commands = self.sprites_to_commands();

            // Copy sprite data directly to GPU-visible memory
            std::ptr::copy_nonoverlapping(
                sprite_commands.as_ptr(),
                self.data.sprite_command_buffer_mapped,
                sprite_commands.len(),
            );

            Ok(())
        }
    }

    // Recreates swapchain when window is resized or other changes occur
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe {
            // Wait for all operations to complete before recreating
            self.device.device_wait_idle()?;
            self.destroy_swapchain();
            create_swapchain(window, &self.instance, &self.device, &mut self.data, true)?;
            create_swapchain_image_views(&self.instance, &self.device, &mut self.data)?;
            Ok(())
        }
    }

    // Cleans up swapchain resources
    unsafe fn destroy_swapchain(&mut self) {
        unsafe {
            for image_view in &self.data.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.data.swapchain_image_views.clear();
            self.device.destroy_swapchain_khr(self.data.swapchain, None);
        }
    }

    unsafe fn destroy(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.destroy_swapchain();

            // Destroy semaphores (one per swapchain image)
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

            self.device
                .destroy_command_pool(self.data.command_pool, None);

            // Destroy texture resources
            self.device.destroy_sampler(self.data.texture_sampler, None);
            self.device
                .destroy_image_view(self.data.texture_image_view, None);
            self.device.destroy_image(self.data.texture_image, None);
            self.device
                .free_memory(self.data.texture_image_memory, None);

            // Destroy buffer resources
            // Destroy descriptor resources
            self.device
                .destroy_descriptor_pool(self.data.descriptor_pool, None);

            // Unmap and destroy buffers
            if !self.data.sprite_command_buffer_mapped.is_null() {
                self.device
                    .unmap_memory(self.data.sprite_command_buffer_memory);
            }
            self.device
                .destroy_buffer(self.data.sprite_command_buffer, None);
            self.device
                .free_memory(self.data.sprite_command_buffer_memory, None);

            self.device.destroy_pipeline(self.data.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.data.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_surface_khr(self.data.surface, None);

            self.instance.destroy_instance(None);
        }
    }
}

// Creates the Vulkan instance - the connection between our app and the Vulkan library
unsafe fn create_instance(window: &Window, entry: &Entry, _data: &mut AppData) -> Result<Instance> {
    // Check what Vulkan version is available
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

    // Request Vulkan 1.4 if available, otherwise use what's available
    let requested_version =
        if vk::version_major(loader_version) >= 1 && vk::version_minor(loader_version) >= 4 {
            vk::make_version(1, 4, 0)
        } else {
            loader_version
        };

    // Application metadata
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Ferris Mark VK\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(requested_version);

    // Get extensions needed to render to our window
    let extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_extension_names(&extensions);

    // Create the Vulkan instance
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
                log::info!(
                    "Device supports Vulkan 1.4 - maintenance6 and other 1.4 features available"
                );
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
        if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
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

    // Extensions this device needs to support
    let extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };

    // Check Vulkan version for advanced features
    let device_version = properties.api_version;
    let supports_vulkan_14 =
        vk::version_major(device_version) >= 1 && vk::version_minor(device_version) >= 4;

    if supports_vulkan_14 {
        log::info!("Vulkan 1.4 device detected - maintenance6 and enhanced features available");
    } else {
        log::info!("Using basic Vulkan features (pre-1.4 device)");
    }

    // Enable modern Vulkan features we need
    let mut sync2_features =
        vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);

    let mut dynamic_rendering_features =
        vk::PhysicalDeviceDynamicRenderingFeatures::builder().dynamic_rendering(true);

    let mut maintenance4_features =
        vk::PhysicalDeviceMaintenance4Features::builder().maintenance4(true);

    let mut maintenance5_features =
        vk::PhysicalDeviceMaintenance5Features::builder().maintenance5(true);

    let mut buffer_device_address_features =
        vk::PhysicalDeviceBufferDeviceAddressFeatures::builder().buffer_device_address(true);

    let features = vk::PhysicalDeviceFeatures::builder();

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut sync2_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut maintenance4_features)
        .push_next(&mut maintenance5_features)
        .push_next(&mut buffer_device_address_features);

    unsafe {
        // Create the logical device
        let device = instance.create_device(data.physical_device, &device_create_info, None)?;

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

        // Choose the best format, present mode, and extent for our swapchain
        let surface_format = get_swapchain_surface_format(&support.formats);
        let present_mode = get_swapchain_present_mode(&support.present_modes, vsync_enabled);
        let extent = get_swapchain_extent(window, support.capabilities);

        // Request one more image than minimum for better performance
        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        // Handle queue family sharing for the images
        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.present {
            // Different queues need concurrent access
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            // Same queue family can use exclusive access (better performance)
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(data.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1) // Always 1 unless doing stereoscopic 3D
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) // We'll render into these
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform) // Don't transform images
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // Don't blend with other windows
            .present_mode(present_mode)
            .clipped(true) // Don't care about obscured pixels
            .old_swapchain(vk::SwapchainKHR::null()); // Not recreating an existing swapchain

        data.swapchain = device.create_swapchain_khr(&info, None)?;
        data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
        data.swapchain_format = surface_format.format;
        data.swapchain_extent = extent;

        Ok(())
    }
}

// Chooses the best pixel format for our swapchain images
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|f| {
            // Prefer BGRA 8-bit with sRGB color space (most common)
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .cloned()
        .unwrap_or(formats[0]) // Fall back to first available format
}

fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
    vsync_enabled: bool,
) -> vk::PresentModeKHR {
    if vsync_enabled {
        // VSync enabled - prefer FIFO (guaranteed available)
        vk::PresentModeKHR::FIFO
    } else {
        // VSync disabled - prefer IMMEDIATE, fallback to MAILBOX, then FIFO
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

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
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
        data.swapchain_image_views = data
            .swapchain_images
            .iter()
            .map(|i| {
                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D) // 2D texture (not 1D, 3D, or cubemap)
                    .format(data.swapchain_format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR) // Color data (not depth/stencil)
                            .base_mip_level(0) // Start at mip level 0 (full resolution)
                            .level_count(1) // Use 1 mip level (no mipmapping)
                            .base_array_layer(0) // Start at array layer 0
                            .layer_count(1) // Use 1 array layer (not an array texture)
                            .build(),
                    );
                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
}

// Creates the graphics pipeline - defines how to render our sprites
unsafe fn create_pipeline(_instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Load and compile shaders at runtime
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
                float32: [0.0, 0.0, 0.0, 1.0], // Black background - better for GPU compression
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

        let color_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(data.swapchain_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(color_clear_value);

        let color_attachments = &[color_attachment];
        let rendering_info = vk::RenderingInfo::builder()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(color_attachments);

        device.cmd_begin_rendering(command_buffer, &rendering_info);
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline,
        );

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(data.swapchain_extent.width as f32)
            .height(data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        device.cmd_set_viewport(command_buffer, 0, &[viewport]);

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

        // Push constants with view-projection matrix only
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

        // Instanced draw: 6 vertices per quad, data.sprite_count instances
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
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Load image from disk and convert to RGBA format
    let img = image::open("ferris.png")?.to_rgba8();
    let (width, height) = (img.width(), img.height());
    let size = u64::from(width * height * RGBA_BYTES_PER_PIXEL); // RGBA bytes per pixel

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
        // Create staging buffer and allocate CPU-visible memory for it
        let staging_buffer = device.create_buffer(&staging_buffer_info, None)?;

        let requirements = device.get_buffer_memory_requirements(staging_buffer);
        let memory_type = get_memory_type(
            instance,
            data,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, // CPU-accessible memory
        )?;

        let staging_alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let staging_buffer_memory = device.allocate_memory(&staging_alloc_info, None)?;
        device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0)?;

        // Copy image pixel data from CPU to staging buffer
        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(img.as_raw().as_ptr(), memory.cast(), size as usize);
        device.unmap_memory(staging_buffer_memory);

        // Create the actual GPU texture image
        data.texture_image = device.create_image(&image_info, None)?;

        // Allocate GPU-local memory for the texture
        let requirements = device.get_image_memory_requirements(data.texture_image);
        let memory_type = get_memory_type(
            instance,
            data,
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, // GPU-only memory (faster)
        )?;

        let image_alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        data.texture_image_memory = device.allocate_memory(&image_alloc_info, None)?;
        device.bind_image_memory(data.texture_image, data.texture_image_memory, 0)?;

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

        // Clean up staging resources
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

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
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    sprite_count: usize,
) -> Result<()> {
    unsafe {
        let size = (sprite_count * size_of::<SpriteCommand>()) as u64;

        // Create buffer in CPU-visible memory for frequent updates
        let (sprite_command_buffer, sprite_command_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER, // Storage buffer for shader access
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, // CPU-accessible
        )?;

        data.sprite_command_buffer = sprite_command_buffer;
        data.sprite_command_buffer_memory = sprite_command_buffer_memory;

        // Map the buffer permanently so we can update sprite data each frame
        data.sprite_command_buffer_mapped = device
            .map_memory(
                sprite_command_buffer_memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?
            .cast();

        Ok(())
    }
}

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    unsafe {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);

        let mut alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type(instance, data, requirements, properties)?);

        // Add device address allocation flag if buffer uses shader device address
        let mut alloc_flags_info = vk::MemoryAllocateFlagsInfo::builder();
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            alloc_flags_info = alloc_flags_info.flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            alloc_info = alloc_info.push_next(&mut alloc_flags_info);
        }

        let buffer_memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }
}

// Finds a memory type that meets the requirements and has desired properties
unsafe fn get_memory_type(
    instance: &Instance,
    data: &AppData,
    requirements: vk::MemoryRequirements,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    unsafe {
        // Get information about what types of memory this GPU has
        let memory = instance.get_physical_device_memory_properties(data.physical_device);

        // Find a memory type that both satisfies the requirements and has the properties we want
        (0..memory.memory_type_count)
            .find(|i| {
                // Check if this memory type is allowed by the requirements
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                // Check if this memory type has the properties we want (CPU-visible, GPU-local, etc.)
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
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

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(data.command_pool)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
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

        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;

        device.free_command_buffers(data.command_pool, &[command_buffer]);

        Ok(())
    }
}

unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(data.command_pool)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

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
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
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

        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;

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

fn calculate_scaling_and_offset(window_width: f32, window_height: f32) -> (f32, f32, f32) {
    let window_aspect = window_width / window_height;
    let logical_aspect = LOGICAL_WIDTH / LOGICAL_HEIGHT;

    let (scale, viewport_width, viewport_height) = if window_aspect > logical_aspect {
        // Window is wider than logical aspect ratio - pillarbox
        let scale = window_height / LOGICAL_HEIGHT;
        let viewport_width = LOGICAL_WIDTH * scale;
        (scale, viewport_width, window_height)
    } else {
        // Window is taller than logical aspect ratio - letterbox
        let scale = window_width / LOGICAL_WIDTH;
        let viewport_height = LOGICAL_HEIGHT * scale;
        (scale, window_width, viewport_height)
    };

    let offset_x = (window_width - viewport_width) * 0.5;
    let offset_y = (window_height - viewport_height) * 0.5;

    (scale, offset_x, offset_y)
}

fn create_sprite_transform(window_width: f32, window_height: f32) -> [[f32; 4]; 4] {
    let (scale, offset_x, offset_y) = calculate_scaling_and_offset(window_width, window_height);

    // Use glam for cleaner matrix math
    let logical_size = glam::vec2(LOGICAL_WIDTH, LOGICAL_HEIGHT);
    let window_size = glam::vec2(window_width, window_height);
    let offset = glam::vec2(offset_x, offset_y);

    // Calculate the actual viewport in normalized device coordinates
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

    // Parse command line arguments for sprite count and vsync
    let args: Vec<String> = env::args().collect();
    let sprite_count = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1)
    } else {
        1 // Default to 1 sprite if no argument provided
    };

    let vsync_enabled = args.iter().any(|arg| arg == "--vsync");

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
    let mut app = unsafe { App::create(&window, sprite_count, vsync_enabled)? };

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
            event: WindowEvent::RedrawRequested,
            ..
        } => unsafe {
            // Render one frame
            let _ = app.render(&window);
        },
        Event::AboutToWait => {
            // Request a new frame to be drawn
            window.request_redraw();
        }
        _ => {}
    })?;
    Ok(())
}
