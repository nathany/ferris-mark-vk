use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use std::env;
use std::mem::size_of;
use std::time::Instant;
use vulkanalia::prelude::v1_3::*;
use vulkanalia::vk::{DeviceV1_4, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    window as vk_window, Device, Entry, Instance,
};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

const VALIDATION_ENABLED: bool = false;
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
const MAX_FRAMES_IN_FLIGHT: usize = 2;

// Logical resolution constants (DO NOT CHANGE)
const LOGICAL_WIDTH: f32 = 1920.0;
const LOGICAL_HEIGHT: f32 = 1080.0;

// Initial window size constants (DO NOT CHANGE)
const INITIAL_WINDOW_WIDTH: u32 = 1920;
const INITIAL_WINDOW_HEIGHT: u32 = 1080;

// Physics constants
const GRAVITY: f32 = 0.95;
const BOUNCE_DAMPING: f32 = 0.85;

// Sprite data for physics simulation
#[derive(Clone, Debug)]
struct Sprite {
    pos_x: f32,
    pos_y: f32,
    vel_x: f32,
    vel_y: f32,
}

// Sprite helper functions using glam types
const fn sprite_quad(pos_x: f32, pos_y: f32, size_x: f32, size_y: f32) -> [Vertex; 4] {
    [
        Vertex::new([pos_x, pos_y], [0.0, 0.0]),
        Vertex::new([pos_x + size_x, pos_y], [1.0, 0.0]),
        Vertex::new([pos_x + size_x, pos_y + size_y], [1.0, 1.0]),
        Vertex::new([pos_x, pos_y + size_y], [0.0, 1.0]),
    ]
}

// Generate sprites with random positions and velocities
fn generate_sprites(count: usize) -> Vec<Sprite> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut sprites = Vec::new();
    let sprite_width = 99.0;
    let sprite_height = 70.0;

    // Simple PRNG based on sprite index
    for i in 0..count {
        let mut hasher = DefaultHasher::new();
        (i as u64).hash(&mut hasher);
        let seed = hasher.finish();

        // Generate pseudo-random values
        let rand1 = ((seed.wrapping_mul(16807) % 2147483647) as f32) / 2147483647.0;
        let rand2 = (((seed >> 16).wrapping_mul(16807) % 2147483647) as f32) / 2147483647.0;
        let rand3 = (((seed >> 32).wrapping_mul(16807) % 2147483647) as f32) / 2147483647.0;
        let rand4 = (((seed ^ 0xAAAAAAAA).wrapping_mul(16807) % 2147483647) as f32) / 2147483647.0;

        sprites.push(Sprite {
            pos_x: rand1 * (LOGICAL_WIDTH - sprite_width),
            pos_y: rand2 * (LOGICAL_HEIGHT - sprite_height),
            vel_x: (rand3 - 0.5) * 5.0, // Random velocity between -2.5 and 2.5
            vel_y: rand4 * 2.5 + 2.5,   // Random upward velocity between 2.5 and 5.0
        });
    }

    sprites
}

// Generate vertices from sprite positions
fn sprites_to_vertices(sprites: &[Sprite]) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    let sprite_width = 99.0;
    let sprite_height = 70.0;

    for sprite in sprites {
        let quad = sprite_quad(sprite.pos_x, sprite.pos_y, sprite_width, sprite_height);
        vertices.extend_from_slice(&quad);
    }

    vertices
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PushConstants {
    transform: [[f32; 4]; 4], // 4x4 transformation matrix
}

impl Vertex {
    const fn new(pos: [f32; 2], tex_coord: [f32; 2]) -> Self {
        Self { pos, tex_coord }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 2]>() as u32)
                .build(),
        ]
    }
}

// Indices for a single quad (will be reused for multiple sprites)
const QUAD_INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

#[derive(Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
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

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
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

struct AppData {
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    sprite_count: usize,
    sprites: Vec<Sprite>,
    last_update: Instant,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    descriptor_set_layout: vk::DescriptorSetLayout,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    frame: usize,
}

struct App {
    #[allow(dead_code)]
    entry: Entry,
    instance: Instance,
    device: Device,
    data: AppData,
}

impl App {
    unsafe fn create(window: &Window, sprite_count: usize) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
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
            vertex_buffer: vk::Buffer::null(),
            vertex_buffer_memory: vk::DeviceMemory::null(),
            index_buffer: vk::Buffer::null(),
            index_buffer_memory: vk::DeviceMemory::null(),
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
        };
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, window, window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&instance, &device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&instance, &device, &mut data)?;
        create_texture_sampler(&instance, &device, &mut data)?;
        create_vertex_buffer(&instance, &device, &mut data, sprite_count)?;
        create_index_buffer(&instance, &device, &mut data, sprite_count)?;
        create_pipeline(&instance, &device, &mut data)?;
        create_command_buffers(&instance, &device, &mut data)?;
        create_sync_objects(&instance, &device, &mut data)?;
        Ok(Self {
            entry,
            instance,
            device,
            data,
        })
    }

    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // Update sprite physics
        let now = Instant::now();
        let dt = now.duration_since(self.data.last_update).as_secs_f32();
        self.data.last_update = now;

        self.update_sprites(dt);

        let window_size = window.inner_size();
        let transform =
            create_sprite_transform(window_size.width as f32, window_size.height as f32);

        let in_flight_fence = self.data.in_flight_fences[self.data.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores
                [self.data.frame % self.data.image_available_semaphores.len()],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!("{}", e)),
        };

        self.device.reset_fences(&[in_flight_fence])?;

        let command_buffer = self.data.command_buffers[self.data.frame];
        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        record_command_buffer(
            &self.device,
            &self.data,
            command_buffer,
            image_index,
            &transform,
        )?;

        // Use frame-based acquire semaphore but image-based render finished semaphore
        let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
            .semaphore(
                self.data.image_available_semaphores
                    [self.data.frame % self.data.image_available_semaphores.len()],
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

        self.device
            .queue_submit2(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

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

        self.data.frame = (self.data.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn update_sprites(&mut self, dt: f32) {
        let sprite_width = 99.0;
        let sprite_height = 70.0;

        for sprite in &mut self.data.sprites {
            // Update position
            sprite.pos_x += sprite.vel_x * dt * 60.0; // Scale by 60 for ~60fps feel
            sprite.pos_y += sprite.vel_y * dt * 60.0;
            sprite.vel_y += GRAVITY * dt * 60.0;

            // Handle collisions with screen edges
            if sprite.pos_x + sprite_width > LOGICAL_WIDTH {
                sprite.vel_x *= -1.0;
                sprite.pos_x = LOGICAL_WIDTH - sprite_width;
            }
            if sprite.pos_x < 0.0 {
                sprite.vel_x *= -1.0;
                sprite.pos_x = 0.0;
            }
            if sprite.pos_y + sprite_height > LOGICAL_HEIGHT {
                sprite.vel_y *= -BOUNCE_DAMPING;
                sprite.pos_y = LOGICAL_HEIGHT - sprite_height;

                // Add some random bounce variation
                if sprite.vel_y.abs() < 0.5 {
                    sprite.vel_y -= 1.0;
                }
            }
            if sprite.pos_y < 0.0 {
                sprite.vel_y = 0.0;
                sprite.pos_y = 0.0;
            }
        }

        // Update vertex buffer with new positions
        unsafe {
            self.update_vertex_buffer().unwrap();
        }
    }

    unsafe fn update_vertex_buffer(&mut self) -> Result<()> {
        let vertices = sprites_to_vertices(&self.data.sprites);
        let size = (vertices.len() * size_of::<Vertex>()) as u64;

        // Map and update the vertex buffer
        let memory = self.device.map_memory(
            self.data.vertex_buffer_memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(vertices.as_ptr(), memory.cast(), vertices.len());
        self.device.unmap_memory(self.data.vertex_buffer_memory);

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.instance, &self.device, &mut self.data)?;
        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        for image_view in &self.data.swapchain_image_views {
            self.device.destroy_image_view(*image_view, None);
        }
        self.data.swapchain_image_views.clear();
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    unsafe fn destroy(&mut self) {
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
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);

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

unsafe fn create_instance(window: &Window, entry: &Entry, _data: &mut AppData) -> Result<Instance> {
    // Check loader version support
    let loader_version = entry
        .enumerate_instance_version()
        .unwrap_or(vk::make_version(1, 0, 0));
    log::info!(
        "Vulkan loader version: {}.{}.{}",
        vk::version_major(loader_version),
        vk::version_minor(loader_version),
        vk::version_patch(loader_version)
    );

    // Request Vulkan 1.4 if supported, otherwise fall back
    let requested_version =
        if vk::version_major(loader_version) >= 1 && vk::version_minor(loader_version) >= 4 {
            vk::make_version(1, 4, 0)
        } else {
            loader_version
        };

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Ferris Mark VK\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(requested_version);

    let extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let mut layers = Vec::new();
    if VALIDATION_ENABLED {
        check_validation_layer_support(entry)?;
        layers.push(VALIDATION_LAYER.as_ptr());
    }

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    let instance = entry.create_instance(&info, None)?;

    Ok(instance)
}

unsafe fn check_validation_layer_support(entry: &Entry) -> Result<()> {
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if available_layers.contains(&VALIDATION_LAYER) {
        Ok(())
    } else {
        Err(anyhow!("Validation layer requested but not supported."))
    }
}

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        // Check device API version support
        let device_version = properties.api_version;
        log::info!(
            "Physical device `{}` supports Vulkan {}.{}.{}",
            properties.device_name,
            vk::version_major(device_version),
            vk::version_minor(device_version),
            vk::version_patch(device_version)
        );

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            log::warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name,
                error
            );
        } else {
            log::info!("Selected physical device (`{}`).", properties.device_name);

            // Check if device supports Vulkan 1.4
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

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!("Insufficient swapchain support."));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!("Missing required device extensions."))
    }
}

unsafe fn create_logical_device(instance: &Instance, data: &mut AppData) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Check device properties to determine Vulkan version support
    let properties = instance.get_physical_device_properties(data.physical_device);
    let device_version = properties.api_version;
    let supports_vulkan_14 =
        vk::version_major(device_version) >= 1 && vk::version_minor(device_version) >= 4;

    // For now, use basic features regardless of version
    // We've verified the version support above
    if supports_vulkan_14 {
        log::info!("Vulkan 1.4 device detected - maintenance6 and enhanced features available");
    } else {
        log::info!("Using basic Vulkan features (pre-1.4 device)");
    }

    let mut sync2_features =
        vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);

    let mut dynamic_rendering_features =
        vk::PhysicalDeviceDynamicRenderingFeatures::builder().dynamic_rendering(true);

    let mut maintenance4_features =
        vk::PhysicalDeviceMaintenance4Features::builder().maintenance4(true);

    let mut maintenance5_features =
        vk::PhysicalDeviceMaintenance5Features::builder().maintenance5(true);

    let features = vk::PhysicalDeviceFeatures::builder();

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut sync2_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut maintenance4_features)
        .push_next(&mut maintenance5_features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
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

unsafe fn create_swapchain_image_views(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            let info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::_2D)
                .format(data.swapchain_format)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );

            device.create_image_view(&info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

unsafe fn create_pipeline(_instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
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
        .map_err(|e| anyhow!("Failed to compile vertex shader: {}", e))?;

    let frag_compiled = compiler
        .compile_into_spirv(
            &frag_shader_source,
            shaderc::ShaderKind::Fragment,
            "sprite.frag",
            "main",
            None,
        )
        .map_err(|e| anyhow!("Failed to compile fragment shader: {}", e))?;

    let vert_shader_code = vert_compiled.as_binary_u8();
    let frag_shader_code = frag_compiled.as_binary_u8();

    let vert_shader_module = create_shader_module(device, vert_shader_code)?;
    let frag_shader_module = create_shader_module(device, frag_shader_code)?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    let binding_descriptions = &[Vertex::binding_description()];
    let attribute_descriptions = &Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(attribute_descriptions);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let _viewports = &[viewport];
    let _scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

    // Create descriptor set layout for push descriptors
    let bindings = &[vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
        .bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    let set_layouts = &[data.descriptor_set_layout];
    let push_constant_ranges = &[vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(size_of::<PushConstants>() as u32)
        .build()];

    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);
    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vert_stage, frag_stage];
    let color_attachment_formats = &[data.swapchain_format];
    let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::builder()
        .color_attachment_formats(color_attachment_formats);

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

    let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?;
    data.pipeline = pipelines.0[0];

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode =
        std::slice::from_raw_parts(bytecode.as_ptr().cast(), bytecode.len() / size_of::<u32>());
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(std::mem::size_of_val(bytecode))
        .code(bytecode);

    Ok(device.create_shader_module(&info, None)?)
}

unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}

unsafe fn create_command_buffers(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    Ok(())
}

unsafe fn record_command_buffer(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
    image_index: usize,
    transform: &[[f32; 4]; 4],
) -> Result<()> {
    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(data.swapchain_extent);

    let color_clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.3, 0.5, 0.7, 1.0], // Blue background for logical area
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

    // Bind vertex buffer
    let vertex_buffers = &[data.vertex_buffer];
    let offsets = &[0_u64];
    device.cmd_bind_vertex_buffers(command_buffer, 0, vertex_buffers, offsets);

    // Bind index buffer
    device.cmd_bind_index_buffer(command_buffer, data.index_buffer, 0, vk::IndexType::UINT16);

    // Push descriptor for texture
    let image_info = vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(data.texture_image_view)
        .sampler(data.texture_sampler);

    let descriptor_write = vk::WriteDescriptorSet::builder()
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&image_info));

    device.cmd_push_descriptor_set(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        data.pipeline_layout,
        0,
        std::slice::from_ref(&descriptor_write),
    );

    // Push transformation matrix
    let push_constants = PushConstants {
        transform: *transform,
    };
    device.cmd_push_constants(
        command_buffer,
        data.pipeline_layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        std::slice::from_raw_parts(
            &push_constants as *const _ as *const u8,
            size_of::<PushConstants>(),
        ),
    );

    device.cmd_draw_indexed(
        command_buffer,
        (QUAD_INDICES.len() * data.sprite_count) as u32,
        1,
        0,
        0,
        0,
    );
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

unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let img = image::open("ferris.png")?.to_rgba8();
    let (width, height) = (img.width(), img.height());
    let size = (width * height * 4) as u64;

    // Create staging buffer
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(staging_buffer);
    let memory_type = get_memory_type(
        instance,
        data,
        requirements,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);

    let staging_buffer_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0)?;

    // Copy image data to staging buffer
    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    std::ptr::copy_nonoverlapping(img.as_raw().as_ptr(), memory.cast(), size as usize);
    device.unmap_memory(staging_buffer_memory);

    // Create texture image
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_SRGB)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::_1);

    data.texture_image = device.create_image(&image_info, None)?;

    let requirements = device.get_image_memory_requirements(data.texture_image);
    let memory_type = get_memory_type(
        instance,
        data,
        requirements,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);

    data.texture_image_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(data.texture_image, data.texture_image_memory, 0)?;

    // Transition and copy image
    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_texture_image_view(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let info = vk::ImageViewCreateInfo::builder()
        .image(data.texture_image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );

    data.texture_image_view = device.create_image_view(&info, None)?;
    Ok(())
}

unsafe fn create_texture_sampler(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    data.texture_sampler = device.create_sampler(&info, None)?;
    Ok(())
}

unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    _sprite_count: usize,
) -> Result<()> {
    let vertices = sprites_to_vertices(&data.sprites);
    let size = (vertices.len() * size_of::<Vertex>()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    std::ptr::copy_nonoverlapping(vertices.as_ptr(), memory.cast(), vertices.len());
    device.unmap_memory(staging_buffer_memory);

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    sprite_count: usize,
) -> Result<()> {
    // Generate indices for multiple sprites
    let mut indices = Vec::new();
    for i in 0..sprite_count {
        let base_vertex = (i * 4) as u16;
        for &index in QUAD_INDICES {
            indices.push(base_vertex + index);
        }
    }

    let size = (indices.len() * size_of::<u16>()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    std::ptr::copy_nonoverlapping(indices.as_ptr(), memory.cast(), indices.len());
    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);
    let memory_type = get_memory_type(instance, data, requirements, properties)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);

    let buffer_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn get_memory_type(
    instance: &Instance,
    data: &AppData,
    requirements: vk::MemoryRequirements,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);

    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: u64,
) -> Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    let regions = vk::BufferCopy::builder()
        .src_offset(0)
        .dst_offset(0)
        .size(size);

    device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[regions]);

    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

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

unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
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

unsafe fn create_sync_objects(
    _instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    // Create semaphores per swapchain image to avoid reuse issues
    for _ in 0..data.swapchain_images.len() {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
    }

    // Create fences and command buffers per frame in flight
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    Ok(())
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

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Parse command line arguments for sprite count
    let args: Vec<String> = env::args().collect();
    let sprite_count = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1)
    } else {
        1
    };

    println!("Rendering {sprite_count} sprites");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Ferris Mark VK - Sprite System")
        .with_inner_size(winit::dpi::LogicalSize::new(
            INITIAL_WINDOW_WIDTH,
            INITIAL_WINDOW_HEIGHT,
        ))
        .build(&event_loop)?;

    let mut app = unsafe { App::create(&window, sprite_count)? };

    event_loop.run(move |event, target| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            unsafe {
                app.destroy();
            }
            target.exit();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => unsafe {
            let _ = app.render(&window);
        },
        Event::AboutToWait => {
            window.request_redraw();
        }
        _ => {}
    })?;
    Ok(())
}
