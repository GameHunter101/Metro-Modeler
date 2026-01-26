use image::{EncodableLayout, ImageBuffer, Rgba};
use rand::Rng;
use v4::{
    builtin_components::transform_component::TransformComponent,
    component,
    ecs::{
        component::{Component, ComponentDetails, ComponentId, ComponentSystem, UpdateParams},
        compute::Compute,
        material::{GeneralTexture, Material, ShaderAttachment},
    }, engine_support::texture_support::Texture,
};
use wgpu::{CommandEncoder, Device, Extent3d, Queue, TextureUsages};

use crate::tensor_field::{EvalEigenvectors, GRID_SIZE, Point, TensorField};

#[component]
pub struct FieldVisualizationComponent {
    compute: ComponentId,
    material: ComponentId,
    street_transform: ComponentId,
    #[default(0)]
    time: u32,
    #[default(true)]
    show: bool,
}

#[async_trait::async_trait]
impl ComponentSystem for FieldVisualizationComponent {
    async fn update(
        &mut self,
        UpdateParams {
            queue,
            computes,
            input_manager,
            materials,
            other_components,
            ..
        }: UpdateParams<'_, '_>,
    ) -> v4::ecs::actions::ActionQueue {
        if self.show {
            let compute = computes
                .iter()
                .filter(|comp| comp.id() == self.compute)
                .next()
                .unwrap();

            self.time += 1;

            let time_attachment = &compute.input_attachments()[3];
            if let ShaderAttachment::Buffer(time_buf) = time_attachment {
                queue.write_buffer(time_buf.buffer(), 0, bytemuck::cast_slice(&[self.time]));
            }
        }

        let mut other_components = other_components.lock().unwrap();

        let street_transform = other_components
            .iter_mut()
            .filter(|comp| comp.id() == self.street_transform)
            .next();

        if input_manager.key_pressed(winit::keyboard::KeyCode::KeyT) {
            self.show = !self.show;

            let mut materials = materials.lock().unwrap();
            if let Some(mat) = materials
                .iter_mut()
                .filter(|mat| mat.id() == self.material)
                .next()
            {
                mat.set_enabled_state(self.show);
            }

            if let Some(street_transform) = street_transform
                && let Some(street_transform_component) =
                    street_transform.downcast_mut::<TransformComponent>()
            {
                street_transform_component.set_scale(
                    if !self.show { 0.5_f32 } else { 1.0 } * nalgebra::Vector3::new(1.0, 1.0, 1.0),
                );
            }
        }

        Vec::new()
    }

    fn command_encoder_operations(
        &self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        _other_components: &[&Component],
        materials: &[Material],
        computes: &[Compute],
    ) {
        if !self.show {
            return;
        }

        let compute = computes
            .iter()
            .filter(|comp| comp.id() == self.compute)
            .next()
            .unwrap();

        let material = materials
            .iter()
            .filter(|mat| mat.id() == self.material)
            .next()
            .unwrap();

        let output = compute.output_attachments().unwrap();

        let vis = &material.attachments()[0];

        if let ShaderAttachment::Texture(output_tex) = output
            && let ShaderAttachment::Texture(vis_tex) = vis
        {
            encoder.copy_texture_to_texture(
                output_tex.texture.texture().as_image_copy(),
                vis_tex.texture.texture().as_image_copy(),
                Extent3d {
                    width: GRID_SIZE,
                    height: GRID_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

fn smooth_field(eigenvector: Point) -> [u8; 4] {
    let field_x = if eigenvector.x >= 0.0 {
        eigenvector
    } else {
        -eigenvector
    };

    let field_y = if eigenvector.y >= 0.0 {
        eigenvector
    } else {
        -eigenvector
    };

    [field_x * 255.0, field_y * 255.0]
        .into_iter()
        .flat_map(|p| [p.x as u8, p.y as u8])
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub fn create_visualizer_textures(
    device: &Device,
    queue: &Queue,
    tensor_field: &TensorField,
) -> ([GeneralTexture; 3], [GeneralTexture; 2], GeneralTexture) {
    let mut rng = rand::rng();

    let visualization_input_img = ImageBuffer::from_fn(GRID_SIZE, GRID_SIZE, |_, _| {
        let val1 = rng.random();
        let val2 = rng.random();
        Rgba([val1, val2, 0, 0])
    });
    let visualization_output_img = visualization_input_img.clone();
    let mut major_eigenvector_img = visualization_output_img.clone();
    let mut minor_eigenvector_img = major_eigenvector_img.clone();
    let mut blending_img = minor_eigenvector_img.clone();
    let tensorfield_vis_img = blending_img.clone();

    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            let eigenvectors = tensor_field
                .evaluate_smoothed_field_at_point(Point::new(x as f32, y as f32))
                .eigenvectors();
            let major_eigenvector = eigenvectors.major.normalize();
            let minor_eigenvector = eigenvectors.minor.normalize();

            major_eigenvector_img.put_pixel(x, y, Rgba(smooth_field(major_eigenvector)));
            minor_eigenvector_img.put_pixel(x, y, Rgba(smooth_field(minor_eigenvector)));

            let major_w_x = ((major_eigenvector.x * major_eigenvector.x) * 255.0) as u8;
            let minor_w_x = ((minor_eigenvector.x * minor_eigenvector.x) * 255.0) as u8;

            blending_img.put_pixel(x, y, Rgba([major_w_x, minor_w_x, 0, 0]));
        }
    }

    let [visualization_input_tex, visualization_output_tex] = [
        (visualization_input_img, TextureUsages::empty()),
        (visualization_output_img, TextureUsages::COPY_SRC),
    ]
    .map(|(img, usages)| {
        Texture::from_bytes(
            img.as_bytes(),
            (GRID_SIZE, GRID_SIZE),
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::StorageTextureAccess::ReadWrite),
            false,
            usages,
        )
    });

    let [major_eigenvector_tex, minor_eigenvector_tex] =
        [major_eigenvector_img, minor_eigenvector_img].map(|img| {
            Texture::from_bytes(
                img.as_bytes(),
                (GRID_SIZE, GRID_SIZE),
                device,
                queue,
                wgpu::TextureFormat::Rgba8Unorm,
                None,
                false,
                TextureUsages::empty(),
            )
        });

    let [blending_tex, tensorfield_vis_tex] = [
        (blending_img, TextureUsages::empty()),
        (tensorfield_vis_img, TextureUsages::COPY_DST),
    ]
    .map(|(img, usage)| {
        Texture::from_bytes(
            img.as_bytes(),
            (GRID_SIZE, GRID_SIZE),
            device,
            queue,
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            true,
            usage,
        )
    });

    (
        [
            visualization_input_tex,
            major_eigenvector_tex,
            minor_eigenvector_tex,
        ],
        [blending_tex, tensorfield_vis_tex],
        visualization_output_tex,
    )
}

