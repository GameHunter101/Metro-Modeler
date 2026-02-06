use algoe::bivector::Bivector;
use image::{EncodableLayout, ImageBuffer, Rgba};
use nalgebra::Vector3;
use rand::Rng;
use v4::{
    builtin_actions::EntityToggleAction,
    builtin_components::transform_component::TransformComponent,
    component,
    ecs::{
        component::{ComponentDetails, ComponentId, ComponentSystem, UpdateParams},
        entity::EntityId,
    },
    engine_support::texture_support::{CompleteTexture, TextureBundle, TextureProperties},
};
use wgpu::{Device, Queue, TextureUsages};

use crate::tensor_field::{EvalEigenvectors, GRID_SIZE, Point, TensorField};

#[component]
pub struct FieldVisualizationComponent {
    material: ComponentId,
    street_mat: ComponentId,
    street_transform: ComponentId,
    plot_entity: EntityId,
    #[default(true)]
    show: bool,
}

impl ComponentSystem for FieldVisualizationComponent {
    fn update(
        &mut self,
        UpdateParams {
            input_manager,
            materials,
            other_components,
            ..
        }: UpdateParams<'_, '_>,
    ) -> v4::ecs::actions::ActionQueue {
        let street_transform = other_components
            .iter_mut()
            .filter(|comp| comp.id() == self.street_transform)
            .next();

        if input_manager.key_pressed(winit::keyboard::KeyCode::KeyT) {
            self.show = !self.show;

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
                street_transform_component.set_position(if self.show {
                    Vector3::zeros()
                } else {
                    GRID_SIZE as f32 / 2.0 * Vector3::new(1.0, 0.0, 1.0)
                });
                street_transform_component.set_scale(
                    if self.show {
                        1.0_f32
                    } else {
                        GRID_SIZE as f32 / 2.0
                    } * nalgebra::Vector3::new(1.0, 1.0, 1.0),
                );
                street_transform_component.set_rotation(
                    Bivector::new(
                        0.0,
                        if self.show {
                            0.0
                        } else {
                            -std::f32::consts::FRAC_PI_4
                        },
                        0.0,
                    )
                    .exponentiate(),
                );
            }

            if let Some(street_mat) = materials
                .iter_mut()
                .filter(|mat| mat.id() == self.street_mat)
                .next()
            {
                street_mat.set_immediate_data(bytemuck::cast_slice(&[!self.show as u32]));
            }

            vec![Box::new(EntityToggleAction(self.plot_entity, None))]
        } else {
            Vec::new()
        }
    }

    /* fn command_encoder_operations(
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

        let output = compute.attachments().unwrap();

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
    } */
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
) -> ([CompleteTexture; 3], CompleteTexture, CompleteTexture) {
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

    let [visualization_input_tex, visualization_output_tex] =
        [visualization_input_img, visualization_output_img].map(|img| {
            TextureBundle::from_bytes(
                img.as_bytes(),
                (GRID_SIZE, GRID_SIZE),
                device,
                queue,
                TextureProperties {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    storage_texture: Some(wgpu::StorageTextureAccess::ReadWrite),
                    is_sampled: false,
                    extra_usages: TextureUsages::TEXTURE_BINDING,
                    ..Default::default()
                },
            )
        });

    let [major_eigenvector_tex, minor_eigenvector_tex] =
        [major_eigenvector_img, minor_eigenvector_img].map(|img| {
            TextureBundle::from_bytes(
                img.as_bytes(),
                (GRID_SIZE, GRID_SIZE),
                device,
                queue,
                TextureProperties {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    storage_texture: None,
                    is_sampled: false,
                    extra_usages: TextureUsages::empty(),
                    ..Default::default()
                },
            )
        });

    let blending_tex = TextureBundle::from_bytes(
        blending_img.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        TextureProperties {
            format: wgpu::TextureFormat::Rgba8Unorm,
            storage_texture: None,
            is_sampled: true,
            extra_usages: TextureUsages::TEXTURE_BINDING,
            ..Default::default()
        },
    );

    (
        [
            visualization_input_tex,
            major_eigenvector_tex,
            minor_eigenvector_tex,
        ],
        blending_tex,
        visualization_output_tex,
    )
}
