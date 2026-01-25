use v4::{
    component,
    ecs::{
        component::{Component, ComponentDetails, ComponentId, ComponentSystem, UpdateParams},
        compute::Compute,
        material::{Material, ShaderAttachment},
    },
};
use wgpu::{CommandEncoder, Device, Extent3d, Queue};

use crate::tensor_field::GRID_SIZE;

#[component]
pub struct FieldVisualizationComponent {
    compute: ComponentId,
    material: ComponentId,
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
            queue, computes, input_manager, materials, ..
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


        if input_manager.key_pressed(winit::keyboard::KeyCode::KeyT) {
            self.show = !self.show;

            let mut materials = materials.lock().unwrap();
            if let Some(mat) = materials.iter_mut().filter(|mat| mat.id() == self.material).next() {
                mat.set_enabled_state(self.show);
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
