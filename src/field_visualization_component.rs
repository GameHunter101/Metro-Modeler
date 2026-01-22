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
}

#[async_trait::async_trait]
impl ComponentSystem for FieldVisualizationComponent {
    async fn update(&mut self, UpdateParams { queue, computes, .. }: UpdateParams<'_,'_>) -> v4::ecs::actions::ActionQueue {
        let compute = computes
            .iter()
            .filter(|comp| comp.id() == self.compute)
            .next()
            .unwrap();

        self.time += 1;

        let time_attachment = &compute.input_attachments()[2];
        if let ShaderAttachment::Buffer(time_buf) = time_attachment {
            queue.write_buffer(time_buf.buffer(), 0, bytemuck::cast_slice(&[self.time]));
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
        let compute = computes
            .iter()
            .filter(|comp| comp.id() == self.compute)
            .next()
            .unwrap();

        let material = materials.iter().filter(|mat| mat.id() == self.material).next().unwrap();

        let output = compute.output_attachments().unwrap();
        let input = &compute.input_attachments()[0];

        let vis = &material.attachments()[0];

        if let ShaderAttachment::Texture(output_tex) = output
            && let ShaderAttachment::Texture(input_tex) = input
        && let ShaderAttachment::Texture(vis_tex) = vis
        {
            encoder.copy_texture_to_texture(
                output_tex.texture.texture().as_image_copy(),
                input_tex.texture.texture().as_image_copy(),
                Extent3d {
                    width: GRID_SIZE,
                    height: GRID_SIZE,
                    depth_or_array_layers: 1,
                },
            );

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
