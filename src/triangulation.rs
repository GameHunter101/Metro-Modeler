use nalgebra::Vector3;

fn triangulate<T>(face: &[T]) -> Vec<[u32; 3]> {
    triangulate_helper(face, vec![true; face.len()], face.len())
}

/// Invariant: valid_vertices_count >= 3
fn triangulate_helper<T>(
    face: &[T],
    mut valid_vertices_mask: Vec<bool>,
    valid_vertices_count: usize,
) -> Vec<[u32; 3]> {
    if valid_vertices_count < 3 {
        return Vec::new();
    }

    if valid_vertices_count == 3 {
        let tri: [u32; 3] = valid_vertices_mask
            .iter()
            .enumerate()
            .flat_map(|(i, &valid)| if valid { Some(i as u32) } else { None })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        return vec![tri];
    }

    if face.len() % 2 == 1 {
        valid_vertices_mask[face.len() - 1] = false;
        Some(triangle_around_vertex(face.len() - 1, &valid_vertices_mask))
            .into_iter()
            .chain(triangulate_helper(
                &face[0..face.len() - 1],
                valid_vertices_mask,
                valid_vertices_count - 1,
            ))
            .collect()
    } else {
        let valid_indices: Vec<usize> = valid_vertices_mask
            .iter()
            .enumerate()
            .flat_map(|(i, valid)| if *valid { Some(i) } else { None })
            .collect();

        let new_triangles: Vec<[u32; 3]> = valid_indices
            .iter()
            .enumerate()
            .flat_map(|(i, vert_index)| {
                if i % 2 == 0 {
                    valid_vertices_mask[*vert_index] = false;
                    Some(triangle_around_vertex(*vert_index, &valid_vertices_mask))
                } else {
                    None
                }
            })
            .collect();

        let new_triangles_count = new_triangles.len();

        new_triangles
            .into_iter()
            .chain(triangulate_helper(
                face,
                valid_vertices_mask,
                valid_vertices_count - new_triangles_count,
            ))
            .collect()
    }
}

fn triangle_around_vertex(vert_index: usize, ignored_vertices_mask: &[bool]) -> [u32; 3] {
    let mut prev = None;
    let mut next = None;

    for dist in 1..ignored_vertices_mask.len() {
        let cur_prev_index = (vert_index as isize - dist as isize)
            .rem_euclid(ignored_vertices_mask.len() as isize) as usize;
        if prev.is_none() && ignored_vertices_mask[cur_prev_index] {
            prev = Some(cur_prev_index as u32);
        }

        let cur_next_index = (vert_index + dist) % ignored_vertices_mask.len();
        if next.is_none() && ignored_vertices_mask[cur_next_index] {
            next = Some(cur_next_index as u32);
        }

        if let Some(prev) = prev
            && let Some(next) = next
        {
            return [prev, vert_index as u32, next];
        }
    }

    panic!(
        "Invariant not maintained while finding triangle for vertex {vert_index}: valid_vertices_count < 3"
    );
}

pub fn triangulate_faces(faces: &[Vec<Vector3<f32>>]) -> Vec<u32> {
    let mut index_offset = 0;

    faces
        .iter()
        .flat_map(|face| {
            let indices = triangulate(face)
                .into_iter()
                .flatten()
                .map(|v| v + index_offset)
                .collect::<Vec<_>>();
            index_offset += face.len() as u32;

            indices
        })
        .collect()
}

#[cfg(test)]
mod test {
    use crate::tensor_field::Point;

    use super::triangulate;

    #[test]
    fn trianglulating_triangle_does_nothing() {
        let face = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
        ];

        let triangulated = triangulate(&face);
        assert_eq!(triangulated.len(), 1);
        assert_eq!(triangulated[0], [0, 1, 2]);
    }

    #[test]
    fn triangulating_quad() {
        let face = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];

        let triangulated = triangulate(&face);
        assert_eq!(triangulated.len(), 2);

        let expected_triangulated_faces = vec![[3, 0, 1], [1, 2, 3]];

        assert_eq!(triangulated, expected_triangulated_faces);
    }

    #[test]
    fn triangulating_ngon_single_extra_layer() {
        let face = vec![
            Point::new(1.12, 1.81),
            Point::new(4.25, 1.62),
            Point::new(4.9, 3.06),
            Point::new(4.83, 4.85),
            Point::new(2.0, 5.0),
            Point::new(0.86, 3.63),
        ];

        let triangulated = triangulate(&face);
        assert_eq!(triangulated.len(), 4);

        let expected_triangulated_faces = vec![[5, 0, 1], [1, 2, 3], [3, 4, 5], [1, 3, 5]];

        assert_eq!(triangulated, expected_triangulated_faces);
    }

    #[test]
    fn triangulating_ngon_single_extra_layer_odd_n() {
        let face = vec![
            Point::new(1.12, 1.81),
            Point::new(4.25, 1.62),
            Point::new(4.9, 3.06),
            Point::new(4.83, 4.85),
            Point::new(2.0, 5.0),
            Point::new(0.86, 3.63),
            Point::new(0.67, 3.05),
        ];

        let triangulated = triangulate(&face);
        assert_eq!(triangulated.len(), 5);

        let expected_triangulated_faces =
            vec![[5, 6, 0], [5, 0, 1], [1, 2, 3], [3, 4, 5], [1, 3, 5]];

        assert_eq!(triangulated, expected_triangulated_faces);
    }

    #[test]
    fn triangulating_ngon_two_extra_layers() {
        let face = vec![
            Point::new(1.12, 1.81),
            Point::new(4.25, 1.62),
            Point::new(4.9, 3.06),
            Point::new(4.83, 4.85),
            Point::new(2.0, 5.0),
            Point::new(0.86, 3.63),
            Point::new(0.67, 3.05),
            Point::new(0.65, 2.27),
        ];

        let triangulated = triangulate(&face);
        assert_eq!(triangulated.len(), 6);

        let expected_triangulated_faces = vec![
            [7, 0, 1],
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 1, 3],
            [3, 5, 7],
        ];

        assert_eq!(triangulated, expected_triangulated_faces);
    }
}
