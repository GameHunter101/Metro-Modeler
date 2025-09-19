use cool_utils::data_structures::dcel::DCEL;
use dyn_clone::DynClone;
use rand::prelude::*;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use crate::{
    street_graph::{OrderedPoint, order_point, segments_to_adjacency_list},
    tensor_field::Point,
};

const MAX_SUBDIVISION_COUNT: u32 = 4;
const FOUNDATION_HEIGHT: u32 = 3;

#[derive(Debug, PartialEq, PartialOrd, Eq)]
enum LOD {
    LOD2,
    LOD1,
    LOD0,
}

impl std::cmp::Ord for LOD {
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            LOD::LOD2 => match other {
                LOD::LOD2 => Ordering::Equal,
                LOD::LOD1 | LOD::LOD0 => Ordering::Greater,
            },
            LOD::LOD1 => match other {
                LOD::LOD2 => Ordering::Less,
                LOD::LOD1 => Ordering::Equal,
                LOD::LOD0 => Ordering::Greater,
            },
            LOD::LOD0 => {
                if let LOD::LOD0 = other {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            }
        }
    }
}

/// [lower uniform scale, upper uniform scale]
type ScaleXZ = [f32; 2];

/// All heights are measured in discrete storeys. Storeys are adjustable
trait GrammarShape: DynClone {
    fn is_terminal(&self) -> bool {
        false
    }

    fn advance_lod(&self) -> LOD;

    fn advance(
        &self,
        previous_shape: Option<&dyn GrammarShape>,
        rng: &mut dyn RngCore,
        max_building_height: u32,
    ) -> Vec<Box<dyn GrammarShape>>;
}

#[derive(Debug, Clone)]
struct Footprint {
    points: Vec<Point>,
}

impl Footprint {
    fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
}

impl GrammarShape for Footprint {
    fn advance_lod(&self) -> LOD {
        LOD::LOD2
    }

    fn advance(
        &self,
        previous_shape: Option<&dyn GrammarShape>,
        rng: &mut dyn RngCore,
        max_building_height: u32,
    ) -> Vec<Box<dyn GrammarShape>> {
        let foundation: Box<dyn GrammarShape> =
            Box::new(Foundation::new(self.points.clone(), FOUNDATION_HEIGHT));
        let shape_count = rng.random_range(1..=MAX_SUBDIVISION_COUNT);
        let shapes: Vec<Box<dyn GrammarShape>> = if rng.random() {
            vec![Box::new(BuildingShape::new(
                Point::zeros(),
                self.points.clone(),
                rng.random_range((FOUNDATION_HEIGHT + 1)..max_building_height),
            ))]
        } else {
            // TODO: Maybe planar subdivision overlay from textbook?
            Vec::new()
        };

        Some(foundation).into_iter().chain(shapes).collect()
    }
}

#[derive(Debug, Clone)]
struct Foundation {
    shape: Vec<Point>,
    height: u32,
}

impl Foundation {
    fn new(shape: Vec<Point>, height: u32) -> Self {
        Self { shape, height }
    }
}

impl GrammarShape for Foundation {
    fn advance_lod(&self) -> LOD {
        LOD::LOD0
    }

    fn advance(
        &self,
        previous_shape: Option<&dyn GrammarShape>,
        rng: &mut dyn RngCore,
        max_building_height: u32,
    ) -> Vec<Box<dyn GrammarShape>> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct BuildingShape {
    pos: Point,
    shape: Vec<Point>,
    height: u32,
    scale: ScaleXZ,
}

impl BuildingShape {
    fn new(pos: Point, shape: Vec<Point>, height: u32) -> Self {
        Self {
            pos,
            shape,
            height,
            scale: [1.0, 1.0],
        }
    }
}

impl GrammarShape for BuildingShape {
    fn advance_lod(&self) -> LOD {
        LOD::LOD1
    }

    fn advance(
        &self,
        previous_shape: Option<&dyn GrammarShape>,
        rng: &mut dyn RngCore,
        max_building_height: u32,
    ) -> Vec<Box<dyn GrammarShape>> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct TierBottom {
    pos: Point,
    shape: Vec<Point>,
    bottom: u32,
    height: u32,
    smooth_transition: bool,
    scale: ScaleXZ,
}

impl GrammarShape for TierBottom {
    fn advance_lod(&self) -> LOD {
        todo!()
    }

    fn advance(
        &self,
        previous_shape: Option<&dyn GrammarShape>,
        rng: &mut dyn RngCore,
        max_building_height: u32,
    ) -> Vec<Box<dyn GrammarShape>> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct TierTop {
    pos: Point,
    shape: Vec<Point>,
    height: u32,
    scale: ScaleXZ,
}

fn iterate_grammar(
    shapes: Vec<Box<dyn GrammarShape>>,
    rng: &mut dyn RngCore,
    max_building_height: u32,
) -> Vec<Box<dyn GrammarShape>> {
    let index_to_update = shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| (i, shape.advance_lod()))
        .fold((0_usize, shapes[0].advance_lod()), |acc, e| {
            if e.1 > acc.1 { e } else { acc }
        })
        .0;
    shapes[0..index_to_update]
        .iter()
        .map(|grammar_shape| dyn_clone::clone_box(&**grammar_shape))
        .into_iter()
        .chain(shapes[index_to_update].advance(
            if index_to_update > 0 {
                Some(&*shapes[index_to_update - 1])
            } else {
                None
            },
            rng,
            max_building_height,
        ))
        .chain(
            shapes[(index_to_update + 1)..]
                .iter()
                .map(|grammar_shape| dyn_clone::clone_box(&**grammar_shape)),
        )
        .collect()
}

fn cut_shapes(shapes: Vec<Vec<Point>>, shapes_cut: usize) -> Vec<Vec<Point>> {
    if shapes.len() == 1 {
        return shapes;
    }

    let first_shape = shapes[1].clone();
    let shapes_len = shapes.len();

    let other_cut_shapes: Vec<Vec<Point>> = shapes
        .into_iter()
        .enumerate()
        .flat_map(|(shape_index, shape)| {
            if shape_index == 0 {
                return None;
            }
            let shape_index = shape_index + 1 + shapes_cut;
            let (all_vertices, disjoint_faces, vert_indices_to_shape) =
                split_shapes_to_disjoint_faces(vec![first_shape.clone(), shape]);

            disjoint_faces
                .into_iter()
                .filter(|face| {
                    face_to_shape(face, &all_vertices, &vert_indices_to_shape) == Some(shape_index)
                })
                .next()
                .map(|face_by_indices| {
                    face_by_indices
                        .iter()
                        .map(|vert_idx| all_vertices[*vert_idx])
                        .collect::<Vec<_>>()
                })
        })
        .collect();

    let mut final_shapes = Vec::with_capacity(shapes_len);
    final_shapes.push(first_shape);
    final_shapes.extend(other_cut_shapes);
    todo!()
}

fn split_shapes_to_disjoint_faces(
    shapes: Vec<Vec<Point>>,
) -> (Vec<Point>, Vec<Vec<usize>>, HashMap<usize, usize>) {
    let verts_to_shape: HashMap<OrderedPoint, usize> = shapes
        .iter()
        .enumerate()
        .flat_map(|(shape_index, shape)| {
            shape
                .iter()
                .map(|point| (order_point(*point), shape_index))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut segments: Vec<[Point; 2]> = shapes
        .into_iter()
        .flat_map(|shape| {
            shape
                .iter()
                .enumerate()
                .map(|(i, point)| [*point, shape[(i + 1) % shape.len()]])
                .collect::<Vec<_>>()
        })
        .collect();

    let (vertices, adjacency_list) = segments_to_adjacency_list(&mut segments);

    let dcel = DCEL::new(&vertices, &adjacency_list);

    let vert_indices_to_shape = vertices
        .iter()
        .enumerate()
        .map(|(i, vert)| (i, verts_to_shape[&order_point(*vert)]))
        .collect();

    (vertices, dcel.faces().to_vec(), vert_indices_to_shape)
}

fn face_to_shape(
    face: &[usize],
    vertices: &[Point],
    verts_to_shape: &HashMap<usize, usize>,
) -> Option<usize> {
    let start_vertex_index = (0..face.len()).fold(0, |acc, vert_index| {
        let current_vert = vertices[face[vert_index]];
        let previous_vert = vertices[face[acc]];
        if verts_to_shape.contains_key(&face[vert_index]) {
            match current_vert.x.total_cmp(&previous_vert.x) {
                Ordering::Less => vert_index,
                Ordering::Equal => {
                    if current_vert.y < previous_vert.y {
                        vert_index
                    } else {
                        acc
                    }
                }
                Ordering::Greater => acc,
            }
        } else {
            acc
        }
    });

    let mut between_intersection_points = false;
    for i in 0..face.len() {
        let current_vert_index = face[(start_vertex_index + i) % face.len()];

        if !verts_to_shape.contains_key(&current_vert_index) {
            between_intersection_points = !between_intersection_points;
        }

        if !between_intersection_points {
            return Some(verts_to_shape[&current_vert_index]);
        }
    }

    None
}

mod test {
    use std::collections::HashSet;

    use crate::{building_generation::split_shapes_to_disjoint_faces, tensor_field::Point};

    #[test]
    fn simple_overlapping_shapes_are_split() {
        let rect = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let triangle = vec![
            Point::new(1.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(3.0, 1.0),
        ];

        let (_, split_faces, _) = split_shapes_to_disjoint_faces(vec![rect, triangle]);

        assert_eq!(split_faces.len(), 3);

        let real_lengths: HashSet<usize> = split_faces.iter().map(|face| face.len()).collect();
        let expected_lengths = HashSet::from_iter(vec![5, 3, 4]);

        assert_eq!(real_lengths.difference(&expected_lengths).count(), 0);
    }
}
