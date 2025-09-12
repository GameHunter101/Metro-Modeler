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
    fn advance_lod(&self) -> LOD {}

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

fn cut_shapes(shapes: Vec<Vec<Point>>) -> Vec<Vec<Point>> {
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
                .map(|(i, point)| [*point, shape[i + 1]])
                .collect::<Vec<_>>()
        })
        .collect();

    let (vertices, adjacency_list) = segments_to_adjacency_list(&mut segments);

    let dcel = DCEL::new(&vertices, &adjacency_list);

    /* let shape_indices_to_face_indices: HashMap<usize, HashSet<usize>> = dcel
    .faces()
    .iter()
    .enumerate()
    .map(|(face_index, face)| {
        let shape_indices: HashSet<usize> = face
            .iter()
            .flat_map(|vert_idx| {
                if let Some(shape_idx) = verts_to_shape.get(&order_point(vertices[*vert_idx])) {
                    Some(*shape_idx)
                } else {
                    None
                }
            })
            .collect();
        (shape_indices, vec![face_index])
    })
    .collect(); */
    let mut shape_indices_to_face_indices: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (face_index, face) in dcel.faces().iter().enumerate() {
        let shapes_of_face: HashSet<usize> = face
            .iter()
            .flat_map(|vert_idx| {
                verts_to_shape.get(&order_point(vertices[*vert_idx])).copied()
            }).collect();
        for shape_index in shapes_of_face {
            if let Some(faces) = shape_indices_to_face_indices.get_mut(&shape_index) {
                faces.insert(face_index);
            } else {
                shape_indices_to_face_indices.insert(shape_index, HashSet::from_iter(vec![face_index]));
            }
        }
    }

    todo!()
}
