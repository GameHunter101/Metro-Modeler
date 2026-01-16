use cool_utils::data_structures::dcel::DCEL;
use dyn_clone::DynClone;
use nalgebra::{Matrix2, Vector3};
use rand::prelude::*;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use crate::{
    Vertex,
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
    println!("Shapes:");
    shapes.iter().for_each(|shape| {
        println!(
            "polygon({:?})",
            shape
                .iter()
                .map(|vert| (vert.x, vert.y))
                .collect::<Vec<_>>()
        )
    });

    if shapes.len() <= 1 {
        return shapes;
    }

    let first_shape = shapes[0].clone();
    let first_shape_set = HashSet::from_iter(first_shape.iter().map(|vert| order_point(*vert)));
    let shapes_len = shapes.len();

    let other_cut_shapes: Vec<Vec<Point>> = shapes
        .into_iter()
        .enumerate()
        .flat_map(|(shape_index, shape)| {
            if shape_index <= shapes_cut {
                return None;
            }

            if is_face_in_face(&shape, &first_shape) || is_face_in_face(&first_shape, &shape) {
                return None;
            }

            let (all_vertices, disjoint_faces, vert_indices_to_shape) =
                split_shapes_to_disjoint_faces(vec![first_shape.clone(), shape]);

            dbg!(disjoint_faces.len());
            println!(
                "VERTICES: {:?}",
                all_vertices.iter().map(|v| (v.x, v.y)).collect::<Vec<_>>()
            );
            println!(
                "DISJOINT FACES: {:?}",
                disjoint_faces
                    .iter()
                    .map(|face| face
                        .iter()
                        .map(|&vert| (all_vertices[vert].x, all_vertices[vert].y))
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );

            let temp = match disjoint_faces.len() {
                1 => None,
                2 => {
                    if disjoint_faces[0].len() == first_shape.len()
                        && vert_indices_to_shape[&disjoint_faces[0][0]] == 0
                    {
                        Some(1_usize)
                    } else {
                        Some(0)
                    }
                }
                _ => {
                    let faces_as_sets: Vec<HashSet<usize>> = disjoint_faces
                        .iter()
                        .map(|face| HashSet::from_iter(face.clone()))
                        .collect();
                    let pairs = [(0_usize, 1_usize), (0, 2), (1, 2)];

                    let joined_pair = pairs
                        .into_iter()
                        .filter(|(first, second)| {
                            let union: HashSet<OrderedPoint> = faces_as_sets[*first]
                                .union(&faces_as_sets[*second])
                                .map(|idx| order_point(all_vertices[*idx]))
                                .collect();
                            union.intersection(&first_shape_set).count() == first_shape.len()
                                && union.len() == first_shape.len() + 2
                        })
                        .next()
                        .unwrap();
                    HashSet::<usize>::from_iter([0_usize, 1, 2])
                        .difference(&HashSet::from_iter([joined_pair.0, joined_pair.1]))
                        .next()
                        .copied()
                }
            }
            .map(|disjoint_face_index| {
                disjoint_faces[disjoint_face_index]
                    .iter()
                    .map(|vert_index| all_vertices[*vert_index])
                    .collect::<Vec<_>>()
            });
            // dbg!(&temp);
            temp
        })
        .collect();

    dbg!(other_cut_shapes.len());

    let mut final_shapes = Vec::with_capacity(shapes_len);
    final_shapes.push(first_shape);
    final_shapes.extend(cut_shapes(other_cut_shapes, 0));
    final_shapes
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
        .flat_map(|(i, vert)| {
            verts_to_shape
                .get(&order_point(*vert))
                .map(|point| (i, *point))
        })
        .collect();

    (vertices, dcel.faces().to_vec(), vert_indices_to_shape)
}

fn is_face_in_face(inner: &[Point], outer: &[Point]) -> bool {
    inner.iter().all(|vert| is_point_in_face(*vert, outer))
}

fn is_point_in_face(point: Point, face: &[Point]) -> bool {
    (0..face.len())
        .map(|i| {
            Matrix2::from_columns(&[face[i], point]).determinant()
                + Matrix2::from_columns(&[point, face[(i + 1) % face.len()]]).determinant()
                + Matrix2::from_columns(&[face[(i + 1) % face.len()], face[i]]).determinant()
        })
        .all(|det_res| det_res < 0.0)
}

pub fn footprint_to_building(footprint: &[Point], height: f32) -> Vec<Vec<Vertex>> {
    let (base, roof): (Vec<Vertex>, Vec<Vertex>) = footprint
        .iter()
        .map(|point| {
            (
                Vertex {pos: [point.x, 0.0, point.y], normal: [0.0, -1.0, 0.0], col: [0.0, 1.0, 0.0, 1.0] },
                Vertex {pos: [point.x, height, point.y], normal: [0.0, 1.0, 0.0], col: [0.0, 1.0, 0.0, 1.0] },
            )
        })
        .unzip();

    let lateral_faces = (0..footprint.len()).map(|i| {
        // let normal = Vector3::new();
        let next_idx = (i + 1) % footprint.len();
        let coords = [
            Vector3::new(footprint[i].x, 0.0, footprint[i].y),
            Vector3::new(footprint[next_idx].x, 0.0, footprint[next_idx].y),
            Vector3::new(footprint[next_idx].x, height, footprint[next_idx].y),
            Vector3::new(footprint[i].x, height, footprint[i].y),
        ];
        let normal = Vector3::y().cross(&(coords[1] - coords[0])).normalize();
        coords.into_iter().map(|pos| Vertex {
            pos: pos.into(),
            normal: normal.into(),
            col: [0.0, 1.0, 0.0, 1.0],
        }).collect::<Vec<_>>()
    });

    lateral_faces.chain(vec![base, roof]).collect()
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::{
        building_generation::{cut_shapes, split_shapes_to_disjoint_faces},
        tensor_field::Point,
    };

    #[test]
    fn simple_overlapping_shapes_are_split() {
        let rect = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let triangle = vec![
            Point::new(1.0, -0.1),
            Point::new(4.0, -0.1),
            Point::new(3.0, 1.0),
        ];

        let (_, split_faces, _) = split_shapes_to_disjoint_faces(vec![rect, triangle]);

        assert_eq!(split_faces.len(), 3);

        let real_lengths: HashSet<usize> = split_faces.iter().map(|face| face.len()).collect();
        let expected_lengths = HashSet::from_iter(vec![5, 3, 6]);

        assert_eq!(real_lengths.difference(&expected_lengths).count(), 0);
    }

    /* #[test]
    fn colinear_overlapping_shapes_are_split() {
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
    } */

    #[test]
    fn simple_shape_cutting() {
        let rect = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let triangle = vec![
            Point::new(1.0, -0.1),
            Point::new(4.0, -0.1),
            Point::new(3.0, 1.0),
        ];

        let cut_shapes = cut_shapes(vec![rect, triangle], 0);

        assert_eq!(cut_shapes.len(), 2);
        assert_eq!(cut_shapes[0].len(), 4);
        // println!("{:?}", cut_shapes.iter().map(|face| face.iter().map(|vert| (vert.x, vert.y)).collect::<Vec<_>>()).collect::<Vec<_>>());
        assert_eq!(cut_shapes[1].len(), 6);
    }

    #[test]
    fn complex_shape_cutting() {
        let rect = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let triangle = vec![
            Point::new(1.0, -0.1),
            Point::new(4.0, -0.1),
            Point::new(3.0, 1.0),
        ];
        let circle: Vec<Point> = (0..10)
            .map(|i| {
                Point::new(
                    (std::f32::consts::PI / 5.0 * i as f32).cos() + 1.5,
                    (std::f32::consts::PI / 5.0 * i as f32).sin() + 1.0,
                )
            })
            .collect();

        let cut_shapes = cut_shapes(vec![rect, triangle, circle], 0);
        println!(
            "{:?}",
            cut_shapes
                .iter()
                .map(|face| face.iter().map(|vert| (vert.x, vert.y)).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );

        assert_eq!(cut_shapes.len(), 3);
        assert_eq!(cut_shapes[0].len(), 4);
        // assert_eq!(cut_shapes[1].len(), 6);
    }

    #[test]
    fn distant_shapes_are_not_cut() {
        let rect_1 = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];

        let rect_2 = vec![
            Point::new(5.0, 5.0),
            Point::new(6.0, 5.0),
            Point::new(6.0, 6.0),
            Point::new(5.0, 6.0),
        ];

        let cut_shapes = cut_shapes(vec![rect_1, rect_2], 0);

        assert_eq!(cut_shapes.len(), 2);
        assert_eq!(cut_shapes[0].len(), 4);
        assert_eq!(cut_shapes[1].len(), 4);
    }

    #[test]
    fn inset_face_is_removed() {
        let rect_1 = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(0.0, 1.0),
        ];

        let rect_2 = vec![
            Point::new(1.0, 0.25),
            Point::new(1.5, 0.25),
            Point::new(1.5, 0.5),
            Point::new(1.0, 0.5),
        ];

        let cut_shapes = cut_shapes(vec![rect_1, rect_2], 0);

        assert_eq!(cut_shapes.len(), 1);
        assert_eq!(cut_shapes[0].len(), 4);
    }
}
