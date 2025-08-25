use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use cool_utils::data_structures::dcel::DCEL;
use nalgebra::Vector2;
use ordered_float::OrderedFloat;

use crate::event_queue::EventQueue;
use crate::status::{SkipList, get_x_val_of_segment_at_height};
use crate::street_plan::HermiteCurve;
use crate::tensor_field::Point;

#[derive(Debug, PartialOrd, Clone)]
pub struct IntersectionPoint {
    pub position: Point,
    pub intersecting_segment_indices: Vec<usize>,
}

impl IntersectionPoint {
    pub fn position(&self) -> Point {
        self.position
    }

    pub fn replace_intersecting_index(&mut self, target: usize, replacement: usize) {
        for segment_idx in &mut self.intersecting_segment_indices {
            if *segment_idx == target {
                *segment_idx = replacement;
                return;
            }
        }
    }
}

impl Hash for IntersectionPoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        [OrderedFloat(self.position.x), OrderedFloat(self.position.y)].hash(state);
    }
}

impl PartialEq for IntersectionPoint {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
    }
}

impl Eq for IntersectionPoint {}

#[derive(Debug, Clone)]
pub struct EventPoint {
    pub position: Point,
    pub segment_indices: HashSet<usize>,
    pub event_type: EventPointType,
}

impl EventPoint {
    pub fn new(
        position: Point,
        segment_indices: HashSet<usize>,
        event_type: EventPointType,
    ) -> Self {
        Self {
            position,
            segment_indices,
            event_type,
        }
    }

    pub fn position(&self) -> Point {
        self.position
    }

    pub fn segment_indices(&self) -> &HashSet<usize> {
        &self.segment_indices
    }

    pub fn event_type(&self) -> EventPointType {
        self.event_type
    }

    pub fn add_segments(&mut self, segment_indices: &HashSet<usize>) {
        self.segment_indices = self
            .segment_indices
            .union(segment_indices)
            .copied()
            .collect();
    }
}

#[derive(PartialEq, PartialOrd, Eq, Debug, Clone, Copy)]
pub enum EventPointType {
    StartPoint,
    Intersection,
    EndPoint,
}

impl Eq for EventPoint {}

impl PartialEq for EventPoint {
    fn eq(&self, other: &Self) -> bool {
        self.event_type == other.event_type
            && (other.position - self.position).norm_squared() < 0.001
    }
}

impl PartialOrd for EventPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if points_are_close(self.position(), other.position()) {
            match self.event_type() {
                EventPointType::StartPoint => {
                    if other.event_type() == EventPointType::StartPoint {
                        Some(Ordering::Equal)
                    } else {
                        Some(Ordering::Greater)
                    }
                }
                EventPointType::Intersection => {
                    if other.event_type() == EventPointType::EndPoint {
                        Some(Ordering::Greater)
                    } else {
                        Some(Ordering::Less)
                    }
                }
                EventPointType::EndPoint => Some(Ordering::Less),
            }
        } else {
            match self.position.y.partial_cmp(&other.position.y) {
                Some(y_comp) => {
                    if y_comp == Ordering::Equal {
                        std::cmp::Reverse(self.position.x)
                            .partial_cmp(&std::cmp::Reverse(other.position.x))
                    } else {
                        Some(y_comp)
                    }
                }
                None => None,
            }
        }
    }
}

impl Ord for EventPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Greater)
    }
}

pub type Segment = [Point; 2];

fn point_cmp(p_0: Point, p_1: Point) -> Ordering {
    match p_0.y.total_cmp(&p_1.y) {
        Ordering::Less => Ordering::Less,
        Ordering::Equal => {
            if p_0.x < p_1.x {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        Ordering::Greater => Ordering::Greater,
    }
}

pub fn segment_start(segment: Segment) -> Point {
    if point_cmp(segment[0], segment[1]) == Ordering::Greater {
        segment[0]
    } else {
        segment[1]
    }
}

pub fn segment_end(segment: Segment) -> Point {
    if point_cmp(segment[0], segment[1]) == Ordering::Less {
        segment[0]
    } else {
        segment[1]
    }
}

pub fn new_point_lower_than_event(event: Point, new_point: Point) -> bool {
    match event.y.total_cmp(&new_point.y) {
        Ordering::Less => false,
        Ordering::Equal => event.x <= new_point.x + 0.001,
        Ordering::Greater => true,
    }
}

pub fn find_interesctions(
    segments: &[Segment],
    start_and_end_points_are_intersections: bool,
) -> Vec<IntersectionPoint> {
    let mut event_queue = EventQueue::from_segments(segments);
    let mut status = SkipList::new();

    let mut intersections = Vec::new();

    while let Some(event) = event_queue.pop() {
        let (new_events, possible_intersection) = update_status(
            &mut status,
            event.clone(),
            segments,
            intersections.last_mut(),
            start_and_end_points_are_intersections,
        );
        intersections.extend(possible_intersection);
        for new_event in new_events {
            event_queue.push(new_event);
        }
        if let Some(peek) = event_queue.peek() {
            assert!(event.position().y >= peek.position().y);
        }

        let sorted = status.is_sorted(event.position().y, segments);
        assert!(sorted);
    }

    intersections
}

fn update_status(
    status: &mut SkipList,
    event: EventPoint,
    segments: &[Segment],
    last_intersection: Option<&mut IntersectionPoint>,
    start_and_end_points_are_intersections: bool,
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    match event.event_type {
        EventPointType::StartPoint => update_status_with_start_point(
            event,
            status,
            segments,
            start_and_end_points_are_intersections,
        ),
        EventPointType::Intersection => {
            update_status_with_intersection_point(event, status, segments)
        }
        EventPointType::EndPoint => update_status_with_end_point(
            event,
            status,
            segments,
            last_intersection,
            start_and_end_points_are_intersections,
        ),
    }
}

fn sort_joint_segments(
    segment_indices: &[usize],
    segments: &[Segment],
    check_above: bool,
) -> (Vec<usize>, f32) {
    let get_closest_height = |indices: &[usize]| {
        let first_segment = segments[indices[0]];
        indices
            .iter()
            .map(|segment_index| {
                let [p_0, p_1] = segments[*segment_index];
                if check_above {
                    p_0.y.max(p_1.y)
                } else {
                    p_0.y.min(p_1.y)
                }
            })
            .fold(
                if check_above {
                    first_segment[0].y.max(first_segment[1].y)
                } else {
                    first_segment[0].y.min(first_segment[1].y)
                },
                |acc, e| if check_above { acc.min(e) } else { acc.max(e) },
            )
    };

    if let Some(horizontal_segment_index) = segment_indices
        .iter()
        .find(|&&segment_index| segments[segment_index][0].y == segments[segment_index][1].y)
    {
        let mut filtered_indices: Vec<usize> = segment_indices
            .iter()
            .filter(|&index| index != horizontal_segment_index)
            .copied()
            .collect();
        if filtered_indices.is_empty() {
            (
                vec![*horizontal_segment_index],
                segments[*horizontal_segment_index][0].y,
            )
        } else {
            let new_lowest_segment_start = get_closest_height(&filtered_indices);

            filtered_indices.sort_by(|a, b| {
                let a_x = get_x_val_of_segment_at_height(segments[*a], new_lowest_segment_start);
                let b_x = get_x_val_of_segment_at_height(segments[*b], new_lowest_segment_start);
                a_x.total_cmp(&b_x)
            });

            if check_above {
                (
                    std::iter::once(*horizontal_segment_index)
                        .chain(filtered_indices)
                        .collect(),
                    new_lowest_segment_start,
                )
            } else {
                (
                    filtered_indices
                        .into_iter()
                        .chain(std::iter::once(*horizontal_segment_index))
                        .collect(),
                    new_lowest_segment_start,
                )
            }
        }
    } else {
        let mut sorted_indices = segment_indices.to_vec();
        let lowest_segment_start = get_closest_height(&segment_indices);
        sorted_indices.sort_by(|a, b| {
            get_x_val_of_segment_at_height(segments[*a], lowest_segment_start).total_cmp(
                &get_x_val_of_segment_at_height(segments[*b], lowest_segment_start),
            )
        });

        (sorted_indices, lowest_segment_start)
    }
}

fn update_status_with_start_point(
    event: EventPoint,
    status: &mut SkipList,
    segments: &[Segment],
    start_points_are_intersections: bool,
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    let original_status_len = status.len();

    let event_segment_indices_vec: Vec<usize> = event.segment_indices.iter().copied().collect();

    let (sorted_event_segment_indices, _) =
        sort_joint_segments(&event_segment_indices_vec, segments, false);

    let insert_results: Vec<(Option<usize>, Option<usize>)> = sorted_event_segment_indices
        .iter()
        .map(|&segment_index| status.insert(segment_index, segments, event.position().y))
        .collect();

    let potential_left_neighbor = insert_results[0].0;
    let potential_right_neighbor = insert_results.last().unwrap().1;

    let mut start_intersection_segments = if sorted_event_segment_indices.len() > 1 {
        HashSet::from_iter(sorted_event_segment_indices.clone())
    } else {
        HashSet::new()
    };

    let mut later_intersection_events = Vec::new();

    let mut get_neighbor_intersections = |segment: usize, neighbor: usize| {
        let unbounded_intersection =
            calc_intersection_point_unbounded(segments[neighbor], segments[segment]);
        if points_are_close(unbounded_intersection, segments[neighbor][0])
            || points_are_close(unbounded_intersection, segments[neighbor][1])
        {
            return;
        }
        if points_are_close(unbounded_intersection, event.position()) {
            start_intersection_segments.extend(vec![neighbor, segment]);
            return;
        }
        let endpoint = segment_end(segments[segment]);
        if points_are_close(unbounded_intersection, endpoint) {
            return;
        }
        if let Some(intersection) = calc_intersection_point(segments[neighbor], segments[segment])
            && new_point_lower_than_event(event.position(), intersection)
        {
            later_intersection_events.push(EventPoint {
                position: intersection,
                segment_indices: HashSet::from_iter(vec![neighbor, segment]),
                event_type: EventPointType::Intersection,
            });
        }
    };

    for &segment in &sorted_event_segment_indices {
        if let Some(left_neighbor) = potential_left_neighbor {
            get_neighbor_intersections(segment, left_neighbor);
        }
        if let Some(right_neighbor) = potential_right_neighbor {
            get_neighbor_intersections(segment, right_neighbor);
        }
    }

    if start_points_are_intersections {
        start_intersection_segments.extend(sorted_event_segment_indices);
    }

    assert_eq!(
        status.len(),
        original_status_len + event_segment_indices_vec.len()
    );

    (
        later_intersection_events,
        if start_points_are_intersections || !start_intersection_segments.is_empty() {
            Some(IntersectionPoint {
                position: event.position(),
                intersecting_segment_indices: start_intersection_segments.into_iter().collect(),
            })
        } else {
            None
        },
    )
}

fn update_status_with_intersection_point(
    event: EventPoint,
    status: &mut SkipList,
    segments: &[Segment],
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    let original_status_len = status.len();

    let event_segment_indices_vec: Vec<usize> = event.segment_indices().iter().copied().collect();
    let (mut sorted_segment_indices, _) =
        sort_joint_segments(&event_segment_indices_vec, segments, true);

    let (potential_left_neighbor, potential_right_neighbor) = status.reverse(
        sorted_segment_indices[0],
        *sorted_segment_indices.last().unwrap(),
        segments,
        event.position().y,
    );

    sorted_segment_indices.reverse();

    let mut new_events = Vec::new();
    if let Some(left) = potential_left_neighbor
        && let Some(intersection_point) =
            calc_intersection_point(segments[left], segments[sorted_segment_indices[0]])
        && !points_are_close(
            intersection_point,
            segment_end(segments[sorted_segment_indices[0]]),
        )
        && new_point_lower_than_event(event.position(), intersection_point)
    {
        new_events.push(EventPoint {
            position: intersection_point,
            segment_indices: HashSet::from_iter(vec![left, sorted_segment_indices[0]]),
            event_type: EventPointType::Intersection,
        });
    }

    if let Some(right) = potential_right_neighbor
        && let Some(intersection_point) = calc_intersection_point(
            segments[right],
            segments[*sorted_segment_indices.last().unwrap()],
        )
        && !points_are_close(
            intersection_point,
            segment_end(segments[*sorted_segment_indices.last().unwrap()]),
        )
        && new_point_lower_than_event(event.position(), intersection_point)
    {
        new_events.push(EventPoint {
            position: intersection_point,
            segment_indices: HashSet::from_iter(vec![
                right,
                *sorted_segment_indices.last().unwrap(),
            ]),
            event_type: EventPointType::Intersection,
        });
    }

    assert_eq!(status.len(), original_status_len);

    (
        new_events,
        Some(IntersectionPoint {
            position: event.position(),
            intersecting_segment_indices: event_segment_indices_vec,
        }),
    )
}

fn update_status_with_end_point(
    event: EventPoint,
    status: &mut SkipList,
    segments: &[Segment],
    potential_last_intersection: Option<&mut IntersectionPoint>,
    mut end_points_are_intersections: bool,
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    let original_status_len = status.len();

    let event_segment_indices_vec: Vec<usize> = event.segment_indices().iter().copied().collect();
    let (sorted_segment_indices, _) =
        sort_joint_segments(&event_segment_indices_vec, segments, true);

    let deletion_results: Vec<(Option<usize>, Option<usize>)> = sorted_segment_indices
        .iter()
        .map(|index| status.remove(*index, segments, event.position().y))
        .collect();
    let potential_left_neighbor = deletion_results[0].0;
    let potential_right_neighbor = deletion_results.last().unwrap().1;

    let potential_lower_intersection_event = if let Some(left) = potential_left_neighbor
        && let Some(right) = potential_right_neighbor
        && let Some(position) = calc_intersection_point(segments[left], segments[right])
        && !points_are_close(position, segment_end(segments[left]))
        && !points_are_close(position, segment_end(segments[right]))
        && new_point_lower_than_event(event.position(), position)
    {
        vec![EventPoint {
            position,
            segment_indices: HashSet::from_iter(vec![left, right]),
            event_type: EventPointType::Intersection,
        }]
    } else {
        Vec::new()
    };

    let mut end_intersection_segments = if sorted_segment_indices.len() > 1 {
        HashSet::from_iter(sorted_segment_indices.clone())
    } else {
        HashSet::new()
    };

    if let Some(left_neighbor) = potential_left_neighbor
        && points_are_close(
            calc_intersection_point_unbounded(
                segments[left_neighbor],
                segments[sorted_segment_indices[0]],
            ),
            event.position(),
        )
    {
        end_intersection_segments.insert(left_neighbor);
        end_intersection_segments.extend(sorted_segment_indices.clone());
    }

    if let Some(right_neighbor) = potential_right_neighbor
        && points_are_close(
            calc_intersection_point_unbounded(
                segments[right_neighbor],
                segments[sorted_segment_indices[0]],
            ),
            event.position(),
        )
    {
        end_intersection_segments.insert(right_neighbor);
        end_intersection_segments.extend(sorted_segment_indices.clone());
    }

    if let Some(last_intersection) = potential_last_intersection
        && !end_intersection_segments.is_empty()
        && points_are_close(last_intersection.position(), event.position())
    {
        for index in end_intersection_segments.drain() {
            if !last_intersection
                .intersecting_segment_indices
                .contains(&index)
            {
                last_intersection.intersecting_segment_indices.push(index);
            }
        }
        end_points_are_intersections = false;
    } else if end_points_are_intersections {
        end_intersection_segments.extend(sorted_segment_indices);
    }

    assert_eq!(
        status.len(),
        original_status_len - event_segment_indices_vec.len()
    );
    (
        potential_lower_intersection_event,
        if end_points_are_intersections || !end_intersection_segments.is_empty() {
            Some(IntersectionPoint {
                position: event.position(),
                intersecting_segment_indices: end_intersection_segments.into_iter().collect(),
            })
        } else {
            None
        },
    )
}

fn calc_intersection_point(segment_0: Segment, segment_1: Segment) -> Option<Point> {
    let cross = |p_0: Point, p_1: Point| p_0.x * p_1.y - p_0.y * p_1.x;

    let p = segment_0[0];
    let q = segment_1[0];
    let r = segment_0[1];
    let s = segment_1[1];

    let denominator = cross(s - q, r - p);

    let u = cross(p - q, s - q) / denominator;
    let t = cross(p - q, r - p) / denominator;
    if (u >= 0.0 && u <= 1.0) && (t >= 0.0 && t <= 1.0) {
        Some(p + u * (r - p))
    } else {
        None
    }
}

fn calc_intersection_point_unbounded(segment_0: Segment, segment_1: Segment) -> Point {
    let cross = |p_0: Point, p_1: Point| p_0.x * p_1.y - p_0.y * p_1.x;

    let p = segment_0[0];
    let q = segment_1[0];
    let r = segment_0[1];
    let s = segment_1[1];

    let denominator = cross(s - q, r - p);
    if denominator == 0.0 {
        if segment_0[1] == segment_1[0] {
            segment_0[1]
        } else {
            segment_0[0]
        }
    } else {
        let u = cross(p - q, s - q) / denominator;
        p + u * (r - p)
    }
}

pub fn points_are_close(p_1: Point, p_2: Point) -> bool {
    if p_1 == p_2 {
        return true;
    }
    (p_1 - p_2).norm_squared() < 0.0001
}

pub fn split_segments_at_intersections(
    all_intersections: &mut [IntersectionPoint],
    segments: &mut [Segment],
) -> Vec<Segment> {
    let which_intersections_are_on_which_segments =
        list_segments_and_the_points_that_intersect_them(&all_intersections, segments);

    let mut segment_indices_sorted_by_start_height: Vec<usize> = (0..segments.len()).collect();
    segment_indices_sorted_by_start_height.sort_by(|a, b| {
        let a_segment = segments[*a];
        let a_start_height = segment_start(a_segment).y;
        let b_segment = segments[*b];
        let b_start_height = segment_start(b_segment).y;

        a_start_height.total_cmp(&b_start_height).reverse()
    });

    let mut new_segments = Vec::new();
    let mut new_segment_index = segments.len();

    for (segment_index, (segment_start_point_index, intersection_indices_on_segment)) in
        segment_indices_sorted_by_start_height
            .into_iter()
            .flat_map(|segment_index| {
                let (segment_start_point_index, points) =
                    &which_intersections_are_on_which_segments[segment_index];
                if points.len() > 1 {
                    Some((segment_index, (segment_start_point_index, points)))
                } else {
                    None
                }
            })
    {
        let replacement = intersection_indices_on_segment.iter().fold(
            Point::new(f32::MIN, f32::MIN),
            |acc, &index| {
                let current_point = all_intersections[index].position();
                match acc.y.total_cmp(&current_point.y) {
                    Ordering::Less => current_point,
                    Ordering::Equal => {
                        if acc.x < current_point.x {
                            acc
                        } else {
                            current_point
                        }
                    }
                    Ordering::Greater => acc,
                }
            },
        );
        if segments[segment_index][0] == segment_start(segments[segment_index]) {
            segments[segment_index][1] = replacement;
        } else {
            segments[segment_index][0] = replacement;
        }
        all_intersections[*segment_start_point_index].intersecting_segment_indices =
            all_intersections[*segment_start_point_index]
                .intersecting_segment_indices
                .iter()
                .filter(|intersecting_segment_index| {
                    let start_pos = all_intersections[*segment_start_point_index].position();
                    let (segment_start, segment_end) =
                        if **intersecting_segment_index >= segments.len() {
                            (
                                segment_start(
                                    new_segments[**intersecting_segment_index - segments.len()],
                                ),
                                segment_end(
                                    new_segments[**intersecting_segment_index - segments.len()],
                                ),
                            )
                        } else {
                            (
                                segment_start(segments[**intersecting_segment_index]),
                                segment_end(segments[**intersecting_segment_index]),
                            )
                        };

                    points_are_close(start_pos, segment_start)
                        || points_are_close(start_pos, segment_end)
                })
                .copied()
                .collect();

        let mut sorted_intersection_indices_on_segment = intersection_indices_on_segment.clone();
        sorted_intersection_indices_on_segment.sort_by(|a, b| {
            point_cmp(
                all_intersections[*a].position(),
                all_intersections[*b].position(),
            )
            .reverse()
        });

        for (i, &intersection_index) in sorted_intersection_indices_on_segment[1..]
            .iter()
            .enumerate()
        {
            // Because we are starting at the second position (index 1), `enumerate()` makes it so
            // i is always 1 index behind the current intersection
            let previous_intersection_index = sorted_intersection_indices_on_segment[i];
            let previous_position = all_intersections[previous_intersection_index].position();
            let current_position = all_intersections[intersection_index].position();
            new_segments.push([previous_position, current_position]);
            all_intersections[previous_intersection_index]
                .intersecting_segment_indices
                .push(new_segment_index);

            all_intersections[intersection_index]
                .replace_intersecting_index(segment_index, new_segment_index);
            new_segment_index += 1;
        }
    }

    new_segments
}

fn list_segments_and_the_points_that_intersect_them(
    intersection_points: &[IntersectionPoint],
    segments: &[Segment],
) -> Vec<(usize, Vec<usize>)> {
    let mut points_on_each_segment = vec![(usize::MAX, Vec::new()); segments.len()];

    for (
        intersection_index,
        IntersectionPoint {
            intersecting_segment_indices,
            position,
        },
    ) in intersection_points.iter().enumerate()
    {
        for &intersecting_segment in intersecting_segment_indices {
            if !points_are_close(*position, segment_start(segments[intersecting_segment])) {
                points_on_each_segment[intersecting_segment]
                    .1
                    .push(intersection_index);
            } else {
                points_on_each_segment[intersecting_segment].0 = intersection_index;
            }
        }
    }

    points_on_each_segment
}

pub fn path_to_graph(paths: &[HermiteCurve]) -> Vec<Vec<Point>> {
    let all_segment_points = paths.iter().map(|curve| {
        curve
            .into_iter()
            .map(|pos| pos.position)
            .collect::<Vec<_>>()
    });

    let mut segments: Vec<[Point; 2]> = all_segment_points
        .flat_map(|curve| {
            curve[..curve.len() - 1]
                .into_iter()
                .enumerate()
                .map(|(i, point)| [*point, curve[i + 1]])
                .collect::<Vec<_>>()
        })
        .collect();

    let (vertices, adjacency_list) = segments_to_adjacency_list(&mut segments);

    let dcel = DCEL::new(&vertices, adjacency_list);

    let new_faces = dcel
        .faces()
        .iter()
        .flat_map(|face_from_indices| {
            correct_face_with_degenerate_points(
                face_from_indices
                    .iter()
                    .map(|index| vertices[*index])
                    .collect(),
            )
        })
        .collect();

    new_faces
}

type AdjacencyList = HashMap<usize, HashSet<usize>>;

fn vertices_to_adjacency_list(
    vertices: Vec<IntersectionPoint>,
    segments: &[Segment],
) -> (Vec<Point>, AdjacencyList) {
    let inverse_vertices: HashMap<IntersectionPoint, usize> =
        HashMap::from_iter(vertices.iter().enumerate().map(
            |(
                i,
                IntersectionPoint {
                    position,
                    intersecting_segment_indices,
                },
            )| {
                (
                    IntersectionPoint {
                        position: truncate_point_to_decimal_place(*position, 3),
                        intersecting_segment_indices: intersecting_segment_indices.clone(),
                    },
                    i,
                )
            },
        ));

    assert_eq!(inverse_vertices.len(), vertices.len());

    let adjacency_list = vertices
        .iter()
        .enumerate()
        .map(|(i, intersection)| {
            let all_connected_vertex_indices = intersection
                .intersecting_segment_indices
                .iter()
                .flat_map(|&segment_index| segments[segment_index]);
            let all_conneced_vertices = all_connected_vertex_indices
                .map(|vertex| {
                    *inverse_vertices
                        .get(&IntersectionPoint {
                            position: truncate_point_to_decimal_place(vertex, 3),
                            intersecting_segment_indices: Vec::new(),
                        })
                        .expect(&format!("No inverse for {vertex:?}"))
                })
                .filter(|&vertex_index| vertex_index != i)
                .collect();
            (i, all_conneced_vertices)
        })
        .collect();

    (
        vertices
            .into_iter()
            .map(|IntersectionPoint { position, .. }| position)
            .collect(),
        adjacency_list,
    )
}

fn truncate_point_to_decimal_place(point: Point, decimal_places: u32) -> Point {
    let fac = 10_i32.pow(decimal_places) as f32;
    let point_x = (point.x * fac).trunc();
    let point_y = (point.y * fac).trunc();

    Point::new(point_x / fac, point_y / fac)
}

fn segments_to_adjacency_list(segments: &mut [Segment]) -> (Vec<Point>, AdjacencyList) {
    let intersections: HashSet<IntersectionPoint> =
        HashSet::from_iter(find_interesctions(&segments, true));

    let mut intersections_vec: Vec<IntersectionPoint> = intersections.iter().cloned().collect();

    let new_segments = split_segments_at_intersections(&mut intersections_vec, segments);

    let all_segments: Vec<Segment> = segments.to_vec().into_iter().chain(new_segments).collect();

    vertices_to_adjacency_list(intersections_vec, &all_segments)
}

fn correct_face_with_degenerate_points(face: Vec<Point>) -> Vec<Vec<Point>> {
    let mut full_face = Vec::new();

    let mut vertex_pos_to_index: HashMap<(OrderedFloat<f32>, OrderedFloat<f32>), usize> =
        HashMap::new();

    let mut adjacency_list = AdjacencyList::new();

    let mut skipped_original_indices = HashSet::new();

    for (current_vertex_index, vertex) in face.iter().enumerate() {
        let point_as_key = (
            OrderedFloat(truncate_point_to_decimal_place(*vertex, 3).x),
            OrderedFloat(truncate_point_to_decimal_place(*vertex, 3).y),
        );

        let neighbors: Vec<usize> = if current_vertex_index == 0 {
            vec![face.len() - 1, 1]
        } else if current_vertex_index == face.len() - 1 {
            vec![face.len() - 2, 0]
        } else {
            vec![current_vertex_index - 1, current_vertex_index + 1]
        };

        if let Some(existing_vertex_index) = vertex_pos_to_index.get(&point_as_key) {
            adjacency_list
                .get_mut(existing_vertex_index)
                .unwrap()
                .extend(neighbors);
            skipped_original_indices.insert(current_vertex_index);
        } else {
            adjacency_list.insert(full_face.len(), HashSet::from_iter(neighbors));
            vertex_pos_to_index.insert(point_as_key, full_face.len());
            full_face.push(*vertex);
        }
    }

    // Correct indices after skipping duplicates
    for neighbors in adjacency_list.values_mut() {
        for neighbor_index in neighbors.clone() {
            let index_to_replace_with = if skipped_original_indices.contains(&neighbor_index) {
                let truncated_pos = truncate_point_to_decimal_place(face[neighbor_index], 3);
                vertex_pos_to_index[&(OrderedFloat(truncated_pos.x), OrderedFloat(truncated_pos.y))]
            } else {
                let offset = skipped_original_indices
                    .iter()
                    .map(|skipped_index| (neighbor_index > *skipped_index) as usize)
                    .sum::<usize>();

                neighbor_index - offset
            };

            neighbors.remove(&neighbor_index);
            neighbors.insert(index_to_replace_with);
        }
    }

    let mut degenerate_points_indices: Vec<usize> = (0..full_face.len())
        .filter(|i| adjacency_list[i].len() == 1)
        .collect();

    degenerate_points_indices.sort_by(|a, b| {
        let a_point = full_face[*a];
        let b_point = full_face[*b];
        a_point.y.total_cmp(&b_point.y).reverse()
    });

    for degenerate_point_index in degenerate_points_indices {
        let degenerate_point = full_face[degenerate_point_index];
        let degenerate_previous_point =
            full_face[(degenerate_point_index as i32 - 1).rem_euclid(full_face.len() as i32) as usize];

        let all_segments_by_indices: HashSet<(usize, usize)> =
            HashSet::from_iter(adjacency_list.iter().flat_map(
                |(vertex_index, vertex_neighbors)| {
                    vertex_neighbors.iter().map(|neighbor_index| {
                        (
                            *vertex_index.min(neighbor_index),
                            *vertex_index.max(neighbor_index),
                        )
                    })
                },
            ));

        let segment_indices: Vec<(usize, usize)> = all_segments_by_indices
            .iter()
            .filter(|(p_0, p_1)| {
                full_face[*p_0] != degenerate_point
                    && full_face[*p_1] != degenerate_point
                    && full_face[*p_0] != degenerate_previous_point
                    && full_face[*p_1] != degenerate_previous_point
            })
            .copied()
            .collect();

        let segments: Vec<Segment> = segment_indices
            .iter()
            .map(|&(p_0, p_1)| [full_face[p_0], full_face[p_1]])
            .collect();

        let mut raycast_points = raycast_through_segments(
            degenerate_point,
            degenerate_point - degenerate_previous_point,
            &segments,
        );

        raycast_points.sort_by(|a, b| {
            (degenerate_point - a.0)
                .norm_squared()
                .total_cmp(&(degenerate_point - b.0).norm_squared())
        });

        let mut new_segment_origin_index = degenerate_point_index;

        for (raycast_intersection, intersecting_segment_index) in raycast_points {
            let new_point_index = full_face.len();
            let (left_intersection_index, right_intersection_index) =
                segment_indices[intersecting_segment_index];

            adjacency_list.insert(
                new_point_index,
                HashSet::from_iter(vec![
                    new_segment_origin_index,
                    left_intersection_index,
                    right_intersection_index,
                ]),
            );

            let left_intersection_set = adjacency_list.get_mut(&left_intersection_index).unwrap();
            left_intersection_set.insert(new_point_index);
            left_intersection_set.remove(&right_intersection_index);
            let right_intersection_set = adjacency_list.get_mut(&right_intersection_index).unwrap();
            right_intersection_set.insert(new_point_index);
            right_intersection_set.remove(&left_intersection_index);

            full_face.push(raycast_intersection);
            adjacency_list
                .get_mut(&new_segment_origin_index)
                .unwrap()
                .insert(new_point_index);

            new_segment_origin_index = new_point_index;
        }
    }

    let dcel = DCEL::new(&full_face, adjacency_list);

    dcel.faces()
        .iter()
        .map(|current_face| {
            current_face
                .iter()
                .map(|index| full_face[*index])
                .collect::<Vec<Point>>()
        })
        .collect()
}

fn raycast_through_segments(
    origin: Point,
    direction: Vector2<f32>,
    segments: &[Segment],
) -> Vec<(Point, usize)> {
    let cross = |p_0: Point, p_1: Point| p_0.x * p_1.y - p_0.y * p_1.x;
    segments
        .iter()
        .enumerate()
        .flat_map(|(i, segment)| {
            let p = segment[0];
            let q = origin;
            let r = segment[1];
            let s = origin + direction;

            let denominator = cross(s - q, r - p);
            if denominator == 0.0 {
                None
            } else {
                let u = cross(p - q, s - q) / denominator;
                let new_pos = p + u * (r - p);
                if u >= 0.0
                    && u <= 1.0
                    && (new_pos - origin).normalize().dot(&direction.normalize()) > 0.0
                {
                    Some((new_pos, i))
                } else {
                    None
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod test {
    use cool_utils::data_structures::dcel::DCEL;
    use ordered_float::OrderedFloat;

    use crate::{
        street_graph::{
            AdjacencyList, calc_intersection_point_unbounded, path_to_graph, points_are_close,
            segment_end, segment_start, sort_joint_segments,
        },
        street_plan::ControlPoint,
    };
    use std::{
        collections::{HashMap, HashSet},
        iter,
    };

    use crate::{event_queue::EventQueue, street_graph::IntersectionPoint, tensor_field::Point};

    use super::{
        EventPoint, EventPointType, Segment, SkipList, calc_intersection_point,
        correct_face_with_degenerate_points, find_interesctions, segments_to_adjacency_list,
        split_segments_at_intersections, update_status, vertices_to_adjacency_list,
    };

    fn same_intersecting_segments(test: Vec<usize>, expected: Vec<usize>) -> bool {
        test.len() == expected.len()
            && HashSet::<usize>::from_iter(test)
                .difference(&HashSet::from_iter(expected))
                .count()
                == 0
    }

    #[test]
    fn sorting_simple_joint_start_segments() {
        let segments = [
            [Point::new(1.0, 1.0), Point::new(4.0, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(1.0, 1.0), Point::new(0.0, 0.0)],
            [Point::new(1.0, 1.0), Point::new(5.0, 0.0)],
        ];

        assert_eq!(
            sort_joint_segments(&[0, 1, 2, 3], &segments, false),
            (vec![2, 1, 0, 3], 0.0)
        );
    }

    #[test]
    fn sorting_complex_joint_start_segments() {
        let segments = [
            [Point::new(1.0, 1.0), Point::new(4.0, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(1.0, 1.0), Point::new(0.0, 0.0)],
            [Point::new(1.0, 1.0), Point::new(5.0, 0.0)],
            [Point::new(1.0, 1.0), Point::new(4.0, 1.0)],
        ];

        assert_eq!(
            sort_joint_segments(&[0, 1, 2, 3, 4], &segments, false),
            (vec![2, 1, 0, 3, 4], 0.0)
        );
    }

    #[test]
    fn sorting_simple_joint_intersection_segments() {
        let segments = [
            [Point::new(2.0, 2.0), Point::new(0.0, 0.0)],
            [Point::new(1.5, 0.5), Point::new(0.0, 2.0)],
            [Point::new(1.0, 2.0), Point::new(1.0, 0.0)],
        ];

        assert_eq!(
            sort_joint_segments(&[0, 1, 2], &segments, false),
            (vec![0, 2, 1], 0.5)
        );
    }
    #[test]
    fn sorting_complex_joint_intersection_segments() {
        let segments = [
            [Point::new(2.0, 2.0), Point::new(0.0, 0.0)],
            [Point::new(2.0, 0.0), Point::new(0.0, 2.0)],
            [Point::new(0.0, 1.0), Point::new(2.0, 1.0)],
            [Point::new(1.0, 2.0), Point::new(1.0, 0.0)],
        ];

        assert_eq!(
            sort_joint_segments(&[0, 1, 2, 3], &segments, true),
            (vec![2, 1, 3, 0], 2.0)
        );
    }

    #[test]
    fn segments_to_event_queue_correctly_prioritizes_single_segment() {
        let segments = [[Point::new(1.0, 1.0), Point::new(2.0, -4.0)]];

        let event_queue = EventQueue::from_segments(&segments);

        let event_queue_vec = event_queue.into_ordered_vec();

        let expected_event_queue_vec = vec![
            EventPoint {
                position: Point::new(1.0, 1.0),
                segment_indices: HashSet::from_iter(std::iter::once(0)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, -4.0),
                segment_indices: HashSet::from_iter(std::iter::once(0)),
                event_type: EventPointType::EndPoint,
            },
        ];

        assert_eq!(event_queue_vec, expected_event_queue_vec);
    }

    #[test]
    fn segments_to_event_queue_correctly_prioritizes_segments() {
        let segments = [
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
            [Point::new(-10.0, -5.0), Point::new(2.0, 5.0)],
        ];

        let event_queue = EventQueue::from_segments(&segments);
        let event_queue_as_vec = event_queue.into_ordered_vec();

        let proper_events = vec![
            EventPoint {
                position: Point::new(-5.0, 12.0),
                segment_indices: HashSet::from_iter(iter::once(1)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, 5.0),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, 2.0),
                segment_indices: HashSet::from_iter(iter::once(2)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(1.0, 1.0),
                segment_indices: HashSet::from_iter(iter::once(0)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(13.0, -0.2),
                segment_indices: HashSet::from_iter(iter::once(1)),
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(2.0, -4.0),
                segment_indices: HashSet::from_iter(iter::once(0)),
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(-10.0, -5.0),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(0.0, -10.0),
                segment_indices: HashSet::from_iter(iter::once(2)),
                event_type: EventPointType::EndPoint,
            },
        ];

        assert_eq!(event_queue_as_vec, proper_events);
    }

    #[test]
    fn intersection_calc_correctly_determines_intersection_point() {
        let segment_0 = [Point::new(1.0, 1.0), Point::new(2.0, -4.0)];
        let segment_1 = [Point::new(4.0, 2.0), Point::new(1.5, -3.0)];

        let intersection_res = calc_intersection_point(segment_0, segment_1);

        assert!((intersection_res.unwrap() - Point::new(1.71428571, -2.57142857)).norm() < 0.00001);
    }

    #[test]
    fn intersection_calc_correctly_detects_when_no_intersection() {
        let segment_0 = [Point::new(2.0, 2.0), Point::new(0.0, -10.0)];
        let segment_1 = [Point::new(4.0, 2.0), Point::new(1.5, -3.0)];

        let intersection_res = calc_intersection_point(segment_0, segment_1);

        assert!(intersection_res.is_none());
    }

    #[test]
    fn intersection_calc_correctly_ignores_parallel_lines() {
        let segment_0 = [Point::new(2.0, 2.0), Point::new(0.0, -10.0)];
        let segment_1 = [Point::new(2.0, 0.0), Point::new(0.0, -12.0)];

        let intersection_res = calc_intersection_point(segment_0, segment_1);

        assert!(intersection_res.is_none());
    }

    fn load_segment_start_event_to_event_queue(segments: &[Segment]) -> EventQueue {
        let mut event_queue = EventQueue::new();
        for (i, segment) in segments.iter().enumerate() {
            event_queue.push(EventPoint {
                position: if segment[0].y > segment[1].y {
                    segment[0]
                } else if segment[0].y == segment[1].y {
                    if segment[0].x < segment[1].x {
                        segment[0]
                    } else {
                        segment[1]
                    }
                } else {
                    segment[1]
                },
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            });
        }

        event_queue
    }

    #[test]
    fn start_points_are_correctly_inserted_into_status() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert!(new_events.is_empty());
        assert!(new_intersections.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec = vec![1, 2, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    }

    #[test]
    fn start_points_correctly_calculate_single_future_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(-0.7, 1.5), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_new_events = vec![EventPoint {
            position: Point::new(1.16666667, 0.16666667),
            segment_indices: HashSet::from_iter(iter::once(3)),
            event_type: EventPointType::Intersection,
        }];

        assert_eq!(new_events.len(), expected_new_events.len());
        for (i, expected_event) in expected_new_events.iter().enumerate() {
            let real_event = &new_events[i];
            assert!(points_are_close(
                real_event.position,
                expected_event.position
            ));
        }
        assert!(new_intersections.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec = vec![1, 2, 3, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    }

    #[test]
    fn start_points_correctly_calculate_double_future_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
            [Point::new(-0.7, 1.5), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_new_events = vec![
            EventPoint {
                position: Point::new(1.16666667, 0.16666667),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::Intersection,
            },
            EventPoint {
                position: Point::new(1.45454545, -1.27272727),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::Intersection,
            },
        ];

        assert_eq!(new_events.len(), expected_new_events.len());
        for (i, expected_event) in expected_new_events.iter().enumerate() {
            let real_event = &new_events[i];
            assert!(points_are_close(
                real_event.position,
                expected_event.position
            ));
        }

        assert!(new_intersections.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec = vec![1, 3, 4, 2, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    }

    #[test]
    fn start_points_correctly_identify_single_start_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(1.0, 1.0), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_intersections = vec![IntersectionPoint {
            position: Point::new(1.0, 1.0),
            intersecting_segment_indices: vec![3, 2],
        }];

        // Uncertainty in the order of elements is caused by the use of hashsets, either order is
        // valid and should not affect algorithm performance

        assert_eq!(new_intersections.len(), 1);
        assert_eq!(
            new_intersections[0].position,
            expected_intersections[0].position
        );
        assert!(same_intersecting_segments(
            new_intersections[0].intersecting_segment_indices.clone(),
            expected_intersections[0]
                .intersecting_segment_indices
                .clone()
        ));

        assert!(new_events.is_empty());

        let status_as_vec = status.to_vec();

        let expected_vals_1 = vec![1, 2, 3, 0];
        let expected_vals_2 = vec![1, 3, 2, 0];
        assert!(status_as_vec == expected_vals_1 || status_as_vec == expected_vals_2);
    }

    #[test]
    fn start_points_correctly_identify_single_intersection_in_joint_start_point() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(1.0, 1.0), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(0.4, 0.4), Point::new(1.3, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert_eq!(new_intersections.len(), 1);
        assert_eq!(new_intersections[0].position, Point::new(1.0, 1.0));
        assert!(same_intersecting_segments(
            new_intersections[0].intersecting_segment_indices.clone(),
            vec![2, 3]
        ));

        assert_eq!(new_events.len(), 1);
        assert_eq!(new_events[0].event_type(), EventPointType::Intersection);
        assert!(points_are_close(
            new_events[0].position(),
            Point::new(1.1902439, 0.04878049)
        ));
        let intersection_segments: Vec<usize> =
            new_events[0].segment_indices.iter().copied().collect();
        assert_eq!(intersection_segments.len(), 2);
        assert!(intersection_segments.contains(&3));
        assert!(intersection_segments.contains(&4));

        let status_as_vec = status.to_vec();

        let expected_vals_1 = vec![1, 4, 2, 3, 0];
        let expected_vals_2 = vec![1, 4, 3, 2, 0];
        assert!(status_as_vec == expected_vals_1 || status_as_vec == expected_vals_2);
    }

    #[test]
    fn start_points_correctly_identify_segment_start_on_other_segment() {
        let segments = vec![
            [Point::new(0.0, 5.0), Point::new(5.0, 5.0)],
            [Point::new(2.0, 5.0), Point::new(2.0, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert!(new_events.is_empty());
        assert_eq!(new_intersections.len(), 1);
        assert_eq!(new_intersections[0].position, Point::new(2.0, 5.0));
        assert!(same_intersecting_segments(
            new_intersections[0].intersecting_segment_indices.clone(),
            vec![0, 1]
        ));
    }

    #[test]
    fn start_points_correctly_identify_segment_start_on_other_segment_2() {
        let segments = vec![
            [
                Point::new(133.69496, 359.18222),
                Point::new(145.6927, 330.59787),
            ],
            [
                Point::new(138.1485, 348.57175),
                Point::new(118.75913, 338.8408),
            ],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) = update_status(
                &mut status,
                event.clone(),
                &segments,
                new_intersections.last_mut(),
                false,
            );

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }
        assert_eq!(status.to_vec(), vec![1, 0]);

        assert!(new_events.is_empty());
        assert_eq!(new_intersections.len(), 1);
        assert_eq!(
            new_intersections[0].position,
            Point::new(138.1485, 348.57175)
        );
        assert!(same_intersecting_segments(
            new_intersections[0].intersecting_segment_indices.clone(),
            vec![0, 1]
        ));
    }

    #[test]
    fn simple_intersection_point_correctly_correctly_flips_status() {
        let segments = vec![
            [Point::new(0.0, 5.0), Point::new(5.0, 5.0)],
            [Point::new(2.0, 7.0), Point::new(2.0, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 1]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert!(result_events.is_empty());

        assert!(result_intersection.is_some(),);
        assert_eq!(
            result_intersection.as_ref().unwrap().position,
            Point::new(2.0, 5.0)
        );
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![0, 1]
        ));

        assert_eq!(status.to_vec(), vec![1, 0]);
    }

    #[test]
    fn complex_intersection_point_corectly_flips_status() {
        let segments = vec![
            [Point::new(0.0, 1.0), Point::new(5.0, 1.0)],
            [Point::new(1.0, 2.0), Point::new(1.0, 0.0)],
            [Point::new(0.0, 0.0), Point::new(2.0, 2.0)],
            [Point::new(0.0, 2.0), Point::new(2.0, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 3, 1, 2]);
        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert!(result_events.is_empty());

        assert!(result_intersection.is_some(),);
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![0, 1, 2, 3]
        ));

        assert_eq!(status.to_vec(), vec![2, 1, 3, 0]);
    }

    #[test]
    fn simple_intersection_point_correctly_detects_no_new_intersections() {
        let segments = vec![
            [Point::new(1.3, 2.8), Point::new(4.5, 0.26)],
            [Point::new(-0.28, 0.37), Point::new(0.38, 1.95)],
            [Point::new(0.49, 2.43), Point::new(1.94, -0.14)],
            [Point::new(0.0, 0.0), Point::new(2.0, 2.0)],
        ];

        /*
            polygon((1.3, 2.8), (4.5, 0.26))
            polygon((-0.28, 0.37), (0.38, 1.95))
            polygon((0.49, 2.43), (1.94, -0.14))
            polygon((0.0, 0.0), (2.0, 2.0))
        */

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![1, 2, 3, 0]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert!(result_events.is_empty());

        assert!(result_intersection.is_some(),);
        assert!(points_are_close(
            result_intersection.as_ref().unwrap().position,
            Point::new(1.18975124, 1.18975124)
        ));
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![2, 3]
        ));

        assert_eq!(status.to_vec(), vec![1, 3, 2, 0]);
    }

    #[test]
    fn complex_intersection_point_correctly_detects_no_new_intersections() {
        let segments = vec![
            [Point::new(0.46, 1.4), Point::new(4.5, 0.236)],
            [Point::new(-0.28, 0.37), Point::new(0.38, 1.95)],
            [Point::new(0.49, 2.43), Point::new(1.94, -0.14)],
            [Point::new(0.0, 0.0), Point::new(2.0, 2.0)],
            [Point::new(1.19, 1.96), Point::new(1.19, 0.07)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event.clone(), &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![1, 0, 2, 4, 3]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert!(result_events.is_empty());

        assert!(result_intersection.is_some(),);
        assert!(points_are_close(
            result_intersection.as_ref().unwrap().position,
            Point::new(1.18975124, 1.18975124)
        ));
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![0, 2, 3, 4]
        ));

        assert_eq!(status.to_vec(), vec![1, 3, 4, 2, 0]);
    }

    #[test]
    fn simple_intersection_point_correctly_detects_one_new_intersection() {
        let segments = vec![
            [Point::new(-0.4, 1.6), Point::new(-0.9, 0.3)],
            [Point::new(-0.3, 0.4), Point::new(1.0, 2.0)],
            [Point::new(0.3, 1.7), Point::new(1.9, -0.1)],
            [Point::new(0.0, 0.0), Point::new(3.0, 2.4)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 2, 1, 3]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert_eq!(result_events.len(), 1);
        assert!(points_are_close(
            result_events[0].position(),
            Point::new(1.05844156, 0.84675325)
        ));
        assert_eq!(
            result_events[0]
                .segment_indices
                .difference(&HashSet::from_iter(vec![2, 3]))
                .count(),
            0
        );

        assert!(result_intersection.is_some(),);
        assert!(points_are_close(
            result_intersection.as_ref().unwrap().position,
            Point::new(0.53836735, 1.43183673)
        ));
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![1, 2]
        ));

        assert_eq!(status.to_vec(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn complex_intersection_point_correctly_detects_one_new_intersection() {
        let segments = vec![
            [Point::new(-0.4, 1.6), Point::new(0.0, 0.4)],
            [Point::new(-0.3, 0.4), Point::new(1.0, 2.0)],
            [Point::new(0.3, 1.7), Point::new(1.9, -0.1)],
            [Point::new(0.0, 0.0), Point::new(0.752, 2.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 2, 3, 1]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert_eq!(result_events.len(), 1);
        assert!(points_are_close(
            result_events[0].position(),
            Point::new(-0.08727273, 0.66181818)
        ));
        assert_eq!(
            result_events[0]
                .segment_indices
                .difference(&HashSet::from_iter(vec![0, 1]))
                .count(),
            0
        );

        assert!(result_intersection.is_some(),);
        assert!(points_are_close(
            result_intersection.as_ref().unwrap().position,
            Point::new(0.53836735, 1.43183673)
        ));
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![1, 2, 3]
        ));

        assert_eq!(status.to_vec(), vec![0, 1, 3, 2]);
    }

    #[test]
    fn simple_intersection_point_correctly_detects_two_new_intersections() {
        let segments = vec![
            [Point::new(-0.4, 1.6), Point::new(0.5, 0.7)],
            [Point::new(0.2, 0.4), Point::new(1.2, 2.0)],
            [Point::new(0.5, 0.5), Point::new(1.9, 1.5)],
            [Point::new(0.6, 0.1), Point::new(0.5, 2.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 3, 1, 2]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert_eq!(result_events.len(), 2);
        assert!(points_are_close(
            result_events[0].position(),
            Point::new(0.43076923, 0.76923077)
        ));
        assert_eq!(
            result_events[0]
                .segment_indices
                .difference(&HashSet::from_iter(vec![0, 1]))
                .count(),
            0
        );
        assert!(points_are_close(
            result_events[1].position(),
            Point::new(0.57608696, 0.55434783)
        ));
        assert_eq!(
            result_events[1]
                .segment_indices
                .difference(&HashSet::from_iter(vec![2, 3]))
                .count(),
            0
        );

        assert!(result_intersection.is_some(),);
        assert!(points_are_close(
            result_intersection.as_ref().unwrap().position,
            Point::new(0.55436893, 0.96699029)
        ));
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![1, 3]
        ));

        assert_eq!(status.to_vec(), vec![0, 1, 3, 2]);
    }

    #[test]
    fn complex_intersection_point_correctly_detects_two_new_intersections() {
        let segments = vec![
            [Point::new(-0.4, 1.6), Point::new(0.0, 0.4)],
            [Point::new(-0.3, 0.4), Point::new(1.0, 2.0)],
            [Point::new(0.3, 1.7), Point::new(1.9, -0.1)],
            [Point::new(0.0, 0.0), Point::new(0.752, 2.0)],
            [Point::new(1.4, 1.9), Point::new(1.0, 0.25)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) =
                update_status(&mut status, event, &segments, None, false);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 2, 3, 1, 4]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) = update_status(
            &mut status,
            new_events.pop().unwrap().clone(),
            &segments,
            None,
            false,
        );
        assert_eq!(result_events.len(), 2);
        assert!(points_are_close(
            result_events[0].position(),
            Point::new(-0.08727273, 0.66181818)
        ));
        assert_eq!(
            result_events[0]
                .segment_indices
                .difference(&HashSet::from_iter(vec![0, 1]))
                .count(),
            0
        );
        assert!(points_are_close(
            result_events[1].position(),
            Point::new(1.12619048, 0.77053571)
        ));
        assert_eq!(
            result_events[1]
                .segment_indices
                .difference(&HashSet::from_iter(vec![2, 4]))
                .count(),
            0
        );

        assert!(result_intersection.is_some(),);
        assert!(points_are_close(
            result_intersection.as_ref().unwrap().position,
            Point::new(0.53836735, 1.43183673)
        ));
        assert!(same_intersecting_segments(
            result_intersection.unwrap().intersecting_segment_indices,
            vec![1, 2, 3]
        ));

        assert_eq!(status.to_vec(), vec![0, 1, 3, 2, 4]);
    }

    #[test]
    fn end_segment_properly_removes_from_status() {
        let segments = vec![[Point::new(12.0, 8.0), Point::new(0.0, 4.0)]];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        while let Some(event) = event_queue.pop() {
            let _ = update_status(&mut status, event.clone(), &segments, None, false);
        }

        assert_eq!(status.len(), 0);
    }

    #[test]
    fn intersecting_end_segments_properly_removes_from_status() {
        let segments = vec![
            [Point::new(12.0, 8.0), Point::new(0.0, 4.0)],
            [Point::new(2.0, 5.0), Point::new(0.0, 4.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (_, potential_new_intersection) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_intersections.extend(potential_new_intersection);
        }

        assert_eq!(status.len(), 0);
        assert_eq!(all_new_intersections.len(), 1);
        assert_eq!(all_new_intersections[0].position, Point::new(0.0, 4.0));
        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![1, 0]
        ));
    }

    #[test]
    fn simple_end_segment_results_in_intersection() {
        let segments = vec![
            [Point::new(6.0, 6.0), Point::new(2.0, 4.0)],
            [Point::new(2.0, 5.0), Point::new(0.0, 2.0)],
            [Point::new(0.0, 3.0), Point::new(4.0, 4.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut new_event_points = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, _) = update_status(&mut status, event.clone(), &segments, None, false);
            new_event_points.extend(new_events.clone());
            for event in new_events {
                event_queue.push(event);
            }
        }

        assert_eq!(status.len(), 0);
        assert_eq!(new_event_points.len(), 1);
        assert!(points_are_close(
            new_event_points[0].position(),
            Point::new(0.8, 3.2)
        ));
        assert_eq!(
            new_event_points[0]
                .segment_indices()
                .difference(&HashSet::from_iter(vec![1, 2]))
                .count(),
            0
        );
        assert_eq!(
            new_event_points[0].event_type(),
            EventPointType::Intersection
        );
    }

    #[test]
    fn complex_end_segment_results_in_intersection() {
        let segments = vec![
            [Point::new(6.0, 6.0), Point::new(2.0, 4.0)],
            [Point::new(2.0, 5.0), Point::new(0.0, 2.0)],
            [Point::new(0.0, 3.0), Point::new(4.0, 3.5)],
            [Point::new(2.0, 4.0), Point::new(2.3, 6.5)],
            [Point::new(1.5, 4.1), Point::new(2.0, 4.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert_eq!(status.len(), 0);

        assert_eq!(all_new_intersections.len(), 2);
        assert_eq!(all_new_intersections[0].position, Point::new(2.0, 4.0));
        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![4, 3, 0]
        ));

        assert_eq!(all_new_event_points.len(), 1);
        assert!(points_are_close(
            all_new_event_points[0].position(),
            Point::new(0.72727273, 3.09090909)
        ));
        assert_eq!(
            all_new_event_points[0]
                .segment_indices()
                .difference(&HashSet::from_iter(vec![1, 2]))
                .count(),
            0
        );
        assert_eq!(
            all_new_event_points[0].event_type(),
            EventPointType::Intersection
        );
    }

    #[test]
    fn joint_simple_start_simple_end_intersection_works_properly() {
        let segments = vec![
            [Point::new(6.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(2.0, 2.0), Point::new(4.0, 0.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert!(all_new_event_points.is_empty());
        assert_eq!(all_new_intersections.len(), 1);

        assert_eq!(all_new_intersections[0].position, Point::new(2.0, 2.0));
        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![0, 1]
        ));
    }

    #[test]
    fn joint_complex_start_simple_end_intersection_works_properly() {
        let segments = vec![
            [Point::new(6.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(2.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(2.0, 2.0), Point::new(4.0, 0.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert!(all_new_event_points.is_empty());
        assert_eq!(all_new_intersections.len(), 1);

        assert_eq!(all_new_intersections[0].position, Point::new(2.0, 2.0));
        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![1, 0, 2]
        ));
    }

    #[test]
    fn joint_simple_start_complex_end_intersection_works_properly() {
        let segments = vec![
            [Point::new(6.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(2.0, 2.0), Point::new(4.0, 0.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, 0.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert!(all_new_event_points.is_empty());
        assert_eq!(all_new_intersections.len(), 1);

        assert_eq!(all_new_intersections[0].position, Point::new(2.0, 2.0));
        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![0, 1, 2]
        ));
    }

    #[test]
    fn joint_complex_start_complex_end_intersection_works_properly() {
        let segments = vec![
            [Point::new(0.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(6.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(2.0, 2.0), Point::new(4.0, 0.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, 0.0)],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert!(all_new_event_points.is_empty());
        assert_eq!(all_new_intersections.len(), 1);

        assert_eq!(all_new_intersections[0].position, Point::new(2.0, 2.0));
        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![0, 1, 2, 3]
        ));
    }

    #[test]
    fn end_point_on_other_segment_correctly_detects_intersection() {
        let segments = vec![
            [
                Point::new(477.3223, 385.51642),
                Point::new(475.40628, 404.6209),
            ],
            [
                Point::new(471.94916, 384.97754),
                Point::new(491.84967, 386.9734),
            ],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert!(all_new_event_points.is_empty());
        assert_eq!(all_new_intersections.len(), 1);

        assert!(points_are_close(
            all_new_intersections[0].position(),
            Point::new(477.3223, 385.51642)
        ));

        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![0, 1]
        ));
    }

    #[test]
    fn end_point_on_other_segment_correctly_detects_intersection_2() {
        let segments = vec![
            [
                Point::new(20.7622, 389.67697),
                Point::new(40.08881, 366.73386),
            ],
            [
                Point::new(45.58288, 397.7746),
                Point::new(27.07552, 382.18222),
            ],
        ];

        let mut event_queue = EventQueue::from_segments(&segments);
        let mut status = SkipList::new();

        let mut all_new_event_points = Vec::new();
        let mut all_new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (new_events, new_intersections) = update_status(
                &mut status,
                event.clone(),
                &segments,
                all_new_intersections.last_mut(),
                false,
            );
            all_new_event_points.extend(new_events.clone());
            all_new_intersections.extend(new_intersections);
            for new_event in new_events {
                event_queue.push(new_event);
            }
        }

        assert!(all_new_event_points.is_empty());
        assert_eq!(all_new_intersections.len(), 1);

        assert!(points_are_close(
            all_new_intersections[0].position(),
            Point::new(27.07552, 382.18222),
        ));

        assert!(same_intersecting_segments(
            all_new_intersections[0]
                .intersecting_segment_indices
                .clone(),
            vec![0, 1]
        ));
    }

    #[test]
    fn temp() {
        let segments = vec![
            [
                Point::new(432.74377, 443.7191),
                Point::new(434.66434, 424.56927),
            ],
            [
                Point::new(453.9633, 426.55057),
                Point::new(427.1264, 423.7954),
            ],
        ];

        let intersections = find_interesctions(&segments, false);

        assert_eq!(intersections.len(), 1);
        assert!(points_are_close(
            intersections[0].position(),
            calc_intersection_point_unbounded(segments[0], segments[1])
        ));
    }

    #[test]
    fn right_angle_intersection() {
        let segments = vec![
            [Point::new(2.0, 1.0), Point::new(2.0, 4.0)],
            [Point::new(2.0, 4.0), Point::new(4.0, 4.0)],
            [Point::new(4.0, 2.0), Point::new(4.0, 4.0)],
        ];

        let intersections = find_interesctions(&segments, false);

        assert_eq!(intersections.len(), 2);
        assert!(intersections[0].position() == Point::new(2.0, 4.0));
        assert!(intersections[1].position() == Point::new(4.0, 4.0));
    }

    #[test]
    fn simple_split_segments_preprocess() {
        let mut segments = vec![
            [Point::new(0.0, 0.0), Point::new(4.0, 2.0)],
            [Point::new(2.0, 1.0), Point::new(2.0, 4.0)],
            [Point::new(2.0, 4.0), Point::new(4.0, 4.0)],
            [Point::new(4.0, 2.0), Point::new(4.0, 4.0)],
        ];

        let intersections = find_interesctions(&segments, false);

        let mut all_intersections_set =
            HashSet::<IntersectionPoint>::from_iter(intersections.clone());

        segments.iter().enumerate().for_each(|(i, segment)| {
            all_intersections_set.insert(IntersectionPoint {
                position: segment[0],
                intersecting_segment_indices: vec![i],
            });
            all_intersections_set.insert(IntersectionPoint {
                position: segment[1],
                intersecting_segment_indices: vec![i],
            });
        });

        let mut all_intersections: Vec<IntersectionPoint> =
            all_intersections_set.into_iter().collect();

        let new_segments = split_segments_at_intersections(&mut all_intersections, &mut segments);

        assert!(all_intersections.contains(&IntersectionPoint {
            position: Point::new(2.0, 1.0),
            intersecting_segment_indices: vec![0, 1, 4]
        }));

        assert_eq!(segment_start(new_segments[0]), Point::new(2.0, 1.0));
        assert_eq!(segment_end(new_segments[0]), Point::new(0.0, 0.0));
    }

    #[test]
    fn complex_split_segments_preprocess() {
        let mut segments = vec![
            [Point::new(0.0, 0.0), Point::new(4.0, 2.0)],
            [Point::new(2.0, 3.0), Point::new(2.0, 4.0)],
            [Point::new(2.0, 4.0), Point::new(4.0, 4.0)],
            [Point::new(4.0, 2.0), Point::new(4.0, 4.0)],
            [Point::new(2.0, 3.0), Point::new(6.0, 2.0)],
            [Point::new(6.0, 2.0), Point::new(0.0, 1.0)],
        ];

        let intersections = find_interesctions(&segments, false);

        let mut all_intersections_set =
            HashSet::<IntersectionPoint>::from_iter(intersections.clone());

        segments.iter().enumerate().for_each(|(i, segment)| {
            all_intersections_set.insert(IntersectionPoint {
                position: segment[0],
                intersecting_segment_indices: vec![i],
            });
            all_intersections_set.insert(IntersectionPoint {
                position: segment[1],
                intersecting_segment_indices: vec![i],
            });
        });

        let mut all_intersections: Vec<IntersectionPoint> =
            all_intersections_set.into_iter().collect();

        let new_segments = split_segments_at_intersections(&mut all_intersections, &mut segments);

        assert!(all_intersections.contains(&IntersectionPoint {
            position: Point::new(4.0, 2.5),
            intersecting_segment_indices: vec![3, 4, 6]
        }));

        assert!(all_intersections.contains(&IntersectionPoint {
            position: Point::new(3.0, 1.5),
            intersecting_segment_indices: vec![0, 5, 7]
        }));

        let expected_new_segments = vec![
            [Point::new(4.0, 2.5), Point::new(6.0, 2.0)],
            [Point::new(4.0, 2.5), Point::new(4.0, 2.0)],
            [Point::new(3.0, 1.5), Point::new(0.0, 1.0)],
            [Point::new(3.0, 1.5), Point::new(0.0, 0.0)],
        ];

        assert_eq!(new_segments.len(), expected_new_segments.len());

        for new_segment in &expected_new_segments {
            assert!(new_segments.iter().any(|segment| segment_start(*segment)
                == segment_start(*new_segment)
                && segment_end(*segment) == segment_end(*new_segment)));
        }
    }

    #[test]
    fn simple_segments_to_adjacency_list() {
        let mut segments = vec![
            [Point::new(0.0, 0.0), Point::new(4.0, 2.0)],
            [Point::new(2.0, 1.0), Point::new(2.0, 4.0)],
            [Point::new(2.0, 4.0), Point::new(4.0, 4.0)],
            [Point::new(4.0, 2.0), Point::new(4.0, 4.0)],
        ];

        let intersections = find_interesctions(&segments, true);

        let mut all_intersections_set =
            HashSet::<IntersectionPoint>::from_iter(intersections.clone());

        segments.iter().enumerate().for_each(|(i, segment)| {
            all_intersections_set.insert(IntersectionPoint {
                position: segment[0],
                intersecting_segment_indices: vec![i],
            });
            all_intersections_set.insert(IntersectionPoint {
                position: segment[1],
                intersecting_segment_indices: vec![i],
            });
        });

        let mut all_intersections: Vec<IntersectionPoint> =
            all_intersections_set.into_iter().collect();

        let new_segments = split_segments_at_intersections(&mut all_intersections, &mut segments);

        let all_segments: Vec<Segment> = segments.into_iter().chain(new_segments).collect();

        let (vertices, adjacency_list) =
            vertices_to_adjacency_list(all_intersections, &all_segments);

        let expected_vertices = [
            Point::new(0.0, 0.0),
            Point::new(4.0, 2.0),
            Point::new(2.0, 1.0),
            Point::new(2.0, 4.0),
            Point::new(4.0, 4.0),
        ];

        assert_eq!(vertices.len(), expected_vertices.len());
        for vertex in &expected_vertices {
            assert!(expected_vertices.contains(vertex));
        }

        let expected_adjacency_list = [
            vec![Point::new(2.0, 1.0)],
            vec![Point::new(2.0, 1.0), Point::new(4.0, 4.0)],
            vec![
                Point::new(0.0, 0.0),
                Point::new(4.0, 2.0),
                Point::new(2.0, 4.0),
            ],
            vec![Point::new(2.0, 1.0), Point::new(4.0, 4.0)],
            vec![Point::new(2.0, 4.0), Point::new(4.0, 2.0)],
        ];

        for (i, expected_neighbors) in expected_adjacency_list.iter().enumerate() {
            let real_index_of_vertex = vertices
                .iter()
                .position(|vert| vert == &expected_vertices[i])
                .unwrap();
            let real_vertex_neighbors = &adjacency_list[&real_index_of_vertex];

            let real_indices_of_expected_neighbors = expected_neighbors.iter().map(|neighbor| {
                vertices
                    .iter()
                    .position(|real_vert| real_vert == neighbor)
                    .unwrap()
            });
            let expected_neighbors_set = HashSet::from_iter(real_indices_of_expected_neighbors);

            assert_eq!(
                real_vertex_neighbors
                    .difference(&expected_neighbors_set)
                    .count(),
                0
            );
        }
    }

    #[test]
    fn straight_line_to_adjacency_list() {
        let mut segments = vec![
            [
                Point::new(466.0258, 498.15332),
                Point::new(469.41873, 464.32245),
            ],
            [
                Point::new(469.41873, 464.32245),
                Point::new(471.41458, 444.42194),
            ],
            [
                Point::new(471.41458, 444.42194),
                Point::new(473.41043, 424.52142),
            ],
            [
                Point::new(473.41043, 424.52142),
                Point::new(475.40628, 404.6209),
            ],
            [
                Point::new(475.40628, 404.6209),
                Point::new(477.3223, 385.51642),
            ],
        ];

        let (vertices, adjacency_list) = segments_to_adjacency_list(&mut segments);

        let expected_vertices = vec![
            Point::new(466.0258, 498.15332),
            Point::new(469.41873, 464.32245),
            Point::new(471.41458, 444.42194),
            Point::new(473.41043, 424.52142),
            Point::new(475.40628, 404.6209),
            Point::new(477.3223, 385.51642),
        ];

        let expected_adjacency_list: AdjacencyList = HashMap::from_iter(vec![
            (0, HashSet::from_iter(vec![1])),
            (1, HashSet::from_iter(vec![0, 2])),
            (2, HashSet::from_iter(vec![1, 3])),
            (3, HashSet::from_iter(vec![2, 4])),
            (4, HashSet::from_iter(vec![3, 5])),
            (5, HashSet::from_iter(vec![4])),
        ]);

        assert_eq!(vertices.len(), expected_vertices.len());
        for vert in &expected_vertices {
            assert!(vertices.contains(vert));
        }

        for (vert_index, vert_neighbors_indices) in adjacency_list {
            let expected_vert_index = expected_vertices
                .iter()
                .position(|pos| pos == &vertices[vert_index])
                .unwrap();
            let expected_neighbors_indices = &expected_adjacency_list[&expected_vert_index];
            let expected_neighbors_vertices: HashSet<(OrderedFloat<f32>, OrderedFloat<f32>)> =
                expected_neighbors_indices
                    .iter()
                    .map(|&idx| {
                        (
                            OrderedFloat(expected_vertices[idx].x),
                            OrderedFloat(expected_vertices[idx].y),
                        )
                    })
                    .collect();
            let real_neighbor_vertices: HashSet<(OrderedFloat<f32>, OrderedFloat<f32>)> =
                vert_neighbors_indices
                    .iter()
                    .map(|&idx| (OrderedFloat(vertices[idx].x), OrderedFloat(vertices[idx].y)))
                    .collect();

            assert_eq!(
                expected_neighbors_vertices.len(),
                real_neighbor_vertices.len()
            );
            assert_eq!(
                expected_neighbors_vertices
                    .difference(&real_neighbor_vertices)
                    .count(),
                0
            );
        }
    }

    #[test]
    fn straight_line_to_dcel() {
        let curve = vec![
            ControlPoint {
                position: Point::new(466.0258, 498.15332),
                velocity: Point::zeros(),
            },
            ControlPoint {
                position: Point::new(469.41873, 464.32245),
                velocity: Point::zeros(),
            },
            ControlPoint {
                position: Point::new(471.41458, 444.42194),
                velocity: Point::zeros(),
            },
            ControlPoint {
                position: Point::new(473.41043, 424.52142),
                velocity: Point::zeros(),
            },
            ControlPoint {
                position: Point::new(475.40628, 404.6209),
                velocity: Point::zeros(),
            },
            ControlPoint {
                position: Point::new(477.3223, 385.51642),
                velocity: Point::zeros(),
            },
        ];

        let faces = path_to_graph(&[curve]);

        assert!(faces.is_empty());
    }

    #[test]
    fn weird_intersection_edge_case() {
        let segments = vec![
            [
                Point::new(48.95642, 511.74203),
                Point::new(28.98356, 492.7882),
            ],
            [
                Point::new(30.06656, 511.51257),
                Point::new(52.61444, 484.7456),
            ],
            /* [
                Point::new(437.64716, 507.6564),
                Point::new(418.54266, 406.7404),
            ], */
        ];

        let intersections = find_interesctions(&segments, true);

        assert_eq!(intersections.len(), 5);
        let intersections_set: HashSet<IntersectionPoint> = HashSet::from_iter(intersections);
        let expected_intersections = HashSet::from_iter(vec![
            IntersectionPoint {
                position: Point::new(48.95642, 511.74203),
                intersecting_segment_indices: Vec::new(),
            },
            IntersectionPoint {
                position: Point::new(28.98356, 492.7882),
                intersecting_segment_indices: Vec::new(),
            },
            IntersectionPoint {
                position: Point::new(30.06656, 511.51257),
                intersecting_segment_indices: Vec::new(),
            },
            IntersectionPoint {
                position: Point::new(52.61444, 484.7456),
                intersecting_segment_indices: Vec::new(),
            },
            IntersectionPoint {
                position: calc_intersection_point_unbounded(segments[0], segments[1]),
                intersecting_segment_indices: Vec::new(),
            },
        ]);

        assert_eq!(
            intersections_set
                .difference(&expected_intersections)
                .count(),
            0
        );
    }

    #[test]
    fn weird_start_edge_case() {
        let segments = vec![
            [Point::new(18.95, 415.7), Point::new(0.2697, 437.9)],
            [Point::new(19.07, 423.7), Point::new(0.3828, 445.9)],
            [Point::new(7.614, 429.2), Point::new(22.3, 441.6)],
            [Point::new(19.03, 431.5), Point::new(0.3436, 453.7)],
            [Point::new(32.03, 424.1), Point::new(19.14, 439.4)],
            [Point::new(31.99, 431.9), Point::new(19.1, 447.2)],
            [Point::new(29.0, 433.6), Point::new(44.3, 446.5)],
            [Point::new(49.5, 419.2), Point::new(32.11, 439.9)],
            [Point::new(44.39, 428.1), Point::new(32.2, 455.6)],
            [Point::new(50.45, 431.2), Point::new(63.35, 442.6)],
            [Point::new(57.94, 432.8), Point::new(45.05, 448.1)],
            [Point::new(42.4, 417.7), Point::new(62.91, 435.0)],
            [Point::new(440.4, 415.4), Point::new(440.5, 434.7)],
            [Point::new(73.53, 422.4), Point::new(60.64, 437.7)],
            [Point::new(76.24, 427.3), Point::new(63.35, 442.6)],
            [Point::new(112.0, 414.2), Point::new(91.27, 438.9)],
            [Point::new(78.76, 418.9), Point::new(102.5, 438.9)],
            [Point::new(109.4, 425.8), Point::new(91.34, 447.2)],
            [Point::new(129.8, 421.2), Point::new(117.8, 436.1)],
            [Point::new(138.0, 433.7), Point::new(157.7, 437.2)],
            [Point::new(156.7, 426.9), Point::new(154.5, 447.8)],
            [Point::new(180.1, 430.4), Point::new(178.1, 450.3)],
            [Point::new(181.4, 433.6), Point::new(199.5, 439.5)],
            [Point::new(200.1, 424.8), Point::new(198.1, 444.7)],
            [Point::new(208.6, 433.0), Point::new(239.6, 447.1)],
            [Point::new(219.3, 419.5), Point::new(217.3, 439.4)],
            [Point::new(242.9, 429.4), Point::new(243.1, 448.7)],
            [Point::new(244.0, 431.0), Point::new(262.0, 439.3)],
            [Point::new(294.4, 430.9), Point::new(308.2, 440.1)],
            [Point::new(306.8, 430.2), Point::new(311.5, 463.1)],
            [Point::new(321.2, 429.1), Point::new(338.0, 437.7)],
            [Point::new(342.1, 432.7), Point::new(368.0, 446.2)],
            [Point::new(360.0, 432.2), Point::new(386.3, 444.9)],
            [Point::new(396.4, 431.5), Point::new(416.1, 438.4)],
            [Point::new(421.0, 430.6), Point::new(419.0, 450.5)],
            [Point::new(426.1, 431.3), Point::new(424.1, 451.2)],
            [Point::new(440.5, 434.7), Point::new(438.5, 454.6)],
            [Point::new(433.7, 434.1), Point::new(450.5, 435.7)],
            [Point::new(452.4, 416.6), Point::new(450.5, 435.7)],
            [Point::new(472.4, 434.5), Point::new(489.5, 436.2)],
        ];

        let intersections = find_interesctions(&segments, false);

        assert!(!intersections.is_empty());
    }

    #[test]
    fn complex_intersections_with_tiny_gaps() {
        let segments = vec![
            [
                Point::new(44.24656, 322.55533),
                Point::new(57.66624, 307.74054),
            ],
            [
                Point::new(62.39522, 313.1381),
                Point::new(46.17724, 294.69046),
            ],
            [
                Point::new(57.66624, 307.74054),
                Point::new(85.21461, 286.23663),
            ],
            [
                Point::new(49.10064, 328.07104),
                Point::new(62.39522, 313.1381),
            ],
            [
                Point::new(53.2079, 327.27344),
                Point::new(29.92469, 309.57895),
            ],
        ];

        let intersections = find_interesctions(&segments, false);
        assert_eq!(intersections.len(), 5);
    }

    #[test]
    fn complex_intersection_and_split_case() {
        let mut segments = vec![
            [
                Point::new(93.185, 311.78308),
                Point::new(69.25536, 282.38647),
            ],
            [
                Point::new(76.59073, 317.20007),
                Point::new(97.12109, 294.93295),
            ],
            [
                Point::new(61.24543, 334.08725),
                Point::new(76.59073, 317.20007),
            ],
            [
                Point::new(84.59501, 327.1706),
                Point::new(69.01391, 312.52502),
            ],
            [
                Point::new(79.88962, 314.0654),
                Point::new(66.12833, 297.94373),
            ],
            [
                Point::new(97.12109, 294.93295),
                Point::new(113.18362, 309.93332),
            ],
            [
                Point::new(84.14339, 278.44148),
                Point::new(97.12109, 294.93295),
            ],
            [
                Point::new(106.17567, 303.3888),
                Point::new(80.89961, 331.2429),
            ],
            [
                Point::new(104.84992, 320.941),
                Point::new(93.185, 311.78308),
            ],
            [
                Point::new(62.39522, 313.1381),
                Point::new(91.93122, 288.34058),
            ],
            [
                Point::new(57.66624, 307.74054),
                Point::new(85.21461, 286.23663),
            ],
        ];

        let mut intersections_vec = find_interesctions(&segments, true);

        for intersection in &intersections_vec {
            assert!(!intersection.intersecting_segment_indices.is_empty());
        }

        let intersections: HashSet<IntersectionPoint> =
            HashSet::from_iter(intersections_vec.clone());

        assert_eq!(intersections.len(), 26);

        let new_split_segments =
            split_segments_at_intersections(&mut intersections_vec, &mut segments);

        let all_segments: Vec<Segment> = segments.into_iter().chain(new_split_segments).collect();

        let expected_adjacency_list = vec![
            (
                Point::new(79.69314, 313.8352),
                vec![
                    Point::new(76.59073, 317.20007),
                    Point::new(79.88962, 314.0654),
                    Point::new(72.12515, 304.96915),
                    Point::new(87.74584, 305.1013),
                ],
            ),
            (
                Point::new(77.37251, 292.35806),
                vec![
                    Point::new(67.76318, 299.859),
                    Point::new(81.340324, 297.23236),
                    Point::new(85.21461, 286.23663),
                    Point::new(69.25536, 282.38647),
                ],
            ),
            (
                Point::new(69.25536, 282.38647),
                vec![Point::new(77.37251, 292.35806)],
            ),
            (
                Point::new(66.12833, 297.94373),
                vec![Point::new(67.76318, 299.859)],
            ),
            (
                Point::new(57.66624, 307.74054),
                vec![Point::new(67.76318, 299.859)],
            ),
            (
                Point::new(67.76318, 299.859),
                vec![
                    Point::new(57.66624, 307.74054),
                    Point::new(66.12833, 297.94373),
                    Point::new(77.37251, 292.35806),
                    Point::new(72.12515, 304.96915),
                ],
            ),
            (
                Point::new(84.59501, 327.1706),
                vec![
                    Point::new(80.89961, 331.2429),
                    Point::new(96.322845, 314.24655),
                    Point::new(75.391556, 318.51974),
                ],
            ),
            (
                Point::new(76.59073, 317.20007),
                vec![
                    Point::new(75.391556, 318.51974),
                    Point::new(79.69314, 313.8352),
                ],
            ),
            (
                Point::new(61.24543, 334.08725),
                vec![Point::new(75.391556, 318.51974)],
            ),
            (
                Point::new(80.89961, 331.2429),
                vec![Point::new(84.59501, 327.1706)],
            ),
            (
                Point::new(96.322845, 314.24655),
                vec![
                    Point::new(84.59501, 327.1706),
                    Point::new(104.84992, 320.941),
                    Point::new(106.17567, 303.3888),
                    Point::new(93.185, 311.78308),
                ],
            ),
            (
                Point::new(87.74584, 305.1013),
                vec![
                    Point::new(79.69314, 313.8352),
                    Point::new(93.185, 311.78308),
                    Point::new(97.12109, 294.93295),
                    Point::new(81.340324, 297.23236),
                ],
            ),
            (
                Point::new(84.14339, 278.44147),
                vec![Point::new(91.93122, 288.34058)],
            ),
            (
                Point::new(93.185, 311.78308),
                vec![
                    Point::new(96.322845, 314.24655),
                    Point::new(87.74584, 305.1013),
                ],
            ),
            (
                Point::new(91.93122, 288.34058),
                vec![
                    Point::new(81.340324, 297.23236),
                    Point::new(97.12109, 294.93295),
                    Point::new(84.14339, 278.44147),
                ],
            ),
            (
                Point::new(81.3403244, 297.23236),
                vec![
                    Point::new(72.12515, 304.96915),
                    Point::new(87.74584, 305.1013),
                    Point::new(91.93122, 288.34058),
                    Point::new(77.37251, 292.35806),
                ],
            ),
            (
                Point::new(69.01391, 312.52502),
                vec![Point::new(75.391556, 318.51974)],
            ),
            (
                Point::new(104.84992, 320.941),
                vec![Point::new(96.322845, 314.24655)],
            ),
            (
                Point::new(113.18362, 309.93332),
                vec![Point::new(106.17567, 303.3888)],
            ),
            (
                Point::new(79.88962, 314.0654),
                vec![Point::new(79.69314, 313.8352)],
            ),
            (
                Point::new(62.39522, 313.1381),
                vec![Point::new(72.12515, 304.96915)],
            ),
            (
                Point::new(72.12515, 304.96915),
                vec![
                    Point::new(79.69314, 313.8352),
                    Point::new(81.340324, 297.23236),
                    Point::new(67.76318, 299.859),
                    Point::new(62.39522, 313.1381),
                ],
            ),
            (
                Point::new(106.17567, 303.3888),
                vec![
                    Point::new(113.18362, 309.93332),
                    Point::new(97.12109, 294.93295),
                    Point::new(96.322845, 314.24655),
                ],
            ),
            (
                Point::new(75.391556, 318.51974),
                vec![
                    Point::new(61.24543, 334.08725),
                    Point::new(84.59501, 327.1706),
                    Point::new(76.59073, 317.20007),
                    Point::new(69.01391, 312.52502),
                ],
            ),
            (
                Point::new(97.12109, 294.93295),
                vec![
                    Point::new(106.17567, 303.3888),
                    Point::new(91.93122, 288.34058),
                    Point::new(87.74584, 305.1013),
                ],
            ),
            (
                Point::new(85.21461, 286.23663),
                vec![Point::new(77.37251, 292.35806)],
            ),
        ];

        let (verts, adjacency_list) = vertices_to_adjacency_list(intersections_vec, &all_segments);

        for (vert_index, vert_neighbors_indices) in &adjacency_list {
            let vert_pos = verts[*vert_index];

            let expected_neighbors = HashSet::from_iter(
                expected_adjacency_list[expected_adjacency_list
                    .iter()
                    .position(|(vert, _)| vert == &vert_pos)
                    .unwrap()]
                .1
                .iter()
                .map(|point| (OrderedFloat(point.x), OrderedFloat(point.y))),
            );

            let real_vert_neighbors: HashSet<(OrderedFloat<f32>, OrderedFloat<f32>)> =
                vert_neighbors_indices
                    .iter()
                    .map(|&idx| (OrderedFloat(verts[idx].x), OrderedFloat(verts[idx].y)))
                    .collect();

            assert_eq!(real_vert_neighbors.len(), expected_neighbors.len());

            assert_eq!(
                real_vert_neighbors.difference(&expected_neighbors).count(),
                0
            );
        }
    }

    #[test]
    fn complex_intersections_correctly_detects_faces() {
        let mut segments = vec![
            [
                Point::new(93.185, 311.78308),
                Point::new(69.25536, 282.38647),
            ],
            [
                Point::new(76.59073, 317.20007),
                Point::new(97.12109, 294.93295),
            ],
            [
                Point::new(61.24543, 334.08725),
                Point::new(76.59073, 317.20007),
            ],
            [
                Point::new(84.59501, 327.1706),
                Point::new(69.01391, 312.52502),
            ],
            [
                Point::new(79.88962, 314.0654),
                Point::new(66.12833, 297.94373),
            ],
            [
                Point::new(97.12109, 294.93295),
                Point::new(113.18362, 309.93332),
            ],
            [
                Point::new(84.14339, 278.44148),
                Point::new(97.12109, 294.93295),
            ],
            [
                Point::new(106.17567, 303.3888),
                Point::new(80.89961, 331.2429),
            ],
            [
                Point::new(104.84992, 320.941),
                Point::new(93.185, 311.78308),
            ],
            [
                Point::new(62.39522, 313.1381),
                Point::new(91.93122, 288.34058),
            ],
            [
                Point::new(57.66624, 307.74054),
                Point::new(85.21461, 286.23663),
            ],
        ];

        let (vertices, adjacency_list) = segments_to_adjacency_list(&mut segments);
        let dcel = DCEL::new(&vertices, adjacency_list);

        assert_eq!(dcel.faces().len(), 5);
    }

    #[test]
    fn simple_joint_segment_split() {
        let mut segments = vec![
            [Point::new(0.0, 4.0), Point::new(4.0, 0.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, 0.0)],
            [Point::new(0.0, 1.0), Point::new(1.0, 0.0)],
            [Point::new(3.0, 0.0), Point::new(4.0, 1.0)],
        ];

        let mut intersections = find_interesctions(&segments, true);

        let new_segments = split_segments_at_intersections(&mut intersections, &mut segments);

        let all_segments: Vec<Segment> = segments.into_iter().chain(new_segments).collect();

        let expected_segments = vec![
            [Point::new(2.0, 2.0), Point::new(0.5, 0.5)],
            [Point::new(0.0, 4.0), Point::new(2.0, 2.0)],
            [Point::new(2.0, 2.0), Point::new(3.5, 0.5)],
            [Point::new(0.5, 0.5), Point::new(0.0, 0.0)],
            [Point::new(3.5, 0.5), Point::new(4.0, 0.0)],
            [Point::new(0.0, 1.0), Point::new(0.5, 0.5)],
            [Point::new(0.5, 0.5), Point::new(1.0, 0.0)],
            [Point::new(4.0, 1.0), Point::new(3.5, 0.5)],
            [Point::new(3.5, 0.5), Point::new(3.0, 0.0)],
        ];

        assert_eq!(all_segments.len(), expected_segments.len());

        for expected_segment in expected_segments {
            assert!(all_segments.iter().any(|real_segment| {
                segment_start(*real_segment) == segment_start(expected_segment)
                    && segment_end(*real_segment) == segment_end(expected_segment)
            }));
        }

        let expected_intersections = vec![
            IntersectionPoint {
                position: Point::new(0.0, 4.0),
                intersecting_segment_indices: vec![0],
            },
            IntersectionPoint {
                position: Point::new(2.0, 2.0),
                intersecting_segment_indices: vec![0, 1, 4],
            },
            IntersectionPoint {
                position: Point::new(0.5, 0.5),
                intersecting_segment_indices: vec![1, 2, 6, 7],
            },
            IntersectionPoint {
                position: Point::new(3.5, 0.5),
                intersecting_segment_indices: vec![4, 3, 5, 8],
            },
            IntersectionPoint {
                position: Point::new(0.0, 1.0),
                intersecting_segment_indices: vec![2],
            },
            IntersectionPoint {
                position: Point::new(0.0, 0.0),
                intersecting_segment_indices: vec![6],
            },
            IntersectionPoint {
                position: Point::new(1.0, 0.0),
                intersecting_segment_indices: vec![7],
            },
            IntersectionPoint {
                position: Point::new(4.0, 1.0),
                intersecting_segment_indices: vec![3],
            },
            IntersectionPoint {
                position: Point::new(4.0, 0.0),
                intersecting_segment_indices: vec![5],
            },
            IntersectionPoint {
                position: Point::new(3.0, 0.0),
                intersecting_segment_indices: vec![8],
            },
        ];

        for expected_intersection in expected_intersections {
            assert!(intersections.iter().any(
                |IntersectionPoint {
                     position,
                     intersecting_segment_indices,
                 }| {
                    position == &expected_intersection.position()
                        && same_intersecting_segments(
                            intersecting_segment_indices.clone(),
                            expected_intersection.intersecting_segment_indices.clone(),
                        )
                },
            ));
        }
    }

    #[test]
    fn simple_degenerate_point_correcting() {
        let face = vec![
            Point::new(362.75317, 138.70242),
            Point::new(368.2237, 158.12653),
            Point::new(350.72037, 163.93617),
            Point::new(346.73752, 147.39384),
            Point::new(345.7601, 145.33516),
            Point::new(353.23737, 142.41664),
            Point::new(356.65186, 150.63629),
            Point::new(353.23737, 142.41664),
        ];

        let new_faces = correct_face_with_degenerate_points(face);
        assert_eq!(new_faces.len(), 2);
    }

    #[test]
    fn complex_degenerate_point_correcting() {
        let face = vec![
            Point::new(122.73693, 142.97884),
            Point::new(134.47775, 150.8719),
            Point::new(137.1224, 152.64983),
            Point::new(146.1432, 159.88219),
            Point::new(147.78151, 146.4958),
            Point::new(149.52223, 142.25769),
            Point::new(155.51039, 127.67841),
            Point::new(155.08409, 127.0715),
            Point::new(150.10312, 117.55582),
            Point::new(147.04967, 111.72247),
            Point::new(150.10312, 117.55582),
            Point::new(154.81042, 108.40404),
            Point::new(144.70998, 95.5031),
            Point::new(139.94247, 101.33901),
            Point::new(140.4403, 102.07162),
            Point::new(144.80272, 124.08911),
            Point::new(140.4403, 102.07162),
            Point::new(139.94247, 101.33901),
            Point::new(132.1, 112.22),
            Point::new(139.73, 124.84),
            Point::new(136.469, 140.9366),
            Point::new(139.73, 124.84),
            Point::new(132.1, 112.22),
            Point::new(126.36971, 120.84164),
            Point::new(131.39497, 139.17522),
            Point::new(126.36971, 120.84164),
            Point::new(122.59472, 131.74731),
            Point::new(121.36434, 136.55379),
        ];

        let new_faces = correct_face_with_degenerate_points(face);
        assert_eq!(new_faces.len(), 6);
    }

    #[test]
    fn correcting_degenerate_points_does_nothing_with_no_degenerate_points() {
        let face = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];

        let new_faces = correct_face_with_degenerate_points(face);
        assert_eq!(new_faces.len(), 1);
    }
}
