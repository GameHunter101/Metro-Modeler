use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use cool_utils::data_structures::dcel::DCEL;
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

pub fn segment_start(segment: Segment) -> Point {
    if segment[0].y > segment[1].y {
        segment[0]
    } else if segment[0].y == segment[1].y {
        if segment[0].x < segment[1].x {
            segment[0]
        } else {
            segment[1]
        }
    } else {
        segment[1]
    }
}

pub fn segment_end(segment: Segment) -> Point {
    if segment[0].y > segment[1].y {
        segment[1]
    } else if segment[0].y == segment[1].y {
        if segment[0].x < segment[1].x {
            segment[1]
        } else {
            segment[0]
        }
    } else {
        segment[0]
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

    dbg!(
        &sorted_segment_indices,
        &potential_left_neighbor,
        &potential_right_neighbor
    );

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
    end_points_are_intersections: bool,
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
    }

    if end_points_are_intersections {
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
    new_intersections: Vec<usize>,
    segments: &mut [Segment],
) -> Vec<Segment> {
    let inverse_intersections: HashMap<IntersectionPoint, usize> = HashMap::from_iter(
        all_intersections
            .into_iter()
            .enumerate()
            .map(|(i, intersection)| (intersection.clone(), i)),
    );

    let mut new_segments = Vec::new();

    for intersection_index in new_intersections {
        let (pre, post_and_intersection) = all_intersections.split_at_mut(intersection_index);
        let (intersection, post) = post_and_intersection.split_first_mut().unwrap();
        let mut other_intersections: Vec<&mut IntersectionPoint> =
            pre.into_iter().chain(post).collect();

        for intersecting_segment_index in intersection.intersecting_segment_indices.clone() {
            if intersecting_segment_index >= segments.len() {
                continue;
            }
            let intersecting_segment = segments[intersecting_segment_index];
            if intersecting_segment[0] != intersection.position()
                && intersecting_segment[1] != intersection.position()
            {
                let new_segment_index = segments.len() + new_segments.len();
                new_segments.push([intersection.position(), intersecting_segment[1]]);
                intersection
                    .intersecting_segment_indices
                    .push(new_segment_index);
                let intersecting_segment_endpoint_index = inverse_intersections
                    [&IntersectionPoint {
                        position: intersecting_segment[1],
                        intersecting_segment_indices: Vec::new(),
                    }];

                other_intersections[intersecting_segment_endpoint_index
                    - if intersecting_segment_endpoint_index < intersection_index {
                        0
                    } else {
                        1
                    }]
                .replace_intersecting_index(intersecting_segment_index, new_segment_index);

                segments[intersecting_segment_index][1] = intersection.position();
            }
        }
    }

    new_segments
}

pub fn path_to_graph(paths: &[HermiteCurve]) -> DCEL {
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

    /* let intersections: HashSet<IntersectionPoint> =
        HashSet::from_iter(find_interesctions(&segments, false));

    let mut all_intersections_set = intersections.clone();

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

    let mut all_intersections: Vec<IntersectionPoint> = all_intersections_set.into_iter().collect();

    let new_intersection_indices = all_intersections
        .iter()
        .enumerate()
        .flat_map(|(i, intersection)| {
            if intersections.contains(intersection) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let new_segments = split_segments_at_intersections(
        &mut all_intersections,
        new_intersection_indices,
        &mut segments,
    );

    let all_segments: Vec<Segment> = segments.into_iter().chain(new_segments).collect(); */
    println!(
        "Segments: {:?}",
        segments
            .iter()
            .flat_map(|[p_0, p_1]| [(p_0.x, p_0.y), (p_1.x, p_1.y)])
            .collect::<Vec<_>>()
    );
    let (vertices, adjacency_list) = segments_to_adjacency_list(&mut segments);
    println!(
        "Verts: {:?}",
        vertices
            .iter()
            .map(|point| (point.x, point.y))
            .collect::<Vec<_>>()
    );

    // dbg!(&adjacency_list);

    DCEL::new(vertices, adjacency_list)
}

fn vertices_to_adjacency_list(
    vertices: Vec<IntersectionPoint>,
    segments: &[Segment],
) -> (Vec<Point>, HashMap<usize, HashSet<usize>>) {
    let inverse_vertices: HashMap<&IntersectionPoint, usize> = HashMap::from_iter(
        vertices
            .iter()
            .enumerate()
            .map(|(i, intersection)| (intersection, i)),
    );

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
                            position: vertex,
                            intersecting_segment_indices: Vec::new(),
                        })
                        .expect(&format!("No inverse for {vertex:?}"))
                })
                .filter(|&vertex_index| vertex_index != i)
                .collect();
            (i, all_conneced_vertices)
        })
        .collect();

    dbg!(&adjacency_list);

    (
        vertices
            .into_iter()
            .map(|IntersectionPoint { position, .. }| position)
            .collect(),
        adjacency_list,
    )
}

fn segments_to_adjacency_list(
    segments: &mut [Segment],
) -> (Vec<Point>, HashMap<usize, HashSet<usize>>) {
    let intersections: HashSet<IntersectionPoint> =
        HashSet::from_iter(find_interesctions(&segments, true));

    let mut intersections_vec: Vec<IntersectionPoint> = intersections.iter().cloned().collect();

    /* let mut all_intersections_set = intersections.clone();

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

    let mut all_intersections: Vec<IntersectionPoint> = all_intersections_set.into_iter().collect(); */

    let new_intersection_indices = intersections_vec
        .iter()
        .enumerate()
        .flat_map(|(i, intersection)| {
            if intersections.contains(intersection) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    println!(
        "Intersections: {:?}",
        intersections_vec
            .iter()
            .map(|IntersectionPoint { position, .. }| (position.x, position.y))
            .collect::<Vec<_>>()
    );
    let new_segments =
        split_segments_at_intersections(&mut intersections_vec, new_intersection_indices, segments);

    let all_segments: Vec<Segment> = segments.to_vec().into_iter().chain(new_segments).collect();

    vertices_to_adjacency_list(intersections_vec, &all_segments)
}

#[cfg(test)]
mod test {
    use ordered_float::OrderedFloat;

    use crate::{
        street_graph::{
            calc_intersection_point_unbounded, path_to_graph, points_are_close, sort_joint_segments,
        },
        street_plan::ControlPoint,
    };
    use std::{
        collections::{HashMap, HashSet},
        iter,
    };

    use crate::{event_queue::EventQueue, street_graph::IntersectionPoint, tensor_field::Point};

    use super::{
        EventPoint, EventPointType, Segment, SkipList, calc_intersection_point, find_interesctions,
        segments_to_adjacency_list, split_segments_at_intersections, update_status,
        vertices_to_adjacency_list,
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

        let new_intersection_indices = all_intersections
            .iter()
            .enumerate()
            .flat_map(|(i, intersection)| {
                if intersections.contains(intersection) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let new_segments = split_segments_at_intersections(
            &mut all_intersections,
            new_intersection_indices,
            &mut segments,
        );

        assert!(all_intersections.contains(&IntersectionPoint {
            position: Point::new(2.0, 1.0),
            intersecting_segment_indices: vec![0, 1, 4]
        }));

        assert_eq!(
            new_segments,
            vec![[Point::new(2.0, 1.0), Point::new(4.0, 2.0)]]
        );
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

        let new_intersection_indices = all_intersections
            .iter()
            .enumerate()
            .flat_map(|(i, intersection)| {
                if intersections.contains(intersection) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let new_segments = split_segments_at_intersections(
            &mut all_intersections,
            new_intersection_indices,
            &mut segments,
        );

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
            [Point::new(3.0, 1.5), Point::new(4.0, 2.0)],
            [Point::new(3.0, 1.5), Point::new(0.0, 1.0)],
            [Point::new(4.0, 2.5), Point::new(4.0, 4.0)],
        ];

        assert_eq!(new_segments.len(), expected_new_segments.len());

        for new_segment in &expected_new_segments {
            assert!(new_segments.contains(new_segment));
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

        let new_intersection_indices = all_intersections
            .iter()
            .enumerate()
            .flat_map(|(i, intersection)| {
                if intersections.contains(intersection) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let new_segments = split_segments_at_intersections(
            &mut all_intersections,
            new_intersection_indices,
            &mut segments,
        );

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

        for (i, expected_list) in expected_adjacency_list.iter().enumerate() {
            let real_connected_vertices: Vec<IntersectionPoint> = adjacency_list[&vertices
                .iter()
                .position(|vertex| vertex == &expected_vertices[i])
                .unwrap()]
                .iter()
                .map(|index| IntersectionPoint {
                    position: vertices[*index],
                    intersecting_segment_indices: Vec::new(),
                })
                .collect();

            assert_eq!(real_connected_vertices.len(), expected_list.len());

            assert_eq!(
                HashSet::<IntersectionPoint>::from_iter(real_connected_vertices)
                    .difference(&HashSet::from_iter(expected_list.iter().map(|pos| {
                        IntersectionPoint {
                            position: *pos,
                            intersecting_segment_indices: Vec::new(),
                        }
                    })))
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

        let expected_adjacency_list: HashMap<usize, HashSet<usize>> = HashMap::from_iter(vec![
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

        let dcel = path_to_graph(&[curve]);

        assert!(dcel.faces().is_empty());
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
            [Point::new(44.24656, 322.55533), Point::new(57.66624, 307.74054)],
            [Point::new(62.39522, 313.1381), Point::new(46.17724, 294.69046)],
            [Point::new(57.66624, 307.74054), Point::new(85.21461, 286.23663)],
            [Point::new(49.10064, 328.07104), Point::new(62.39522, 313.1381)],
            [Point::new(53.2079, 327.27344), Point::new(29.92469, 309.57895)],
        ];

        let intersections = find_interesctions(&segments, false);
        assert_eq!(intersections.len(), 5);
    }
}


/*
[Start-, End]
[Start-, 616, End]
[Start, 875-, 37, 616, End]
[Start, 756, 875-, 37, 616, End]
[Start, 756, 875, 211-, 263, 39, 37, 418, 616, 933, End]
[Start, 768, 756, 875, 211, 916-, 263, 39, 37, 418, 616, 933, End]
[Start, 519, 807, 768, 756, 875, 211, 916, 722, 263, 39, 327, 638, 37, 418, 616, 933, End]
[Start, 519, 807, 523, 526, 768, 756, 875, 211, 916, 722, 697, 263, 39, 327, 638, 37, 418, 616, 933, End]
[Start, 519, 807, 523, 526, 780, 768, 756, 875, 100, 211, 916, 722, 697, 263, 39, 327, 140, 638, 37, 418, 616, 933, 19, End]
[Start, 519, 807, 523, 526, 780, 768, 756, 875, 100, 211, 809, 916, 722, 697, 263, 39, 706, 327, 140, 638, 37, 418, 616, 933, 19, End]
[Start, 820, 519, 815, 807, 801, 523, 526, 787, 780, 768, 756, 875, 100, 211, 809, 916, 722, 697, 263, 39, 706, 327, 140, 638, 37, 418, 616, 933, 19, End]
[Start, 820, 519, 815, 807, 801, 523, 526, 787, 780, 768, 756, 875, 100, 211, 809, 916, 722, 697, 263, 39, 706, 327, 140, 638, 37, 418, 616, 933, 19, End]
[Start, 820, 519, 815, 807, 801, 523, 794, 526, 787, 780, 768, 832, 756, 875, 100, 211, 809, 916, 722, 697, 263, 668, 39, 706, 327, 140, 638, 381, 37, 418, 616, 933, 610, 19, End]
[Start, 820, 519, 815, 807, 801, 523, 794, 526, 787, 780, 768, 832, 756, 875, 100, 211, 809, 916, 722, 697, 263, 118, 668, 39, 706, 327, 140, 638, 381, 37, 418, 616, 933, 610, 19, End]

(18.95, 415.7), (0.2697, 437.9),(19.07, 423.7), (0.3828, 445.9),(7.614, 429.2), (22.3, 441.6),(19.03, 431.5), (0.3436, 453.7),(32.03, 424.1), (19.14, 439.4),(31.99, 431.9), (19.1, 447.2),(29.0, 433.6), (44.3, 446.5),(49.5, 419.2), (32.11, 439.9),(44.39, 428.1), (32.2, 455.6),(50.45, 431.2), (63.35, 442.6),(57.94, 432.8), (45.05, 448.1),(42.4, 417.7), (62.91, 435.0),(440.4, 415.4), (440.5, 434.7),(73.53, 422.4), (60.64, 437.7),(76.24, 427.3), (63.35, 442.6),(112.0, 414.2), (91.27, 438.9),(78.76, 418.9), (102.5, 438.9),(109.4, 425.8), (91.34, 447.2),(129.8, 421.2), (117.8, 436.1),(138.0, 433.7), (157.7, 437.2),(156.7, 426.9), (154.5, 447.8),(180.1, 430.4), (178.1, 450.3),(181.4, 433.6), (199.5, 439.5),(200.1, 424.8), (198.1, 444.7),(208.6, 433.0), (239.6, 447.1),(219.3, 419.5), (217.3, 439.4),(242.9, 429.4), (243.1, 448.7),(244.0, 431.0), (262.0, 439.3),(294.4, 430.9), (308.2, 440.1),(306.8, 430.2), (311.5, 463.1),(321.2, 429.1), (338.0, 437.7),(342.1, 432.7), (368.0, 446.2),(360.0, 432.2), (386.3, 444.9),(396.4, 431.5), (416.1, 438.4),(421.0, 430.6), (419.0, 450.5),(426.1, 431.3), (424.1, 451.2),(440.5, 434.7), (438.5, 454.6),(433.7, 434.1), (450.5, 435.7),(452.4, 416.6), (450.5, 435.7),(472.4, 434.5), (489.5, 436.2),
*/
