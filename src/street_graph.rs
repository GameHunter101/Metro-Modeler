use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ptr::NonNull;

use crate::tensor_field::Point;

use ordered_float::OrderedFloat;
use rand::prelude::*;

struct IntersectionPoint {
    position: Point,
    intersecting_segment_indices: Vec<usize>,
}

#[derive(PartialEq, Debug, Clone, Copy)]
struct EventPoint {
    position: Point,
    segment_index: usize,
    event_type: EventPointType,
}

#[derive(PartialEq, PartialOrd, Eq, Debug, Clone, Copy)]
enum EventPointType {
    StartPoint,
    Intersection,
    EndPoint,
}

impl Eq for EventPoint {}

impl PartialOrd for EventPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.position.y.partial_cmp(&other.position.y) {
            Some(y_comp) => {
                if y_comp == std::cmp::Ordering::Equal {
                    self.position.x.partial_cmp(&other.position.x)
                } else {
                    Some(y_comp)
                }
            }
            None => None,
        }
    }
}

impl Ord for EventPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

type Link = NonNull<Node>;

type Segment = [Point; 2];

#[derive(Debug, Clone)]
struct Node {
    node_type: NodeType,
    next_ptrs: Vec<Link>,
    prev_ptrs: Vec<Link>,
}

impl Node {
    fn new_empty_chain() -> NonNull<Node> {
        unsafe {
            let end_node = NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                node_type: NodeType::End,
                next_ptrs: Vec::new(),
                prev_ptrs: Vec::new(),
            })));

            let start_node = NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                node_type: NodeType::Start,
                next_ptrs: vec![end_node],
                prev_ptrs: Vec::new(),
            })));

            (*end_node.as_ptr()).prev_ptrs = vec![start_node];

            start_node
        }
    }

    fn traverse_level(
        start: Link,
        level: usize,
        element: usize,
        segments: &[Segment],
        height: f32,
    ) -> (Link, Vec<Link>) {
        unsafe {
            if let Some(next) = start.as_ref().next_ptrs.get(level).copied() {
                if (*next.as_ptr())
                    .node_type
                    .cmp(&NodeType::Value(element), segments, height)
                    == Ordering::Greater
                {
                    if level == 0 {
                        (start, Vec::new())
                    } else {
                        let (node, mut path) =
                            Self::traverse_level(start, level - 1, element, segments, height);
                        path.push(start);
                        (node, path)
                    }
                } else {
                    Self::traverse_level(next, level, element, segments, height)
                }
            } else {
                (start, Vec::new())
            }
        }
    }

    fn append(
        origin: Link,
        end: Link,
        node: Link,
        element: usize,
        traversal_path: Vec<Link>,
        rng: &mut ThreadRng,
    ) -> (Option<usize>, Option<usize>) {
        unsafe {
            let mut next_ptrs = Vec::new();
            let mut prev_ptrs = Vec::new();

            next_ptrs.push(node.as_ref().next_ptrs[0]);
            prev_ptrs.push(node);

            for (level, node) in traversal_path.iter().copied().enumerate() {
                let promotion: bool = rng.random();
                if promotion {
                    next_ptrs.push(node.as_ref().next_ptrs[level + 1]);
                    prev_ptrs.push(node);
                } else {
                    break;
                }
            }

            let new_node = NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                node_type: NodeType::Value(element),
                next_ptrs: next_ptrs.clone(),
                prev_ptrs: prev_ptrs.clone(),
            })));

            for (level, node) in prev_ptrs.iter().enumerate() {
                (&mut (*node.as_ptr()).next_ptrs)[level] = new_node;
            }

            let promotion_count = next_ptrs.len();

            for (level, node) in next_ptrs.iter().enumerate() {
                (&mut (*node.as_ptr()).prev_ptrs)[level] = new_node;
            }

            if (traversal_path.len() == 0 || promotion_count == traversal_path.len())
                && rng.random()
            {
                (*origin.as_ptr()).next_ptrs.push(new_node);
                (*end.as_ptr()).prev_ptrs.push(new_node);

                (*new_node.as_ptr()).next_ptrs.push(end);
                (*new_node.as_ptr()).prev_ptrs.push(origin);
            }

            let prev_neighbor_type = &(*prev_ptrs[0].as_ptr()).node_type;
            let next_neighbor_type = &(*next_ptrs[0].as_ptr()).node_type;

            let prev_neighbor = if let NodeType::Value(val) = prev_neighbor_type {
                Some(*val)
            } else {
                None
            };

            let next_neighbor = if let NodeType::Value(val) = next_neighbor_type {
                Some(*val)
            } else {
                None
            };

            (prev_neighbor, next_neighbor)
        }
    }
}

#[derive(Debug, Clone)]
enum NodeType {
    Start,
    Value(usize),
    End,
}
fn get_x_val_of_segment_at_height(segment: &Segment, height: f32) -> f32 {
    segment[0].x
        + ((height - segment[0].y) / (segment[1] - segment[0]).y) * (segment[1] - segment[0]).x
}

impl NodeType {
    fn cmp(&self, other: &NodeType, segments: &[Segment], height: f32) -> Ordering {
        match self {
            NodeType::Start => Ordering::Less,
            NodeType::Value(lhs) => match other {
                NodeType::Start => Ordering::Greater,
                NodeType::Value(rhs) => {
                    if rhs == lhs {
                        Ordering::Equal
                    } else {
                        get_x_val_of_segment_at_height(&segments[*lhs], height)
                            .partial_cmp(&get_x_val_of_segment_at_height(&segments[*rhs], height))
                            .unwrap_or(Ordering::Less)
                    }
                }
                NodeType::End => Ordering::Less,
            },
            NodeType::End => Ordering::Greater,
        }
    }
}

struct SkipList {
    nodes: Link,
    end: Link,
    rng: ThreadRng,
    len: usize,
}

impl SkipList {
    fn new() -> Self {
        unsafe {
            let nodes = Node::new_empty_chain();
            Self {
                nodes,
                end: nodes.as_ref().next_ptrs[0],
                rng: rand::rng(),
                len: 0,
            }
        }
    }

    fn height(&self) -> usize {
        unsafe { (*self.nodes.as_ptr()).next_ptrs.len() }
    }

    fn insert(
        &mut self,
        element: usize,
        segments: &[Segment],
        height: f32,
    ) -> (Option<usize>, Option<usize>) {
        unsafe {
            let (traverse_node, traverse_path) = Node::traverse_level(
                self.nodes,
                (*self.nodes.as_ptr()).next_ptrs.len() - 1,
                element.clone(),
                segments,
                height,
            );

            self.len += 1;

            Node::append(
                self.nodes,
                self.end,
                traverse_node,
                element,
                traverse_path,
                &mut self.rng,
            )
        }
    }

    fn remove(&mut self, element: usize, segments: &[Segment], height: f32) -> bool {
        unsafe {
            let (traverse_target, _) = Node::traverse_level(
                self.nodes,
                self.height() - 1,
                element.clone(),
                segments,
                height,
            );

            if (*traverse_target.as_ptr()).node_type.cmp(
                &NodeType::Value(element),
                segments,
                height,
            ) == Ordering::Equal
            {
                let boxed_target = Box::from_raw(traverse_target.as_ptr());

                let node_prev_ptrs = boxed_target.prev_ptrs.clone();
                let node_next_ptrs = boxed_target.next_ptrs.clone();

                assert_eq!(node_prev_ptrs.len(), node_next_ptrs.len());

                for i in 0..node_next_ptrs.len() {
                    (&mut (*node_prev_ptrs[i].as_ptr()).next_ptrs)[i] = node_next_ptrs[i];
                    (&mut (*node_next_ptrs[i].as_ptr()).prev_ptrs)[i] = node_prev_ptrs[i];
                }

                self.len -= 1;

                true
            } else {
                false
            }
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self, level: usize) -> Iter {
        unsafe {
            Iter {
                next: Some(self.nodes.as_ref()),
                level,
            }
        }
    }
}

impl Drop for SkipList {
    fn drop(&mut self) {
        unsafe {
            while !(*self.nodes.as_ptr()).next_ptrs.is_empty() {
                let boxed_node = Box::from_raw(self.nodes.as_ptr());
                self.nodes = boxed_node.next_ptrs[0];
            }
            let _ = Box::from_raw(self.nodes.as_ptr());
        }
    }
}

struct Iter<'a> {
    next: Option<&'a Node>,
    level: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a NodeType;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            self.next.take().map(|node| {
                self.next = node.next_ptrs.get(self.level).map(|ptr| ptr.as_ref());
                &node.node_type
            })
        }
    }
}

fn segments_to_event_queue(segments: &[Segment]) -> BinaryHeap<EventPoint> {
    BinaryHeap::from_iter(segments.iter().enumerate().flat_map(|(i, &[start, end])| {
        let (start, end) = if start.y > end.y {
            (start, end)
        } else {
            (end, start)
        };
        [
            EventPoint {
                position: start,
                segment_index: i,
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: end,
                segment_index: i,
                event_type: EventPointType::EndPoint,
            },
        ]
    }))
}

fn calc_intersection_point(segment_0: Segment, segment_1: Segment) -> Option<Point> {
    let cross = |p_0: Point, p_1: Point| p_0.x * p_1.y - p_0.y * p_1.x;

    let denominator = cross(segment_0[1] - segment_0[0], segment_1[1] - segment_1[0]);
    let u = cross(segment_1[0] - segment_0[0], segment_0[1] - segment_0[0]) / denominator;
    let t = cross(segment_1[0] - segment_0[0], segment_1[1] - segment_1[0]) / denominator;

    if u >= 0.0 && t >= 0.0 {
        Some(segment_1[0] + u * (segment_1[1] - segment_1[0]))
    } else {
        None
    }
}

fn update_status(
    status: &mut SkipList,
    event: EventPoint,
    segments: &[Segment],
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    match event.event_type {
        EventPointType::StartPoint => {
            let (left_neighbor, right_neighbor) =
                status.insert(event.segment_index, &segments, event.position.y);

            let mut start_point_intersecting_segments = Vec::new();

            let mut lower_intersection_points = Vec::new();

            if let Some(left_neighbor) = left_neighbor {
                if get_x_val_of_segment_at_height(&segments[left_neighbor], event.position.y)
                    == event.position.x
                {
                    start_point_intersecting_segments.push(left_neighbor);
                }

                if let Some(intersection_point) =
                    calc_intersection_point(segments[left_neighbor], segments[event.segment_index])
                {
                    lower_intersection_points.push(intersection_point);
                }
            }

            if let Some(right_neighbor) = right_neighbor {
                if get_x_val_of_segment_at_height(&segments[right_neighbor], event.position.y)
                    == event.position.x
                {
                    start_point_intersecting_segments.push(right_neighbor);
                }

                if let Some(intersection_point) =
                    calc_intersection_point(segments[right_neighbor], segments[event.segment_index])
                {
                    lower_intersection_points.push(intersection_point);
                }
            }

            let event_points: Vec<EventPoint> = lower_intersection_points
                .into_iter()
                .map(|position| EventPoint {
                    position,
                    segment_index: event.segment_index,
                    event_type: EventPointType::Intersection,
                })
                .collect();

            (
                event_points,
                if !start_point_intersecting_segments.is_empty() {
                    Some(IntersectionPoint {
                        position: event.position,
                        intersecting_segment_indices: start_point_intersecting_segments,
                    })
                } else {
                    None
                },
            )
        }
        EventPointType::Intersection => todo!(),
        EventPointType::EndPoint => todo!(),
    }
}

fn find_interesctions(segments: &[Segment]) -> Vec<IntersectionPoint> {
    let mut event_queue = segments_to_event_queue(segments);
    let mut status = SkipList::new();

    let mut intersections = Vec::new();

    while let Some(event) = event_queue.pop() {
        let (new_events, possible_intersection) = update_status(&mut status, event, segments);
        if let Some(intersection) = possible_intersection {
            intersections.push(intersection);
        }

        event_queue.extend(new_events.iter());
    }

    intersections
}

pub fn test_status() {
    let segments = vec![
        [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
        [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
        [Point::new(-10.0, -5.0), Point::new(2.0, 5.0)],
    ];

    let mut list = SkipList::new();
    for i in 0..segments.len() {
        list.insert(i, &segments, 1.0);
    }

    println!("{:?}", list.iter(0).collect::<Vec<_>>());
    println!("{:?}", list.iter(1).collect::<Vec<_>>());
    println!("{:?}", list.iter(2).collect::<Vec<_>>());
    panic!();
}

#[cfg(test)]
mod test {
    use crate::{street_graph::NodeType, street_plan::heap_to_vec, tensor_field::Point};

    use super::{
        EventPoint, EventPointType, SkipList, find_interesctions, segments_to_event_queue,
        update_status,
    };

    #[test]
    fn segments_to_event_queue_correctly_prioritizes_single_segment() {
        let segments = [[Point::new(1.0, 1.0), Point::new(2.0, -4.0)]];

        let event_queue = segments_to_event_queue(&segments);

        let event_queue_vec = event_queue.into_vec();

        let expected_event_queue_vec = vec![
            EventPoint {
                position: Point::new(1.0, 1.0),
                segment_index: 0,
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, -4.0),
                segment_index: 0,
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

        let event_queue = segments_to_event_queue(&segments);
        let event_queue_as_vec = heap_to_vec(event_queue);

        let proper_events = [
            EventPoint {
                position: Point::new(-5.0, 12.0),
                segment_index: 1,
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, 5.0),
                segment_index: 3,
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, 2.0),
                segment_index: 2,
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(1.0, 1.0),
                segment_index: 0,
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(13.0, -0.2),
                segment_index: 1,
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(2.0, -4.0),
                segment_index: 0,
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(-10.0, -5.0),
                segment_index: 3,
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(0.0, -10.0),
                segment_index: 2,
                event_type: EventPointType::EndPoint,
            },
        ];

        assert_eq!(event_queue_as_vec, proper_events);
    }

    #[test]
    fn start_points_are_correctly_inserted_into_status() {
        let segments = [
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
            [Point::new(-10.0, -5.0), Point::new(2.0, 5.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_index: i,
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert!(new_events.is_empty());
        assert!(new_intersections.is_empty());

        let status_as_vec: Vec<usize> = status
            .iter(0)
            .flat_map(|node_type| {
                if let NodeType::Value(curve_index) = node_type {
                    Some(*curve_index)
                } else {
                    None
                }
            })
            .collect();

        let expected_status_vec = vec![3, 1, 0, 2];

        assert_eq!(status_as_vec, expected_status_vec);
    }
}
