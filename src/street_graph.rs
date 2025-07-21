use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ptr::NonNull;

use crate::tensor_field::Point;

use rand::prelude::*;

struct IntersectionPoint {
    position: Point,
    intersecting_line_indices: Vec<usize>,
}

#[derive(PartialEq, PartialOrd)]
struct EventPoint {
    position: Point,
    curve_index: usize,
    event_type: EventPointType,
}

#[derive(PartialEq, PartialOrd, Eq)]
enum EventPointType {
    StartPoint,
    Intersection,
    EndPoint,
}

impl Eq for EventPoint {}

impl Ord for EventPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let y_ord = self
            .position
            .y
            .partial_cmp(&other.position.y)
            .unwrap_or(std::cmp::Ordering::Equal);
        if y_ord.is_eq() {
            self.position
                .x
                .partial_cmp(&other.position.x)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            y_ord
        }
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
    ) {
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

            for (level, node) in prev_ptrs.into_iter().enumerate() {
                (&mut (*node.as_ptr()).next_ptrs)[level] = new_node;
            }

            let promotion_count = next_ptrs.len();

            for (level, node) in next_ptrs.into_iter().enumerate() {
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
        }
    }
}

#[derive(Debug, Clone)]
enum NodeType {
    Start,
    Value(usize),
    End,
}

impl NodeType {
    fn get_x_val_of_segment_at_height(segment: &Segment, height: f32) -> f32 {
        segment[0].x
            + ((height - segment[0].y) / (segment[1] - segment[0]).y) * (segment[1] - segment[0]).x
    }
    fn cmp(&self, other: &NodeType, segments: &[Segment], height: f32) -> Ordering {
        match self {
            NodeType::Start => Ordering::Less,
            NodeType::Value(lhs) => match other {
                NodeType::Start => Ordering::Greater,
                NodeType::Value(rhs) => {
                    if rhs == lhs {
                        Ordering::Equal
                    } else {
                        Self::get_x_val_of_segment_at_height(&segments[*lhs], height)
                            .partial_cmp(&Self::get_x_val_of_segment_at_height(
                                &segments[*rhs],
                                height,
                            ))
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

    fn insert(&mut self, element: usize, segments: &[Segment], height: f32) {
        unsafe {
            let (traverse_node, traverse_path) = Node::traverse_level(
                self.nodes,
                (*self.nodes.as_ptr()).next_ptrs.len() - 1,
                element.clone(),
                segments,
                height,
            );

            Node::append(
                self.nodes,
                self.end,
                traverse_node,
                element,
                traverse_path,
                &mut self.rng,
            );

            self.len += 1;
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

    fn iter(&self) -> Iter {
        unsafe {
            Iter {
                next: Some(self.nodes.as_ref()),
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
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a NodeType;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            self.next.take().map(|node| {
                self.next = node.next_ptrs.get(0).map(|ptr| ptr.as_ref());
                &node.node_type
            })
        }
    }
}

fn find_interesctions(lines: &[[Point; 2]]) -> Vec<IntersectionPoint> {
    let mut event_queue =
        BinaryHeap::from_iter(lines.iter().enumerate().flat_map(|(i, &[start, end])| {
            let (start, end) = if start.y > end.y {
                (start, end)
            } else {
                (end, start)
            };
            [
                EventPoint {
                    position: start,
                    curve_index: i,
                    event_type: EventPointType::StartPoint,
                },
                EventPoint {
                    position: end,
                    curve_index: i,
                    event_type: EventPointType::EndPoint,
                },
            ]
        }));

    while let Some(event) = event_queue.pop() {
        match event.event_type {
            EventPointType::StartPoint => todo!(),
            EventPointType::Intersection => todo!(),
            EventPointType::EndPoint => todo!(),
        }
    }

    todo!()
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
        list.insert(i, &segments, 0.9);
    }

    println!("{:?}", list.iter().collect::<Vec<_>>());
    panic!();
}
