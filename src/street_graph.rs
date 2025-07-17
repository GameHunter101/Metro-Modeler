use std::collections::BinaryHeap;

use crate::tensor_field::Point;

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

fn get_x_val_of_segment_at_height(segment: &Segment, height: f32) -> f32 {
    segment[0].x
        + ((height - segment[0].y) / (segment[1] - segment[0]).y) * (segment[1] - segment[0]).x
}

#[derive(Clone)]
struct Status {
    curve_index: usize,
    left: Option<Box<Status>>,
    right: Option<Box<Status>>,
}

type Segment = [Point; 2];

impl Status {
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    fn height(&self) -> i32 {
        if self.is_leaf() {
            1
        } else {
            let left_height = if let Some(left_tree) = &self.left {
                left_tree.height()
            } else {
                0
            };

            let right_height = if let Some(right_tree) = &self.right {
                right_tree.height()
            } else {
                0
            };

            1 + left_height.max(right_height)
        }
    }

    fn balance(&self) -> i32 {
        (if let Some(right_tree) = &self.right {
            right_tree.height()
        } else {
            0
        }) - (if let Some(left_tree) = &self.left {
            left_tree.height()
        } else {
            0
        })
    }

    fn rebalance_subtree(mut self: Box<Self>) -> Box<Self> {
        if self.is_leaf() {
            return self;
        }

        let balance = self.balance();

        if balance.abs() == 2 {
            let mut left_child = true;
            let mut left_grandchild = true;

            let imbalanced_child = if balance == -2 {
                self.left.as_ref().unwrap()
            } else {
                left_child = false;
                self.right.as_ref().unwrap()
            };

            if imbalanced_child.balance() >= 0 {
                left_grandchild = false;
            }

            if left_child && left_grandchild {
                self.rotate_right()
            } else if !left_child && !left_grandchild {
                self.rotate_left()
            } else if left_child && !left_grandchild {
                self.left = Some(self.left.unwrap().rotate_left());
                self.rotate_right()
            } else {
                self.right = Some(self.right.unwrap().rotate_right());
                self.rotate_left()
            }
        } else {
            self
        }
    }

    fn rebalance(mut self: Box<Self>) -> Box<Self> {
        if self.is_leaf() {
            return self;
        }

        self.left = if let Some(left_node) = self.left {
            Some(left_node.rebalance())
        } else {
            None
        };

        self.right = if let Some(right_node) = self.right {
            Some(right_node.rebalance())
        } else {
            None
        };

        self.rebalance_subtree()
    }

    fn rotate_left(self: Box<Self>) -> Box<Self> {
        let mut new_left = self;
        let mut target = new_left.right.unwrap();

        new_left.right = target.left;
        target.left = Some(new_left);
        target
    }

    fn rotate_right(self: Box<Self>) -> Box<Self> {
        let mut new_right = self;
        let mut target = new_right.left.unwrap();

        new_right.left = target.right;
        target.right = Some(new_right);
        target
    }

    fn insert_helper(
        self: &mut Box<Self>,
        curve_index: usize,
        segment_start_position: Point,
        segments: &[Segment],
    ) {
        let current_x_val =
            get_x_val_of_segment_at_height(&segments[self.curve_index], segment_start_position.y);

        let new_node = Box::new(Status {
            curve_index,
            left: None,
            right: None,
        });

        if current_x_val < segment_start_position.x {
            let right_ref = &mut self.right;
            match right_ref {
                Some(right_node) => right_node.insert_helper(curve_index, segment_start_position, segments),
                None => {
                    *right_ref = Some(new_node);
                }
            }
        } else {
            let left_ref = &mut self.left;
            match left_ref {
                Some(left_node) => left_node.insert_helper(curve_index, segment_start_position, segments),
                None => {
                    *left_ref = Some(new_node);
                }
            }
        };
    }

    fn insert(
        mut self: Box<Self>,
        curve_index: usize,
        segment_start_position: Point,
        segments: &[Segment],
    ) -> Box<Self> {
        self.insert_helper(curve_index, segment_start_position, segments);

        self.rebalance_subtree()
    }

    fn format(&self, indent: String, is_final: bool, post: &str) -> String {
        let mut output = format!(
            "{} -+ {}{post}, b:{}\n",
            indent.clone(),
            self.curve_index,
            self.balance()
        );
        let indent = indent + if is_final { "   " } else { "|  " };
        if let Some(left_node) = &self.left {
            output += &left_node.format(indent.clone(), self.right.is_none(), " # L");
        }
        if let Some(right_node) = &self.right {
            output += &right_node.format(indent, true, " # R");
        }

        output
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format(String::new(), true, ""))
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
    /* let mut status = Box::new(Status {
        curve_index: 55,
        left: Some(Box::new(Status {
            curve_index: 41,
            left: None,
            right: Some(Box::new(Status {
                curve_index: 49,
                left: None,
                right: None,
            })),
        })),
        right: Some(Box::new(Status {
            curve_index: 72,
            left: Some(Box::new(Status {
                curve_index: 64,
                left: None,
                right: Some(Box::new(Status {
                    curve_index: 67,
                    left: Some(Box::new(Status {
                        curve_index: 65,
                        left: None,
                        right: None,
                    })),
                    right: None,
                })),
            })),
            right: Some(Box::new(Status {
                curve_index: 81,
                left: None,
                right: Some(Box::new(Status {
                    curve_index: 82,
                    left: None,
                    right: None,
                })),
            })),
        })),
    }); */

    let segments = vec![
        [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
        [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
        [Point::new(-10.0, -5.0), Point::new(2.0, 5.0)],
    ];

    let mut status = Box::new(Status{curve_index: 0, left: None, right: None});

    for i in 1..segments.len() {
        let start_pos = if segments[i][0].y > segments[i][1].y {segments[i][0]} else {segments[i][1]};
        status = status.insert(i, start_pos, &segments);
    }

    println!("{status}");
    println!("----------");
    status = status.rebalance();
    println!("{status}");
    /* println!("----------");
    let r = status.right.as_mut().unwrap();
    let r_l = r.left.as_mut().unwrap();
    r_l.left = None;
    println!("{status}");
    println!("----------");
    status = status.rebalance();
    println!("{status}"); */
    panic!();
}
