use nalgebra::{Matrix2, Vector2};

const GRID_SIZE: usize = 512;

type Tensor = Matrix2<f32>;
type Point = Vector2<f32>;

#[derive(Debug)]
pub struct TensorField {
    design_elements: Vec<DesignElement>,
    decay_constant: f32,
    field: Option<[[Tensor; GRID_SIZE]; GRID_SIZE]>,
}

impl TensorField {
    fn new(design_elements: Vec<DesignElement>, decay_constant: f32) -> TensorField {
        TensorField {
            design_elements,
            decay_constant,
            field: None,
        }
    }

    fn add_design_element(&mut self, design_element: DesignElement) {
        self.design_elements.push(design_element);
    }

    fn remove_design_element(&mut self, index: usize) {
        self.design_elements.remove(index);
    }

    fn sum_elements(
        design_elements: &[DesignElement],
        point: Point,
        decay_constant: f32,
    ) -> Tensor {
        design_elements
            .iter()
            .map(|element| {
                if let Some(center) = element.center() {
                    (-decay_constant * (point - center).norm_squared()).exp()
                        * element.evaluate_at_point(point)
                } else {
                    element.evaluate_at_point(point)
                }
            })
            .sum()
    }

    fn evaluate_field_at_point(&self, point: Point) -> Tensor {
        Self::sum_elements(&self.design_elements, point, self.decay_constant)
    }

    fn solve_full_field(&mut self) {
        let field: [[Tensor; GRID_SIZE]; GRID_SIZE] = (0..GRID_SIZE)
            .map(|y| {
                let arr: [Tensor; GRID_SIZE] = (0..GRID_SIZE)
                    .map(|x| self.evaluate_field_at_point(Point::new(x as f32, y as f32)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                arr
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        self.field = Some(field);
    }
}

#[derive(Debug)]
pub enum DesignElement {
    Grid {
        center: Point,
        width: f32,
        height: f32,
    },
    Radial {
        center: Point,
    },
    PolyLine {
        points: Vec<Point>,
        decay_constant: f32,
    },
}

impl DesignElement {
    pub fn evaluate_at_point(&self, point: Point) -> Tensor {
        match self {
            DesignElement::Grid { width, height, .. } => {
                let theta = (height / width).atan();
                let len = (height * height + width * width).sqrt();
                let theta_2 = 2.0 * theta;
                len * Tensor::new(theta_2.cos(), theta_2.sin(), theta_2.sin(), -theta_2.cos())
            }
            DesignElement::Radial { center } => {
                let x = point.x - center.x;
                let y = point.x - center.x;

                let y_squared = y * y;
                let x_squared = x * x;

                Tensor::new(
                    y_squared - x_squared,
                    -2.0 * x * y,
                    -2.0 * x * y,
                    -(y_squared - x_squared),
                )
            }
            DesignElement::PolyLine {
                points,
                decay_constant,
            } => {
                let lines = (0..points.len() - 1)
                    .map(|i| {
                        let dir = points[i + 1] - points[i];
                        DesignElement::Grid {
                            center: points[i],
                            width: dir.x,
                            height: dir.y,
                        }
                    })
                    .collect::<Vec<DesignElement>>();

                TensorField::sum_elements(&lines, point, *decay_constant)
            }
        }
    }

    pub fn center(&self) -> Option<Point> {
        match self {
            DesignElement::Grid { center, .. } => Some(*center),
            DesignElement::Radial { center } => Some(*center),
            DesignElement::PolyLine { .. } => None,
        }
    }
}

pub trait Eigenvectors {
    fn eigenvectors(&self) -> [Vector2<f32>; 2];
}

impl Eigenvectors for Tensor {
    fn eigenvectors(&self) -> [Vector2<f32>; 2] {
        let eigenvalues = self.eigenvalues().unwrap();
        eigenvalues
            .into_iter()
            .map(|&eigenvalue| {
                let tensor = self - eigenvalue * Tensor::identity();
                let [a, b]: [f32; 2] = tensor.row(0).into_owned().into();
                let [c, d]: [f32; 2] = tensor.row(1).into_owned().into();
                let new_b = b / a;
                let new_d = d - b / a * c;
                if new_d <= 0.000001 {
                    Vector2::new(-new_b, 1.0)
                } else {
                    Vector2::new(1.0, 1.0)
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
