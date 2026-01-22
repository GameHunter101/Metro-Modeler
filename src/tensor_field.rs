use nalgebra::{Matrix2, Vector2};

pub const GRID_SIZE: u32 = 512;

pub type Tensor = Matrix2<f32>;
pub type Point = Vector2<f32>;

#[allow(unused)]
#[derive(Debug)]
pub struct TensorField {
    grid: Box<[Tensor]>,
    design_elements: Vec<DesignElement>,
    decay_constant: f32,
}

#[allow(unused)]
impl TensorField {
    pub fn new(design_elements: Vec<DesignElement>, decay_constant: f32) -> TensorField {
        let mut grid = vec![Tensor::default(); (GRID_SIZE * GRID_SIZE) as usize].into_boxed_slice();

        let design_elements_ref = &design_elements;
        async_scoped::TokioScope::scope_and_block(|scope| {
            for (row, chunk) in grid.chunks_mut(GRID_SIZE as usize).enumerate() {
                scope.spawn(async move {
                    for col in 0..GRID_SIZE as usize {
                        chunk[col] = Self::calculate_smoothed_field_at_point(
                            Point::new(col as f32, row as f32),
                            design_elements_ref,
                            decay_constant,
                        )
                    }
                });
            }
        });
        TensorField {
            grid,
            design_elements,
            decay_constant,
        }
    }

    pub fn add_design_element(&mut self, design_element: DesignElement) {
        self.design_elements.push(design_element);
    }

    pub fn remove_design_element(&mut self, index: usize) {
        self.design_elements.remove(index);
    }

    fn sum_elements(
        point: Point,
        design_elements: &[DesignElement],
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

    fn calculate_field_at_point(
        point: Point,
        design_elements: &[DesignElement],
        decay_constant: f32,
    ) -> Tensor {
        Self::sum_elements(point, design_elements, decay_constant)
    }

    fn calculate_smoothed_field_at_point(
        point: Point,
        design_elements: &[DesignElement],
        decay_constant: f32,
    ) -> Tensor {
        let mut sum = Self::calculate_field_at_point(point, design_elements, decay_constant);
        let mut count = 1;
        let mut neighbors = Vec::new();
        if point.x >= 1.0 {
            neighbors.push(Point::new(-1.0, 0.0));
            count += 1;
        } else if point.x <= GRID_SIZE as f32 - 1.0 {
            neighbors.push(Point::new(1.0, 0.0));
            count += 1;
        }

        if point.y >= 1.0 {
            neighbors.push(Point::new(0.0, -1.0));
            count += 1;
        } else if point.y <= GRID_SIZE as f32 - 1.0 {
            neighbors.push(Point::new(0.0, 1.0));
            count += 1;
        }

        sum += neighbors
            .into_iter()
            .map(|vec| Self::calculate_field_at_point(point + vec, design_elements, decay_constant))
            .sum::<Tensor>();

        sum / count as f32
    }

    pub fn evaluate_smoothed_field_at_point(&self, point: Point) -> Tensor {
        let x_floor = (point.x as usize).min(GRID_SIZE as usize - 1);
        let x_ceil = (point.x.ceil() as usize).min(GRID_SIZE as usize - 1);
        let y_floor = (point.y as usize).min(GRID_SIZE as usize - 1);
        let y_ceil = (point.y.ceil() as usize).min(GRID_SIZE as usize - 1);

        let top_left = self.grid[y_ceil * GRID_SIZE as usize + x_floor];
        let top_right = self.grid[y_ceil * GRID_SIZE as usize + x_ceil];

        let top_lerp = Self::lerp(top_left, top_right, point.x.fract());

        let bottom_left = self.grid[y_floor * GRID_SIZE as usize + x_floor];
        let bottom_right = self.grid[y_floor * GRID_SIZE as usize + x_ceil];

        let bottom_lerp = Self::lerp(bottom_left, bottom_right, point.x.fract());

        Self::lerp(bottom_lerp, top_lerp, point.y.fract())
    }

    fn lerp(a: Tensor, b: Tensor, t: f32) -> Tensor {
        a * (1.0 - t) + b * t
    }

    pub fn design_elements(&self) -> &[DesignElement] {
        &self.design_elements
    }

    pub fn decay_constant(&self) -> f32 {
        self.decay_constant
    }
}

#[derive(Debug, Clone)]
pub enum DesignElement {
    Grid {
        center: Point,
        theta: f32,
        length: f32,
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
            DesignElement::Grid { theta, length, .. } => {
                let theta_2 = 2.0 * theta;
                *length * Tensor::new(theta_2.cos(), theta_2.sin(), theta_2.sin(), -theta_2.cos())
            }
            DesignElement::Radial { center } => {
                let x = point.x - center.x;
                let y = point.y - center.y;

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
                        let dir = (points[i + 1] - points[i]).normalize();
                        DesignElement::Grid {
                            center: points[i],
                            theta: dir.y.atan2(dir.x) + std::f32::consts::FRAC_PI_2,
                            length: 1.0,
                        }
                    })
                    .collect::<Vec<DesignElement>>();

                TensorField::sum_elements(point, &lines, *decay_constant)
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

#[derive(Debug, Default, Clone, Copy)]
pub struct Eigenvectors {
    pub major: Vector2<f32>,
    pub minor: Vector2<f32>,
}

pub trait EvalEigenvectors {
    fn eigenvectors(&self) -> Eigenvectors;
}

impl EvalEigenvectors for Tensor {
    fn eigenvectors(&self) -> Eigenvectors {
        let eigenvalues = self.eigenvalues().unwrap();
        let calc_eigenvector = |eigenvalue: f32| {
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
        };
        let mut vectors: [Vector2<f32>; 2] = [
            calc_eigenvector(eigenvalues[0]),
            calc_eigenvector(eigenvalues[1]),
        ];
        let up = nalgebra::Vector3::z();
        if f32::is_infinite(vectors[0].norm()) {
            vectors[0] = up
                .cross(&nalgebra::Vector3::new(vectors[1].x, vectors[1].y, 0.0))
                .xy();
        } else if f32::is_infinite(vectors[1].norm()) {
            vectors[1] = up
                .cross(&nalgebra::Vector3::new(vectors[0].x, vectors[0].y, 0.0))
                .xy();
        }
        if eigenvalues[0] > 0.0 {
            Eigenvectors {
                major: vectors[0],
                minor: vectors[1],
            }
        } else {
            Eigenvectors {
                major: vectors[1],
                minor: vectors[0],
            }
        }
    }
}
