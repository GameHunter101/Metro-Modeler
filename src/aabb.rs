use crate::tensor_field::Point;

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    north: f32,
    south: f32,
    east: f32,
    west: f32,
}

impl BoundingBox {
    pub fn new(points: &[Point]) -> BoundingBox {
        let mut northmost_point = points[0];
        let mut southmost_point = points[0];
        let mut eastmost_point = points[0];
        let mut westmost_point = points[0];

        for point in points {
            if point.x > eastmost_point.x {
                eastmost_point = *point;
            }
            if point.x < westmost_point.x {
                westmost_point = *point;
            }
            if point.y > northmost_point.y {
                northmost_point = *point;
            }
            if point.y < southmost_point.y {
                southmost_point = *point;
            }
        }

        BoundingBox {
            north: northmost_point.y,
            south: southmost_point.y,
            east: eastmost_point.x,
            west: westmost_point.x,
        }
    }

    pub fn north(&self) -> f32 {
        self.north
    }

    pub fn south(&self) -> f32 {
        self.south
    }

    pub fn east(&self) -> f32 {
        self.east
    }

    pub fn west(&self) -> f32 {
        self.west
    }
}
