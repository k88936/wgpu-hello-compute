struct Particle {
    position: vec3f,
    v: vec3f,
    C: mat3x3f,
}
struct Cell {
    vx: i32,
    vy: i32,
    vz: i32,
    mass: i32,
}

struct AtomCell {
    vx: atomic<i32>,
    vy: atomic<i32>,
    vz: atomic<i32>,
    mass: atomic<i32>,
}

struct PosVel {
    position: vec3f,
    density: f32,
    v: vec3f,
}
