use nalgebra_glm::Mat4;
use specs::prelude::*;
use specs_hierarchy::Parent as HParent;

use super::*;

use std::collections::VecDeque;
use std::io::Write;

/// Components for defining a node in a render graph
/// Useful when representing 3d models comprised of
/// several GraphicsPrimitive where we might have
/// transforms on several levels that should be concatenated

/// A child always has a parent. A root node will not have this component
#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct RenderGraphChild {
    pub parent: Entity,
}

impl Component for RenderGraphChild {
    type Storage = FlaggedStorage<Self, DenseVecStorage<Self>>;
}

impl HParent for RenderGraphChild {
    fn parent_entity(&self) -> Entity {
        self.parent
    }
}

/// A generic node, used by either roots or nodes that have children
/// A leaf node will not have this component
#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd, Component)]
#[storage(DenseVecStorage)]
pub struct RenderGraphNode {
    pub children: Vec<Entity>,
}

/// Marker node for graph roots
#[derive(Debug, Default, Clone, Eq, Ord, PartialEq, PartialOrd, Component)]
#[storage(NullStorage)]
pub struct RenderGraphRoot {}

pub fn leaf(parent: Entity) -> RenderGraphChild {
    RenderGraphChild { parent }
}

pub fn child(parent: Entity) -> RenderGraphChild {
    RenderGraphChild { parent }
}

pub fn root() -> RenderGraphRoot {
    RenderGraphRoot {}
}

pub fn node(children: Vec<Entity>) -> RenderGraphNode {
    RenderGraphNode { children }
}

/// SPECS system to concatenate model matrices
pub struct TransformPropagation;
impl TransformPropagation {
    fn propagate_transforms_rec<'a>(
        stack: &mut Vec<Mat4>,
        ent: Entity,
        rgnodes: &ReadStorage<'a, RenderGraphNode>,
        transforms: &ReadStorage<'a, Transform>,
        model_matrices: &mut WriteStorage<'a, ModelMatrix>,
    ) {
        let transform: Mat4 = transforms
            .get(ent)
            .copied()
            .unwrap_or(glm::identity::<f32, U4>().into())
            .into();
        let transform = stack.last().unwrap() * transform;

        stack.push(transform);
        model_matrices.insert(ent, ModelMatrix(transform)).unwrap();

        if let Some(node) = rgnodes.get(ent) {
            for child in node.children.iter() {
                TransformPropagation::propagate_transforms_rec(
                    stack,
                    *child,
                    rgnodes,
                    transforms,
                    model_matrices,
                );
            }
        }
        stack.pop();
    }
}

// TODO: Can we use a flagged storage for this?
// E.g. only run propagation for any of the roots
// for which it or children has been changed?
impl<'a> System<'a> for TransformPropagation {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, RenderGraphRoot>,
        ReadStorage<'a, RenderGraphNode>,
        ReadStorage<'a, Transform>,
        WriteStorage<'a, ModelMatrix>,
    );

    fn run(
        &mut self,
        (entities, roots, rgnodes, transforms, mut model_matrices): Self::SystemData,
    ) {
        for (ent, _root) in (&entities, &roots).join() {
            let mut stack: Vec<Mat4> = Vec::new();

            let transform: Mat4 = transforms
                .get(ent)
                .copied()
                .unwrap_or(glm::identity::<f32, U4>().into())
                .into();
            stack.push(transform);

            if let Ok(entry) = model_matrices.entry(ent) {
                // Root node, no need to multiply
                entry.or_insert(ModelMatrix(stack[0]));
            }

            if let Some(node) = rgnodes.get(ent) {
                for child in node.children.iter() {
                    TransformPropagation::propagate_transforms_rec(
                        &mut stack,
                        *child,
                        &rgnodes,
                        &transforms,
                        &mut model_matrices,
                    );
                }
            }
        }
    }
}

pub fn breadth_first(world: &World, root: Entity, mut visit_node: impl FnMut(Entity)) {
    let nodes_storage = world.read_storage::<RenderGraphNode>();

    let mut queue = VecDeque::new();
    queue.push_back(root);

    while !queue.is_empty() {
        let ent = queue.pop_front().unwrap();
        visit_node(ent);

        if let Some(node) = nodes_storage.get(ent) {
            for c in node.children.iter() {
                queue.push_back(*c);
            }
        }
    }
}

pub fn depth_first(world: &World, root: Entity, mut visit_node: impl FnMut(Entity)) {
    let nodes_storage = world.read_storage::<RenderGraphNode>();

    let mut stack = Vec::new();
    stack.push(root);

    while !stack.is_empty() {
        let ent = stack.pop().unwrap();
        visit_node(ent);

        if let Some(node) = nodes_storage.get(ent) {
            for c in node.children.iter() {
                stack.push(*c);
            }
        }
    }
}

fn e2str(e: Entity) -> String {
    format!(
        "({}, {}, {})",
        e.id(),
        e.gen().id(),
        if e.gen().is_alive() { "Live" } else { "Dead " }
    )
}

fn mat2str(m: impl Into<glm::Mat4>) -> String {
    let m: Mat4 = m.into();
    format!(
        "{} {} {} {}\\n{} {} {} {}\\n{} {} {} {}\\n{} {} {} {}\\n",
        m.index((0, 0)),
        m.index((0, 1)),
        m.index((0, 2)),
        m.index((0, 3)),
        m.index((1, 0)),
        m.index((1, 1)),
        m.index((1, 2)),
        m.index((1, 3)),
        m.index((2, 0)),
        m.index((2, 1)),
        m.index((2, 2)),
        m.index((2, 3)),
        m.index((3, 0)),
        m.index((3, 1)),
        m.index((3, 2)),
        m.index((3, 3))
    )
}

fn mat2pos(m: impl Into<glm::Mat4>) -> String {
    let m: Mat4 = m.into();
    format!(
        "{} {} {} {}\\n",
        m.index((0, 3)),
        m.index((1, 3)),
        m.index((2, 3)),
        m.index((3, 3))
    )
}

fn node_to_dot<W: Write>(world: &World, e: Entity, w: &mut W, prefix: &str) {
    let nodes_storage = world.read_storage::<RenderGraphNode>();
    let node_name = format!("\"{} {}\"", prefix, e2str(e));
    let trns = world.read_storage::<Transform>();
    let mats = world.read_storage::<ModelMatrix>();

    let (trm_str, mat_str) = match (trns.get(e), mats.get(e)) {
        (None, None) => return,
        (Some(m), None) => (mat2pos(*m), String::from("")),
        (None, Some(m)) => (String::from(""), mat2pos(*m)),
        (Some(m), Some(n)) => (mat2pos(*m), mat2pos(*n)),
    };

    write!(
        w,
        "  {} [label=\"{}\\n---------\\n{}\"]\n",
        node_name, trm_str, mat_str
    );

    if let Some(node) = nodes_storage.get(e) {
        for child in node.children.iter() {
            if mats.get(*child).is_some() || trns.get(*child).is_some() {
                write!(w, "  {} -> \"node {}\"\n", node_name, e2str(*child));
                node_to_dot(world, *child, w, "node");
            }
        }
    }
}

pub fn print_graph_to_dot(world: &World, roots: Vec<Entity>, mut w: impl Write) {
    write!(w, "digraph {{\n");

    for root in roots.iter() {
        write!(w, "// ===== New subgraph =====\n");
        node_to_dot(world, *root, &mut w, "root");
    }

    write!(w, "}}\n");
}

pub fn print_graph(world: &World, mut w: impl Write) {
    let entities = world.read_resource::<specs::world::EntitiesRes>();
    let roots_storage = world.read_storage::<RenderGraphRoot>();

    write!(w, "digraph {{\n");
    for (root, _) in (&entities, &roots_storage).join() {
        write!(w, "// ===== New subgraph =====\n");
        node_to_dot(world, root, &mut w, "root");
    }
    write!(w, "}}\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Component)]
    #[storage(DenseVecStorage)]
    struct ID(usize);

    fn setup_world() -> World {
        let mut world = World::new();

        world.register::<render_graph::RenderGraphNode>();
        world.register::<render_graph::RenderGraphRoot>();
        world.register::<render_graph::RenderGraphChild>();
        world.register::<ID>();

        world
    }

    fn leaf_with_id(w: &mut World, id: usize) -> Entity {
        w.create_entity().with(ID(id)).build()
    }

    fn node_with_id(w: &mut World, children: Vec<Entity>, id: usize) -> Entity {
        w.create_entity().with(node(children)).with(ID(id)).build()
    }

    fn setup_graph(mut w: &mut World) -> Entity {
        let children = vec![leaf_with_id(&mut w, 5), leaf_with_id(&mut w, 6)];
        let node2 = node_with_id(&mut w, children, 2);
        let node3 = leaf_with_id(&mut w, 3);

        let node7 = leaf_with_id(&mut w, 7);
        let node4 = node_with_id(&mut w, vec![node7], 4);

        let root = w
            .create_entity()
            .with(root())
            .with(node(vec![node2, node3, node4]))
            .with(ID(1))
            .build();

        root
    }

    #[test]
    fn breadth_first_traversal() {
        let mut w = setup_world();
        let root = setup_graph(&mut w);

        let mut order = Vec::new();

        let visit_node = |x: Entity| {
            let ids = w.read_storage::<ID>();
            order.push(ids.get(x).expect("No id!").0);
        };

        render_graph::breadth_first(&w, root, visit_node);
        assert_eq!(order, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn depth_first_traversal() {
        let mut w = setup_world();
        let root = setup_graph(&mut w);

        let mut order = Vec::new();

        let visit_node = |x: Entity| {
            let ids = w.read_storage::<ID>();
            order.push(ids.get(x).expect("No id!").0);
        };

        render_graph::depth_first(&w, root, visit_node);
        assert_eq!(order, vec![1, 4, 7, 3, 2, 6, 5]);
    }
}
