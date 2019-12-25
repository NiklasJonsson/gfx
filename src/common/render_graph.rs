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
            .unwrap_or_else(|| glm::identity::<f32, U4>().into())
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

enum PathDirection {
    RootToNode(usize),
    NodeToRoot(usize),
}

impl PathDirection {
    // TODO: Change to operator++
    fn next(&mut self) {
        use PathDirection::*;
        *self = match *self {
            RootToNode(idx) => RootToNode(idx + 1),
            NodeToRoot(idx) => NodeToRoot(idx - 1),
        }
    }

    fn done(&self, path_len: usize) -> bool {
        use PathDirection::*;
        match self {
            RootToNode(idx) => *idx == path_len,
            NodeToRoot(idx) => *idx == 0,
        }
    }

    // TODO: Change to deref
    fn idx(&self) -> usize {
        use PathDirection::*;
        match self {
            RootToNode(idx) => *idx,
            NodeToRoot(idx) => *idx,
        }
    }
}

struct PathWalker {
    path: Vec<Entity>,
    dir: PathDirection,
}

impl Iterator for PathWalker {
    type Item = Entity;
    fn next(&mut self) -> Option<Self::Item> {
        if self.dir.done(self.path.len()) {
            return None;
        }

        let ent = self.path[self.dir.idx()];
        self.dir.next();

        Some(ent)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.path.len();
        (l, Some(l))
    }
}

impl ExactSizeIterator for PathWalker {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

fn get_root_path(w: &World, node: Entity) -> Vec<Entity> {
    let children = w.read_storage::<RenderGraphChild>();

    let mut path = Vec::with_capacity(8);
    let mut cur = node;
    loop {
        path.push(cur);
        if let Some(child) = children.get(cur) {
            cur = child.parent;
        } else {
            break;
        }
    }

    assert!(w.read_storage::<RenderGraphRoot>().get(cur).is_some());
    path.reverse();

    path
}

/// Returns an iterator that walks the path from the root to the node
pub fn root_to_node_path(world: &World, node: Entity) -> impl ExactSizeIterator<Item = Entity> {
    let path = get_root_path(world, node);
    PathWalker {
        path,
        dir: PathDirection::RootToNode(0),
    }
}

/// Returns an iterator that walks the path from the root to the node
pub fn node_to_root_path(world: &World, node: Entity) -> impl ExactSizeIterator<Item = Entity> {
    let path = get_root_path(world, node);
    let len = path.len();
    PathWalker {
        path,
        dir: PathDirection::NodeToRoot(len - 1),
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

fn node_to_dot<W: Write>(world: &World, e: Entity, w: &mut W, prefix: &str) -> std::io::Result<()> {
    let nodes_storage = world.read_storage::<RenderGraphNode>();
    let node_name = format!("\"{} {}\"", prefix, e2str(e));
    let trns = world.read_storage::<Transform>();
    let mats = world.read_storage::<ModelMatrix>();

    let (trm_str, mat_str) = match (trns.get(e), mats.get(e)) {
        (None, None) => return Ok(()),
        (Some(m), None) => (mat2pos(*m), String::from("")),
        (None, Some(m)) => (String::from(""), mat2pos(*m)),
        (Some(m), Some(n)) => (mat2pos(*m), mat2pos(*n)),
    };

    writeln!(
        w,
        "  {} [label=\"{}\\n---------\\n{}\"]",
        node_name, trm_str, mat_str
    )?;

    if let Some(node) = nodes_storage.get(e) {
        for child in node.children.iter() {
            if mats.get(*child).is_some() || trns.get(*child).is_some() {
                writeln!(w, "  {} -> \"node {}\"", node_name, e2str(*child))?;
                node_to_dot(world, *child, w, "node")?;
            }
        }
    }

    Ok(())
}

pub fn print_graph_to_dot(
    world: &World,
    roots: Vec<Entity>,
    mut w: impl Write,
) -> std::io::Result<()> {
    writeln!(w, "digraph {{")?;

    for root in roots.iter() {
        writeln!(w, "// ===== New subgraph =====")?;
        node_to_dot(world, *root, &mut w, "root")?;
    }

    writeln!(w, "}}")
}

pub fn print_graph(world: &World, mut w: impl Write) -> std::io::Result<()> {
    let entities = world.read_resource::<specs::world::EntitiesRes>();
    let roots_storage = world.read_storage::<RenderGraphRoot>();

    writeln!(w, "digraph {{")?;
    for (root, _) in (&entities, &roots_storage).join() {
        writeln!(w, "// ===== New subgraph =====")?;
        node_to_dot(world, root, &mut w, "root")?;
    }
    writeln!(w, "}}")
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

    fn add_parent_for(w: &mut World, child: Entity, parent: Entity) {
        let mut children = w.write_storage::<RenderGraphChild>();
        children
            .insert(child, render_graph::child(parent))
            .expect("Failed!");
    }

    //      1
    //     /|\
    //    2 3 4
    //   /|   |
    //  5 6   7
    //
    fn setup_graph(w: &mut World) -> Entity {
        let node5 = leaf_with_id(w, 5);
        let node6 = leaf_with_id(w, 6);
        let node2 = node_with_id(w, vec![node5, node6], 2);
        let node3 = leaf_with_id(w, 3);

        let node7 = leaf_with_id(w, 7);
        let node4 = node_with_id(w, vec![node7], 4);

        let root = w
            .create_entity()
            .with(root())
            .with(node(vec![node2, node3, node4]))
            .with(ID(1))
            .build();

        add_parent_for(w, node2, root);
        add_parent_for(w, node3, root);
        add_parent_for(w, node4, root);
        add_parent_for(w, node5, node2);
        add_parent_for(w, node6, node2);
        add_parent_for(w, node7, node4);

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

    fn verify_it(
        id2ent: &Vec<Entity>,
        it: impl Iterator<Item = Entity>,
        order: Vec<usize>,
    ) -> bool {
        let expected = order.into_iter().map(|i| id2ent[i]).collect::<Vec<_>>();
        it.zip(expected.iter())
            .fold(true, |acc, (ent, &exp)| acc || (ent == exp))
    }

    #[test]
    fn node_to_root_path() {
        let mut w = setup_world();
        let root = setup_graph(&mut w);

        let ids = w.read_storage::<ID>();
        let entities = w.read_resource::<specs::world::EntitiesRes>();

        let joined = (&entities, &ids).join();
        let mut id2ent: Vec<Entity> = Vec::new();
        // init to something to be able to insert at pos
        for _ in 0..9 {
            id2ent.push(root);
        }
        for (ent, ID(id)) in joined {
            id2ent[*id] = ent;
        }

        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[2]),
            vec![2, 1]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[5]),
            vec![5, 2, 1]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[3]),
            vec![3, 1]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[6]),
            vec![6, 2, 1]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[4]),
            vec![4, 1]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[7]),
            vec![7, 4, 1]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::node_to_root_path(&w, id2ent[1]),
            vec![1]
        ));
    }

    #[test]
    fn root_to_node_path() {
        let mut w = setup_world();
        let root = setup_graph(&mut w);

        let ids = w.read_storage::<ID>();
        let entities = w.read_resource::<specs::world::EntitiesRes>();

        let joined = (&entities, &ids).join();
        let mut id2ent: Vec<Entity> = Vec::new();
        // init to something to be able to insert at pos
        for _ in 0..9 {
            id2ent.push(root);
        }
        for (ent, ID(id)) in joined {
            id2ent[*id] = ent;
        }

        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[2]),
            vec![1, 2]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[5]),
            vec![1, 2, 5]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[3]),
            vec![1, 3]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[6]),
            vec![1, 2, 6]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[4]),
            vec![1, 4]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[7]),
            vec![1, 4, 7]
        ));
        assert!(verify_it(
            &id2ent,
            render_graph::root_to_node_path(&w, id2ent[1]),
            vec![1]
        ));
    }

    #[test]
    fn size_hint_len() {
        let mut w = setup_world();
        let root = setup_graph(&mut w);

        let ids = w.read_storage::<ID>();
        let entities = w.read_resource::<specs::world::EntitiesRes>();

        let joined = (&entities, &ids).join();
        let mut id2ent: Vec<Entity> = Vec::new();
        // init to something to be able to insert at pos
        for _ in 0..9 {
            id2ent.push(root);
        }

        let expected = vec![0, 1, 2, 2, 2, 3, 3, 3];
        for (ent, ID(id)) in joined {
            let e = expected[*id];
            assert_eq!(render_graph::root_to_node_path(&w, ent).len(), e);
            assert_eq!(render_graph::root_to_node_path(&w, ent).size_hint().0, e);
            assert_eq!(
                render_graph::root_to_node_path(&w, ent).size_hint().1,
                Some(e)
            );
            assert_eq!(render_graph::node_to_root_path(&w, ent).len(), e);
            assert_eq!(render_graph::node_to_root_path(&w, ent).size_hint().0, e);
            assert_eq!(
                render_graph::node_to_root_path(&w, ent).size_hint().1,
                Some(e)
            );
        }
    }
}