use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, U4};
use specs::prelude::*;
use specs::Component;
use specs_hierarchy::Parent as HParent;

use std::collections::VecDeque;
use std::io::Write;

use specs::storage::StorageEntry;

use crate::math::{ModelMatrix, Transform};

/// Components for defining a node in a render graph
/// Useful when representing 3d models comprised of
/// several Mesh where we might have
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

pub fn attach_leaf(world: &mut World, parent: Entity) -> Entity {
    let child_ent = world.create_entity().with(child(parent)).build();

    let mut node_storage = world.write_storage::<RenderGraphNode>();

    let entry = node_storage.entry(parent).expect("Failed to get entry!");
    match entry {
        StorageEntry::Occupied(mut entry) => entry.get_mut().children.push(child_ent),
        StorageEntry::Vacant(entry) => {
            entry.insert(node(vec![child_ent]));
        }
    };

    child_ent
}

pub const TRANSFORM_PROPAGATION_SYSTEM_ID: &str = "transform_propagation";

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
                .unwrap_or_else(|| glm::identity::<f32, U4>().into())
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

#[derive(Component)]
#[storage(HashMapStorage)]
pub struct RenderedBoundingBox(Entity);

/* TODO: TREKANTEN
pub const RENDERED_BOUNDING_BOXES_SYSTEM_ID: &str = "rendered_bounding_boxes";
/// SPECS system to generate bounding boxes for rendering
/// Works per root node in the scene graph
pub struct RenderedBoundingBoxes;
impl<'a> System<'a> for RenderedBoundingBoxes {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, RenderGraphRoot>,
        ReadStorage<'a, ModelMatrix>,
        WriteStorage<'a, Mesh>,
        WriteStorage<'a, Material>,
        WriteStorage<'a, RenderGraphNode>,
        WriteStorage<'a, RenderGraphChild>,
        WriteStorage<'a, RenderedBoundingBox>,
        WriteStorage<'a, crate::render::Renderable>,
        Read<'a, RenderSettings>,
    );

    fn run(
        &mut self,
        (
            entities,
            roots,
            matrices,
            mut meshes,
            mut materials,
            mut rgnodes,
            mut rgchildren,
            mut rbbs,
            mut renderables,
            settings,
        ): Self::SystemData,
    ) {
        let remove_rbbs = !settings.render_bounding_box;

        if remove_rbbs {
            let mut to_remove = Vec::new();
            for (root_ent, bb) in (&entities, &rbbs).join() {
                let bb_node = bb.0;
                meshes.remove(bb_node);
                renderables.remove(bb_node);
                to_remove.push(root_ent);
            }

            for ent in to_remove {
                rbbs.remove(ent);
            }

            return;
        }

        for (root_ent, _root) in (&entities, &roots).join() {
            let mut biggest_bounding_box = BoundingBox::default();
            let update_biggest = |ent: Entity| {
                if let Some(new) = meshes.get(ent) {
                    let m = *matrices.get(ent).unwrap();
                    if let Some(bounding_box) = &new.bounding_box {
                        let min = m * bounding_box.min;
                        let max = m * bounding_box.max;
                        let new = BoundingBox { min, max };
                        biggest_bounding_box.combine_with(&new);
                    }
                }
            };

            transform_graph::breadth_first_sys_mut(&rgnodes, root_ent, update_biggest);
            let child = entities.create();
            rgnodes
                .get_mut(root_ent)
                .expect("Only a single root node")
                .children
                .push(child);
            rgchildren
                .insert(child, transform_graph::child(root_ent))
                .expect("Can't add child->parent");

            rbbs.insert(root_ent, RenderedBoundingBox(child))
                .expect("Can't add rbb");

            let (vertex_data, indices) = biggest_bounding_box.to_vertices_and_indices();
            let ty = MeshType::Line { indices };
            let material = Material {
                data: trekanten::material::MaterialData::Color {
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                compilation_mode: ShaderUse::PreCompiled,
            };
            let mesh = Mesh {
                ty,
                vertex_data,
                bounding_box: None,
            };
            meshes
                .insert(child, mesh)
                .expect("Unable to insert bb mesh");
            materials
                .insert(child, material)
                .expect("Unable to add bb material");
        }
    }
}
*/

fn breadth_first_sys_mut<'a>(
    nodes_storage: &WriteStorage<'a, RenderGraphNode>,
    root: Entity,
    mut visit_node: impl FnMut(Entity),
) {
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

fn breadth_first_sys<'a>(
    nodes_storage: &ReadStorage<'a, RenderGraphNode>,
    root: Entity,
    mut visit_node: impl FnMut(Entity),
) {
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

pub fn breadth_first(world: &World, root: Entity, visit_node: impl FnMut(Entity)) {
    let nodes_storage = world.read_storage::<RenderGraphNode>();
    breadth_first_sys(&nodes_storage, root, visit_node);
}

/// Traverse the graph starting at root and apply the function to each node.
/// Use this when the order does not matter, only that each node is visited once.
pub fn map(world: &World, root: Entity, visit_node: impl FnMut(Entity)) {
    breadth_first(world, root, visit_node);
}

fn depth_first_sys<'a>(
    nodes_storage: &ReadStorage<'a, RenderGraphNode>,
    root: Entity,
    mut visit_node: impl FnMut(Entity),
) {
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

pub fn depth_first(world: &World, root: Entity, visit_node: impl FnMut(Entity)) {
    let nodes_storage = world.read_storage::<RenderGraphNode>();
    depth_first_sys(&nodes_storage, root, visit_node);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Component)]
    #[storage(DenseVecStorage)]
    struct ID(usize);

    fn setup_world() -> World {
        let mut world = World::new();

        world.register::<RenderGraphNode>();
        world.register::<RenderGraphRoot>();
        world.register::<RenderGraphChild>();
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
            .insert(child, super::child(parent))
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

        breadth_first(&w, root, visit_node);
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

        depth_first(&w, root, visit_node);
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
    fn node_to_root() {
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
            node_to_root_path(&w, id2ent[2]),
            vec![2, 1]
        ));
        assert!(verify_it(
            &id2ent,
            node_to_root_path(&w, id2ent[5]),
            vec![5, 2, 1]
        ));
        assert!(verify_it(
            &id2ent,
            node_to_root_path(&w, id2ent[3]),
            vec![3, 1]
        ));
        assert!(verify_it(
            &id2ent,
            node_to_root_path(&w, id2ent[6]),
            vec![6, 2, 1]
        ));
        assert!(verify_it(
            &id2ent,
            node_to_root_path(&w, id2ent[4]),
            vec![4, 1]
        ));
        assert!(verify_it(
            &id2ent,
            node_to_root_path(&w, id2ent[7]),
            vec![7, 4, 1]
        ));
        assert!(verify_it(
            &id2ent,
            node_to_root_path(&w, id2ent[1]),
            vec![1]
        ));
    }

    #[test]
    fn root_to_node() {
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
            root_to_node_path(&w, id2ent[2]),
            vec![1, 2]
        ));
        assert!(verify_it(
            &id2ent,
            root_to_node_path(&w, id2ent[5]),
            vec![1, 2, 5]
        ));
        assert!(verify_it(
            &id2ent,
            root_to_node_path(&w, id2ent[3]),
            vec![1, 3]
        ));
        assert!(verify_it(
            &id2ent,
            root_to_node_path(&w, id2ent[6]),
            vec![1, 2, 6]
        ));
        assert!(verify_it(
            &id2ent,
            root_to_node_path(&w, id2ent[4]),
            vec![1, 4]
        ));
        assert!(verify_it(
            &id2ent,
            root_to_node_path(&w, id2ent[7]),
            vec![1, 4, 7]
        ));
        assert!(verify_it(
            &id2ent,
            root_to_node_path(&w, id2ent[1]),
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
            assert_eq!(root_to_node_path(&w, ent).len(), e);
            assert_eq!(root_to_node_path(&w, ent).size_hint().0, e);
            assert_eq!(root_to_node_path(&w, ent).size_hint().1, Some(e));
            assert_eq!(node_to_root_path(&w, ent).len(), e);
            assert_eq!(node_to_root_path(&w, ent).size_hint().0, e);
            assert_eq!(node_to_root_path(&w, ent).size_hint().1, Some(e));
        }
    }
}
