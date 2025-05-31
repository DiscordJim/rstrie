use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Debug,
    ops::{Index, IndexMut},
    slice::{GetDisjointMutError, Iter},
    vec::Drain,
};

/// A Trie node that holds a key fragment, an array of subkeys, and a value.
/// A key fragment is a type that is part of the greater key type. For instance,
/// a [char] in sequence forms part of a [String], which allows the Trie keys to be
/// collected into the greater type.
///
/// In the case of a root node then the key will just be [Option::None]. For the sake of
/// efficient serialization, if the `value` field is [Option::None] then it will not serialize
/// at all instead of just serializing as `null`,
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub(crate) struct Node<K, V> {
    /// The node key
    key: Option<K>,
    /// The node subkeys, which points to other nodes in the [Slots].
    sub_keys: Vec<NodeIndex>,
    /// The value of the node. This is only populated if this node terminates
    /// a key.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    value: Option<V>,
}

impl<K, V> Node<K, V> {
    /// Creates a new root node. This just consists
    /// of a normal node with a `null` key.
    pub const fn root() -> Self {
        Node {
            key: None,
            sub_keys: Vec::new(),
            value: None,
        }
    }
    /// Creates a keyed node. This is more akin to
    /// what would be considered a "standard" node.
    pub fn keyed(key: K) -> Self {
        Self {
            sub_keys: Vec::default(),
            value: None,
            key: Some(key),
        }
    }
}

impl<K, V> Node<K, V> {
    /// Gets the node key. This will be [Option::None] in the case
    /// that we are dealing with a root node.
    pub fn key(&self) -> &Option<K> {
        &self.key
    }
    /// Performs a binary search with a function that provides an ordering
    /// object for key. This can be used to define more complex searches.
    ///
    /// For instance, if we would like to order them by position in the list
    /// instead of a key value if we are doing some sort of walk.
    pub fn get_with<'a, F>(
        &'a self,
        buffer: &'a Slots<K, V>,
        mut functor: F,
    ) -> Option<&'a NodeIndex>
    where
        F: FnMut(&'a K) -> Ordering,
    {
        let result = self
            .sub_keys
            .binary_search_by(|k| {
                // UNWRAP: This unwrap is okay, because no node can point to the root node.
                // The root node is the only node that has a null key.
                let sub = buffer[*k].key().as_ref().expect("Node key was null.");
                functor(sub)
            })
            .ok()?;

        Some(&self.sub_keys[result])
    }

    /// Gets the inner value of the node regardless
    /// of if it is [Option::None] or not.
    pub fn value_mut_unchecked(&mut self) -> &mut V {
        self.value_mut()
            .as_mut()
            .expect("Tried to get the value as some value but was None.")
    }
    /// Gets an iterator of all the subkeys as a form
    /// of [NodeIndex] iterators.
    pub fn subkeys(&self) -> Iter<'_, NodeIndex> {
        self.sub_keys.iter()
    }
    /// Removes a subkey from the node.
    pub fn remove_subkey(&mut self, result: NodeIndex) -> Option<NodeIndex> {
        if result == NodeIndex::ROOT {
            return Some(NodeIndex::ROOT);
        }
        let position = self.subkeys().position(|s| *s == result)?;
        Some(self.sub_keys.remove(position))
    }
    /// Performs a binary search over all the node keys, comparing them
    /// by ordering. This requires that `K` implements [Ord].
    fn bin_search(&self, reference: NodeIndex, buffer: &Slots<K, V>) -> Result<usize, usize>
    where
        K: Ord,
    {
        let key = buffer[reference]
            .key()
            .as_ref()
            .expect("Reference node was the root node.");
        self.sub_keys.binary_search_by(|node|
                // UNWRAP: This unwrap is okay, because no node can point to the root node.
                // The root node is the only node that has a null key.
                buffer[*node].key.as_ref().expect("Node key was null").cmp(key))
    }
    /// Returns the length of the subkey array, or in other words,
    /// how many subkeys the specific node has.
    pub fn sub_key_len(&self) -> usize {
        self.sub_keys.len()
    }
    /// Turns the node into the inner value `V`
    pub fn into_value(self) -> Option<V> {
        self.value
    }
    /// Returns an immutable reference to the inner value.
    pub fn value(&self) -> &Option<V> {
        &self.value
    }
    /// Returns a mutable reference to the inner value.
    pub fn value_mut(&mut self) -> &mut Option<V> {
        &mut self.value
    }
    /// Checks that two subtrees are equal. The arguments are two
    /// node indexes, one being the node index initiating the semantic search, and
    /// the other being the root of the subtree to compare against.
    ///
    /// Keys are checked for equality, along with subkeys. Since two trees can
    /// have various levels of fragmentation, a simple node comparison cannot be
    /// used.
    pub fn semantic_equals(
        &self,
        other: &Self,
        list_self: &Slots<K, V>,
        list_other: &Slots<K, V>,
    ) -> bool
    where
        K: PartialEq,
        V: PartialEq,
    {
        // Check to see if the K, V are equal.
        if !(self.key.eq(&other.key) && self.value.eq(&other.value)) {
            return false;
        }

        // Subkey lists must be the same.
        if self.sub_keys.len() != other.sub_keys.len() {
            return false;
        }

        for i in 0..self.sub_keys.len() {
            if !list_self[self.sub_keys[i]].semantic_equals(
                &list_other[other.sub_keys[i]],
                list_self,
                list_other,
            ) {
                return false;
            }
        }

        true
    }
}

/// The array that holds all the underlying node data. It works
/// by holding a freelist for filling tombstone slots, and by maintaining
/// a simple vector. Defragmentation happens when called manually.
///
/// The slots will always have a root. In practice, this means that it will
/// never error because the root was indexed and it did not exist. Great
/// care is put into maintaining the root within the list.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub(crate) struct Slots<K, V> {
    /// A list of optional slots containing nodes. These may be
    /// [Option::None] in the case of a tombstone, i.e., a node
    /// that once was but has since been deleted.
    slots: Vec<Option<Node<K, V>>>,
    /// A freelist of all the available space within the array.
    free_list: Vec<usize>,
}

impl<K, V> PartialEq for Slots<K, V>
where
    K: PartialEq,
    V: PartialEq,
{
    /// Performs a semantic equals comparison across the nodes starting from the two roots,
    /// and checking each subtree recursively.
    fn eq(&self, other: &Self) -> bool {
        self[NodeIndex::ROOT].semantic_equals(&other[NodeIndex::ROOT], self, other)
    }
}

/// Represents the index of a node within a slotmap. Requires caution, as
/// defragmenting the map will cause indices to be invalidated. Special care
/// is taken to handle this, and this is why the struct is invisible to the
/// end-developer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize),
    rkyv(derive(PartialEq, Debug))
)]
pub(crate) struct NodeIndex(u64);

impl NodeIndex {
    /// The root node, which always has an internal index of 0.
    pub const ROOT: NodeIndex = NodeIndex(0);
}

impl NodeIndex {
    /// Gets the internal position of the node as a [usize]/
    pub fn position(&self) -> usize {
        self.0 as usize
    }
}

impl<K, V> Slots<K, V> {
    /// Creates a new [Slots] with a certain amount of capacity
    pub fn with_capacity(cap: usize) -> Self {
        let mut new = Self {
            slots: Vec::with_capacity(cap),
            free_list: vec![],
        };
        new.slots.insert(0, Some(Node::root()));
        new
    }
    /// Gets the capacity of the [Slots].
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }
    /// Iterates over the optional slots immutably.
    pub fn slot_iter(&self) -> std::slice::Iter<'_, Option<Node<K, V>>> {
        self.slots.iter()
    }
    /// Iterates over the optional slots mutably.
    pub fn slot_iter_mut(&mut self) -> SlotsIterMut<'_, K, V> {
        SlotsIterMut {
            inner: self.slots.iter_mut(),
        }
    }
    /// Reserves a certain quantity in the underlying [Vec] that makes
    /// up the [Slots].
    pub fn reserve(&mut self, quantity: usize) {
        self.slots.reserve(quantity);
    }
    /// Drains the slots. This is just a thin wrapper around the [Vec::drain]
    /// method. Returns them as optional slots. Used as a helper method to drain
    /// the actual tree.
    ///
    /// This is a very heavy method as it shifts all the elements over to the right.
    pub fn drain_slots(&mut self) -> Drain<'_, Option<Node<K, V>>> {
        self.slots.insert(0, Some(Node::root()));
        self.slots.drain(1..self.slots.len())
    }
    /// Gets disjoint mutably references from the slots. This is
    /// returned as a special object called [DisjointIndices] that
    /// prevents us from having to use `unsafe` code.
    pub fn get_disjoint_mut<const N: usize>(
        &mut self,
        nodes: [NodeIndex; N],
    ) -> Result<DisjointMutIndices<'_, K, V, N>, GetDisjointMutError> {
        // Translate all the nodes into positions.
        let translated: [usize; N] = core::array::from_fn(|i| nodes[i].position());
        // Gets the disjoint mutables for the underlying [Vec].
        let r = self.slots.get_disjoint_mut(translated)?;
        Ok(DisjointMutIndices(r))
    }
    /// Inserts a [Node] into the underlying [Vec], returning
    /// the new [NodeIndex].
    pub fn insert(&mut self, item: Node<K, V>) -> NodeIndex {
        if !self.free_list.is_empty() {
            let avail = self
                .free_list
                .pop()
                .expect("List was not empty but could not pop.");
            self.slots[avail] = Some(item);
            NodeIndex(avail as u64)
        } else {
            self.slots.push(Some(item));
            NodeIndex((self.slots.len() - 1) as u64)
        }
    }
    /// Removes a node from the underlying [Vec].
    pub fn remove(&mut self, index: NodeIndex) -> Option<Node<K, V>> {
        if index.position() == 0 {
            // Take the current root out.
            let root = core::mem::take(&mut self.slots[index.position()]);

            // Swap a fresh root back in.
            self.slots[index.position()] = Some(Node::root());

            // Return the old root.
            return root;
        }
        let pos = &mut self.slots[index.position()];
        if pos.is_some() {
            // Add the position to the free list.
            self.free_list.push(index.position());
        }
        pos.take()
    }
    /// Returns an iterator of node indices alongside the [Node] objects themselves.
    pub fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Node<K, V>)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, f)| Some((NodeIndex(i as u64), f.as_ref()?)))
    }
    pub fn insert_subkey(&mut self, source: NodeIndex, value: NodeIndex) -> Option<NodeIndex>
    where
        K: Ord,
    {
        match self[source].bin_search(value, self) {
            Ok(valid) => {
                let old = std::mem::replace(&mut self[source].sub_keys[valid], value);
                Some(old)
            }
            Err(invalid) => {
                self[source].sub_keys.insert(invalid, value);
                None
            }
        }
    }
    /// Clears the underlying vector, reinserting the root node into
    /// the [Slots].
    pub fn clear(&mut self) {
        self.slots.clear();
        self.slots.insert(0, Some(Node::root()));
    }
    /// This will remove any empty slots from the end of the memory and will
    /// then reduce the internal vector to the minimum possible capacity.
    pub fn shrink_to_fit(&mut self) {
        while self.slots[self.slots.len() - 1].is_none() {
            self.slots.remove(self.slots.len() - 1);
        }
        self.slots.shrink_to_fit();
    }
    fn get_defrag_map(&mut self) -> HashMap<NodeIndex, NodeIndex> {
        // Create a [HashMap] to keep track of the old positions so we can map
        // them to the new positions.
        let mut remapper = HashMap::<NodeIndex, NodeIndex>::new();

        // Initialize a new variable called "drag".
        let mut drag = 0;
        for index in 0..self.slots.len() {
            // If the slot is not vacant and the drag is TRAILING, then we can
            // advance it so that it is pointing to the next slot.
            if self.slots[index].is_some() && drag == index {
                drag += 1;
            } else if self.slots[index].is_some() {
                // If this condition is met, then the drag is trailing behind by more than one.
                remapper.insert(NodeIndex(index as u64), NodeIndex(drag as u64));
                self.slots.swap(index, drag);
                drag += 1;
            }
        }
        remapper
    }
    /// Defragments the map, please note that this will correct all INTERNAL node indices but
    /// any existing (living) ones will become invalidated.
    pub fn defragment(&mut self) {
        let remapper = self.get_defrag_map();

        // We no longer need the free-list!
        self.free_list.clear();

        // Remap all the keys.
        for node in self
            .slots
            .iter_mut()
            .filter_map(<Option<Node<K, V>>>::as_mut)
        {
            for key in &mut node.sub_keys {
                if let Some(new_k) = remapper.get(key) {
                    *key = *new_k;
                }
            }
        }
    }
}

/// Returns an iterator of the underlying vector that skips
/// over the entries that are tombstones.
pub(crate) struct SlotsIterMut<'a, K, V> {
    inner: core::slice::IterMut<'a, Option<Node<K, V>>>,
}

impl<'a, K, V> Iterator for SlotsIterMut<'a, K, V> {
    type Item = &'a mut Node<K, V>;

    /// Skips over the tomstoned entries until we eventually
    /// get to the next valid value.
    fn next(&mut self) -> Option<Self::Item> {
        let mut current = self.inner.next()?;
        while current.is_none() {
            current = self.inner.next()?;
        }
        current.as_mut()
    }
}

impl<K, V> Index<NodeIndex> for Slots<K, V> {
    type Output = Node<K, V>;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.slots[index.position()]
            .as_ref()
            .expect("Could not find node at requested index.")
    }
}

impl<K, V> IndexMut<NodeIndex> for Slots<K, V> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.slots[index.position()]
            .as_mut()
            .expect("Could not find node at requested index.")
    }
}

/// Represents a set of disjoint mutable indices.
pub struct DisjointMutIndices<'a, K, V, const N: usize>([&'a mut Option<Node<K, V>>; N]);

impl<'a, K, V, const N: usize> DisjointMutIndices<'a, K, V, N> {
    /// Returns the length of the disjoint mutable indices.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, K, V, const N: usize> Index<usize> for DisjointMutIndices<'a, K, V, N> {
    type Output = Option<V>;
    fn index(&self, index: usize) -> &Self::Output {
        self.0[index]
            .as_ref()
            .expect("Could not find node at requested index.")
            .value()
    }
}

impl<'a, K, V, const N: usize> IndexMut<usize> for DisjointMutIndices<'a, K, V, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let inner = self.0[index]
            .as_mut()
            .expect("Could not find node at requested index.");
        &mut inner.value
    }
}

#[cfg(test)]
mod tests {
    use crate::{Node, NodeIndex};

    use super::Slots;

    #[test]
    pub fn basic_nodelist() {
        let mut list = Slots::<char, String>::with_capacity(0);
        let key = list.insert(Node::root());
        assert_eq!(list.free_list.len(), 0);
        assert_eq!(list.slots.len(), 2);

        list.remove(key).unwrap();

        // list should retain props
        assert_eq!(list.free_list.len(), 1);
        assert_eq!(list.slots.len(), 2);

        list.insert(Node::root());
        assert_eq!(list.free_list.len(), 0);
        assert_eq!(list.slots.len(), 2);
    }

    #[test]
    pub fn remove_root() {
        let mut list = Slots::<char, String>::with_capacity(0);
        assert!(list.remove(super::NodeIndex::ROOT).unwrap().key().is_none());
    }

    #[test]
    pub fn root_exists_post_drain() {
        let mut list = Slots::<char, usize>::with_capacity(1);
        list.insert(Node::keyed('a'));
        for _ in list.drain_slots() {}
        assert!(list[NodeIndex::ROOT].key().is_none());
    }

    fn check_map_integrity_defragmented(map: &Slots<char, String>) {
        // Condition A.1: All keys are valid.
        // Condition A.2: All the parent references are satisfied.
        for slot in map.slots.iter().filter_map(Option::as_ref) {
            for key in slot.subkeys() {
                if map.slots[key.position()].is_none() {
                    panic!("Key reference invalid!");
                }
            }
        }

        // Condition B: All the nones are at the end of the list.
        let mut has_seen_none = false;
        for slot in &map.slots {
            if slot.is_none() {
                has_seen_none = true;
            } else if slot.is_some() && has_seen_none {
                panic!("There is an empty slot not at end of list.");
            }
        }
    }

    #[test]
    pub fn test_remap_shrink() {
        let mut list = Slots::<char, String>::with_capacity(0);
        let _ = list.insert(Node::root());
        let bruha = list.insert(Node::keyed('a'));
        let bruh = list.insert(Node::keyed('b'));

        let bruhc = list.insert(Node::keyed('c'));
        let bruhd = list.insert(Node::keyed('d'));
        let bruhe = list.insert(Node::keyed('e'));
        let bruhf = list.insert(Node::keyed('f'));

        list.insert_subkey(bruhf, bruhd);
        // list[bruhf].insert_subkey(bruhd, &mut list);
        // list[bruhf].sub_keys.push(bruhd);

        list.remove(bruh);
        list.remove(bruhc);
        list.remove(bruha);
        list.remove(bruhe);

        list.defragment();

        check_map_integrity_defragmented(&list);

        list.shrink_to_fit();

        check_map_integrity_defragmented(&list);
    }
}
