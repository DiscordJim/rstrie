use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    fmt::Debug,
    ops::{Index, IndexMut},
    slice::Iter,
    vec::Drain,
};

#[derive(Debug, Default, Clone)]
pub(crate) struct Node<K, V> {
    key: Option<K>,
    sub_keys: Vec<NodeIndex>,
    value: Option<V>,
    // pub(crate) parent: Option<NodeIndex>,
}

impl<K, V> Node<K, V> {
    pub const fn root() -> Self {
        Node {
            key: None,
            sub_keys: Vec::new(),
            value: None,
        }
    }
    pub fn keyed(key: K) -> Self {
        Self {
            sub_keys: Vec::default(),
            value: None,
            key: Some(key),
        }
    }
}

impl<K, V> Node<K, V> {
    pub fn key(&self) -> &Option<K> {
        &self.key
    }
    pub fn is_root(&self) -> bool {
        self.key.is_none()
    }
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
                let sub = buffer[*k].key().as_ref().unwrap();
                functor(sub)
            })
            .ok()?;

        Some(&self.sub_keys[result])
    }

    pub fn value_mut_unchecked(&mut self) -> &mut V {
        self.value_mut().as_mut().unwrap()
    }

    pub fn sub_keys(&self) -> Iter<'_, NodeIndex> {
        self.sub_keys.iter()
    }
    pub fn insert_subkey(&mut self, value: NodeIndex, bin: &mut Slots<K, V>) -> Option<NodeIndex>
    where
        K: Ord,
    {
        match self.bin_search(value, bin) {
            Ok(valid) => {
                let old = std::mem::replace(&mut self.sub_keys[valid], value);
                Some(old)
            }
            Err(invalid) => {
                self.sub_keys.insert(invalid, value);
                None
            }
        }
    }
    pub fn remove_subkey(&mut self, result: NodeIndex) -> Option<NodeIndex> {
        if result == NodeIndex::ROOT {
            return Some(NodeIndex::ROOT);
        }
        let position = self.sub_keys().position(|s| *s == result)?;
        Some(self.sub_keys.remove(position))
    }
    // pub fn insert(&mut self, key: K, value: NodeIndex) -> Option<NodeIndex>
    // where
    //     K: Ord
    // {

    //     match self.bin_search(&key) {
    //         Ok(found) => {

    //             let (_ , old) = std::mem::replace(&mut self.sub_keys[found], (key, value));
    //             Some(old)
    //         }

    //         Err(found) => {
    //             self.sub_keys.insert(found, (key, value));

    //             None
    //         }
    //     }

    //     // let old = self.remove(&key);

    //     // self.sub_keys.push((key, value));

    //     // old
    // }
    pub fn bin_search(&self, reference: NodeIndex, buffer: &Slots<K, V>) -> Result<usize, usize>
    where
        K: Ord,
    {
        let key = buffer[reference].key().as_ref().unwrap();
        self.sub_keys
            .binary_search_by(|node| buffer[*node].key.as_ref().unwrap().cmp(key))
    }
    // pub fn remove(&mut self, key: &K) -> Option<NodeIndex>
    // where
    //     K: Ord
    // {
    //     Some(self.sub_keys.remove(self.bin_search(key).ok()?).1)
    // }
    pub fn sub_key_len(&self) -> usize {
        self.sub_keys.len()
    }
    pub fn into_value(self) -> Option<V> {
        self.value
    }
    pub fn value(&self) -> &Option<V> {
        &self.value
    }
    pub fn value_mut(&mut self) -> &mut Option<V> {
        &mut self.value
    }

    pub fn semantic_equals(&self, other: &Self, list_self: &Slots<K, V>, list_other: &Slots<K, V>) -> bool
    where
        K: PartialEq + Debug,
        V: PartialEq + Debug,
    {
        // Check to see if the K, V are equal.
        if !(self.key.eq(&other.key) && self.value.eq(&other.value)) {
            println!("Failed A check at {:?} vs. {:?}", self, other);
            return false;
        }

        // Subkey lists must be the same.
        if self.sub_keys.len() != other.sub_keys.len() {
            println!("Failed length check at {:?} vs. {:?}", self, other);
            return false;
        }

        for i in 0..self.sub_keys.len() {
            
            if !list_self[self.sub_keys[i]].semantic_equals(&list_other[other.sub_keys[i]], list_self, list_other) {
                println!("Key comparison failed for keys: {:?} and {:?}", list_self[self.sub_keys[i]].key, list_other[other.sub_keys[i]].key);
                return false;
            }
        }

        true

        
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Slots<K, V> {
    slots: Vec<Option<Node<K, V>>>,
    free_list: Vec<usize>,
}

// impl<K, V> PartialEq for Node<K, V>
// where
//     K: PartialEq,
//     V: PartialEq,
// {
//     fn eq(&self, other: &Self) -> bool {
//         self.key.eq(&other.key) && self.sub_keys.eq(&other.sub_keys) && self.value.eq(&other.value)
//     }
// }

impl<K, V> PartialEq for Slots<K, V>
where
    K: PartialEq + Debug,
    V: PartialEq + Debug,
    Node<K, V>: Debug,
{
    fn eq(&self, other: &Self) -> bool {
        // for (a, b) in self
        //     .slots
        //     .iter()
        //     .filter_map(Option::as_ref)
        //     .zip(other.slots.iter().filter_map(Option::as_ref))
        // {
        //     // for b  in {
        //     //     println!("HELLO: {:?}", a);
        //     //     println!("HELLO2: {:?}", b);
        //     //     if a.ne(b) {
        //     //         return false;
        //     //     }
        //     // }
        //     if !a.semantic_equals(b, self) {
        //         println!("Failed at {a:?} and {b:?}");
        //         return false;
        //     }
        // }
        // return true;


        // let mut stack = VecDeque::new();
        // stack.push_front(NodeIndex::ROOT);
        // while let Some(op) = stack.pop_front() {
        //     println!("Node: {:?}", self[op].key);
        //     for i in self[op].sub_keys() {
        //         stack.push_front(*i);
        //     }
        // }

        self.preorder();
        println!("SECOND:");
        other.preorder();

        self[NodeIndex::ROOT].semantic_equals(&other[NodeIndex::ROOT], self, other)
    }
}




#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NodeIndex {
    pub(crate) position: usize,
}

impl NodeIndex {
    pub const ROOT: NodeIndex = NodeIndex { position: 0 };
}

impl<K, V> Slots<K, V> {
    pub fn with_capacity(cap: usize) -> Self {
        let mut new = Self {
            slots: Vec::with_capacity(cap),
            free_list: vec![],
        };
        new.slots.insert(0, Some(Node::root()));
        new
    }
    pub fn preorder(&self)
    where 
        Node<K, V>: Debug,
        K: Debug
    {
        let mut stack = VecDeque::new();
        stack.push_back(NodeIndex::ROOT);
        while let Some(op) = stack.pop_back() {
            println!("Node: {:?} || {:?}", self[op].key, self[op]);
            for i in self[op].sub_keys() {
                stack.push_front(*i);
            }
        }
    }
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }
    pub fn slot_iter(&self) -> std::slice::Iter<'_, Option<Node<K, V>>> {
        self.slots.iter()
    }
    pub fn slot_iter_mut(&mut self) -> std::slice::IterMut<'_, Option<Node<K, V>>> {
        self.slots.iter_mut()
    }
    pub fn reserve(&mut self, quantity: usize) {
        self.slots.reserve(quantity);
    }
    pub fn drain_slots(&mut self) -> Drain<'_, Option<Node<K, V>>> {
        self.slots.drain(0..self.slots.len())
    }
    pub fn insert(&mut self, item: Node<K, V>) -> NodeIndex {
        if !self.free_list.is_empty() {
            let avail = self.free_list.pop().unwrap();
            self.slots[avail] = Some(item);
            NodeIndex { position: avail }
        } else {
            self.slots.push(Some(item));
            NodeIndex {
                position: self.slots.len() - 1,
            }
        }
    }
    pub fn nullify(&mut self, index: NodeIndex) {
        self.slots[index.position] = None;
    }
    pub fn remove(&mut self, index: NodeIndex) -> Option<Node<K, V>> {
        if index.position == 0 {
            // You cannot remove the root node.
            return None;
        }
        let pos = &mut self.slots[index.position];
        if pos.is_some() {
            // Add the position to the free list.
            self.free_list.push(index.position);
        }
        pos.take()
    }
    pub fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Node<K, V>)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_some())
            .map(|(i, key)| (NodeIndex { position: i }, key.as_ref().unwrap()))
    }
    pub fn drain(&mut self) -> impl Iterator<Item = Node<K, V>> {
        self.clear_root();
        self.slots
            .drain(1..self.slots.len())
            // .enumerate()
            .flatten()
    }
    fn clear_root(&mut self) {
        self[NodeIndex::ROOT].value_mut().take();
        self[NodeIndex::ROOT].sub_keys.clear();
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
    pub fn remove_subkey(&mut self, source: NodeIndex, to_remove: NodeIndex) -> Option<NodeIndex> {
        self[source].remove_subkey(to_remove)
    }
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
    /// Defragments the map, please note that this will correct all INTERNAL node indices but
    /// any existing (living) ones will become invalidated.
    pub fn defragment(&mut self) {
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
                remapper.insert(NodeIndex { position: index }, NodeIndex { position: drag });
                self.slots.swap(index, drag);
                drag += 1;
            }
        }

        // We no longer need the free-list!
        self.free_list.clear();

        // Remap all the keys.
        for node in self
            .slots
            .iter_mut()
            .filter_map(<Option<Node<K, V>>>::as_mut)
        {
            for key in &mut node.sub_keys {
                if remapper.contains_key(key) {
                    *key = *remapper.get(key).unwrap();
                }
            }
        }
    }
}

impl<K, V> Index<NodeIndex> for Slots<K, V> {
    type Output = Node<K, V>;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.slots[index.position].as_ref().unwrap()
    }
}

impl<K, V> IndexMut<NodeIndex> for Slots<K, V> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.slots[index.position].as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::Node;

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
        assert!(list.remove(super::NodeIndex { position: 0 }).is_none());
    }

    fn check_map_integrity_defragmented(map: &Slots<char, String>) {
        // Condition A.1: All keys are valid.
        // Condition A.2: All the parent references are satisfied.
        for slot in map
            .slots
            .iter()
            .filter(|f| f.is_some())
            .map(|f| f.as_ref().unwrap())
        {
            for key in slot.sub_keys() {
                if map.slots[key.position].is_none() {
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
