use std::{collections::HashMap, fmt::Debug, iter::{Enumerate, Filter, Map}, ops::{Index, IndexMut, RangeBounds}, slice};

use crate::Node;


#[derive(Debug, Clone)]
pub(crate) struct Slots<K, V> {
    pub(crate) slots: Vec<Option<Node<K, V>>>,
    free_list: Vec<usize>
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NodeIndex {
    position: usize
}

impl<K, V> Slots<K, V> {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free_list: vec![]
        }
    }
    pub fn insert(&mut self, item: Node<K, V>) -> NodeIndex {
        if !self.free_list.is_empty() {
            let avail = self.free_list.pop().unwrap();
            self.slots[avail] = Some(item);
            NodeIndex {
                position: avail
            }
        } else {
            self.slots.push(Some(item));
            NodeIndex {
                position: self.slots.len() - 1
            }
        }
    }
    pub fn remove(&mut self, index: NodeIndex) -> Option<Node<K, V>> {
        let pos = &mut self.slots[index.position];
        if pos.is_some() {
            // Add the position to the free list.
            self.free_list.push(index.position);
        }
        pos.take()
    }
    //Map<Filter<Enumerate<std::slice::Iter<'_, Option<Node<K, V>>>>
    pub fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Node<K, V>)> {
        self.slots.iter().enumerate()
            .filter(|(_, f)| f.is_some())
            .map(|(i, key)| (NodeIndex { position: i }, key.as_ref().unwrap()))
    }
    pub fn values(&self) -> impl Iterator<Item = &Option<V>> {
        self.slots.iter()
            .filter(|f| f.is_some())
            .map(|f| f.as_ref().unwrap().value())
    }
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Option<V>> {
        self.slots.iter_mut()
            .filter(|f| f.is_some())
            .map(|f| f.as_mut().unwrap().value_mut())
    }
    pub fn drain(&mut self) -> impl Iterator<Item = Node<K, V>> {
        self.slots.drain(0..self.slots.len())
            // .enumerate()
            .filter(|f| f.is_some())
            .map(Option::unwrap)
    }
    pub fn clear(&mut self) {
        self.slots.clear();
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
    pub fn defragment(&mut self)
    where 
        Node<K, V>: Debug
    {
        // Create a [HashMap] to keep track of the old positions so we can map
        // them to the new positions.
        let mut remapper = HashMap::<NodeIndex, NodeIndex>::new();

        // Initialize a new variable called "drag".
        let mut drag = 0;
        for index in 0 .. self.slots.len() {
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
        for node in self.slots.iter_mut().filter(|f| f.is_some()).map(|f| f.as_mut().unwrap()) {
            if let Some(par) = &mut node.parent {
                if remapper.contains_key(&par) {
                    *par = *remapper.get(&par).unwrap();
                }
            }  
            
            for key in &mut node.sub_keys {
                if remapper.contains_key(key) {
                    *key = *remapper.get(&key).unwrap();
                }
            }

        }


    }


    pub fn node_iter_mut(&mut self) -> NodeIterMut<'_, K, V> {
        NodeIterMut { inner: self.slots.iter_mut(), position: 0 }
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

pub(crate) struct NodeIterMut<'a, K, V> {
    inner: std::slice::IterMut<'a, Option<Node<K, V>>>,
    position: usize
}

impl<'a, K, V> Iterator for NodeIterMut<'a, K, V> {
    type Item = (NodeIndex, &'a mut Node<K, V>);




    fn next(&mut self) -> Option<Self::Item> {
        let mut current = None;
        while current.is_none() {
            let candidate = self.inner.next()?.as_mut(); // propagate initial value to check completion.
            match candidate {
                Some(inner) => {
                    current = Some((NodeIndex {
                        position: self.position
                    }, inner));
                    self.position += 1;
                }
                None => {
                    self.position += 1;
                    continue;
                }
            }
            
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use crate::Node;

    use super::Slots;


    #[test]
    pub fn basic_nodelist() {
        let mut list = Slots::<char, String>::new();
        let key = list.insert(Node::root());
        assert_eq!(list.free_list.len(), 0);
        assert_eq!(list.slots.len(), 1);

        list.remove(key).unwrap();

        // list should retain props
        assert_eq!(list.free_list.len(), 1);
        assert_eq!(list.slots.len(), 1);


        list.insert(Node::root());
        assert_eq!(list.free_list.len(), 0 );
        assert_eq!(list.slots.len(), 1);
        
    }


    fn check_map_integrity_defragmented(map: &Slots<char, String>) {
        // Condition A.1: All keys are valid.
        // Condition A.2: All the parent references are satisfied.
        for slot in map.slots
            .iter()
            .filter(|f| f.is_some())
            .map(|f| f.as_ref().unwrap())
        
        {
        

            for key in &slot.sub_keys {
                if map.slots[key.position].is_none() {
                    panic!("Key reference invalid!");
                }
            }

            if let Some(par) = slot.parent() {
                if map.slots[par.position].is_none() {
                    panic!("Parent reference has been made invalid.");
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

        let mut list = Slots::<char, String>::new();
        let key = list.insert(Node::root());
        let bruha = list.insert(Node::keyed('a', key));
        let bruh = list.insert(Node::keyed('b', key));

        let bruhc = list.insert(Node::keyed('c', key));
        let bruhd = list.insert(Node::keyed('d', key));
        let bruhe = list.insert(Node::keyed('e', key));
        let bruhf = list.insert(Node::keyed('f', bruhd));

        list[bruhf].sub_keys.push(bruhd);
  


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