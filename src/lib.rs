use std::{
    collections::VecDeque,
    fmt::Debug,
    iter::{Peekable, Zip},
    marker::PhantomData,
    ops::{Index, IndexMut},
    ptr::NonNull,
    vec::Vec
};

use list::{NodeIndex, NodeIterMut, Slots};
use node::Node;
use slotmap::{
    SlotMap,
    basic::{Values, ValuesMut},
};

mod iter;
mod node;

mod list;

pub use crate::iter::StrIter;



#[derive(Debug, Clone)]
/// A data strucur
pub struct Trie<K, V> {
    /// The node pool, this is where the internal nodes are actually store. This
    /// improves cache locality and ease of access while limiting weird lifetime errors.
    node: Slots<K, V>,
    // node: SlotMap<NodeKey, Node<K, V>>,
    /// The root node of the pool. This is where things actually start searching.
    root: NodeIndex,
    /// The amount of items in the Trie.
    size: usize,
}

#[derive(Debug)]
/// Contains the details of an incomplete [Trie] walk.
/// This is usually used to insert new nodes along incomplete
/// paths. For instance,
/// ```example
///             t
///            /  
///           e   
///          / \    
///         s   a
///          \
///           t
/// ```
/// For instance when "tea" was inserted, we would have a *common* path
/// up until 'e' and then it would diverge.
struct WalkFailure<'a, I, K> {
    /// The root of the walk.
    root: Option<NodeIndex>,
    /// The first node in the walk.
    first: Option<K>,
    /// The remainder of the path as an iterator.
    remainder: &'a mut I,
}

#[derive(Debug)]
struct WalkTrajectory {
    path: Vec<NodeIndex>,
    end: NodeIndex,
}

impl<K, V> Default for Trie<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Trie<K, V> {
    /// Creates a new [Trie] with no keys and records. This will
    /// create a [Trie] with a capacity of zero using the [Trie::with_capacity] method.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// assert_eq!(tree.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
    /// Creates a new [Trie] with a certain specified capacity.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::with_capacity(20);
    /// assert!(tree.is_empty());
    /// ```
    pub fn with_capacity(slots: usize) -> Self {
        let mut node = Slots::with_capacity(slots);
        let root = node.insert(Node::root());
        Self {
            node,
            root,
            size: 0,
        }
    }
    /// Gets the entrypoint of the tree by traversing the root. In the case
    /// where the iterator is empty, we will just return the root.
    fn get_entrypoint<I>(&self, key: &mut I) -> (Option<&NodeIndex>, Option<K>)
    where
        I: Iterator<Item = K>,
        K: Ord,
    {
        match key.next() {
            Some(first) => (self.node[self.root].get(&first, &self.node), Some(first)),
            // We just return the root node.
            None => (Some(&self.root), None),
        }
    }

    /// Performs an internal walk of the [Trie].
    fn internal_walk<'a, I>(
        &self,
        remainder: &'a mut Peekable<I>,
        track_path: bool,
    ) -> Result<WalkTrajectory, WalkFailure<'a, Peekable<I>, K>>
    where
        I: Iterator<Item = K>,
        K: Ord,
    {
        let remainder = remainder;

        let mut trajectory_path = vec![];

        if track_path {
            trajectory_path.push(self.root);
        }

        let (left, first) = self.get_entrypoint(remainder);
        if left.is_none() {
            return Err(WalkFailure {
                root: None,
                first,
                remainder,
            });
        }

        let mut end = left.unwrap();
        loop {
            let Some(current) = remainder.peek() else {
                break;
            };

            if track_path {
                trajectory_path.push(*end);
            }

            let slot = self.node[*end].get(&current, &self.node);

            if let Some(slot) = slot {
                end = slot;
            } else {
                return Err(WalkFailure {
                    root: Some(*end),
                    first,
                    remainder,
                });
            }

            // Actually consume.
            remainder.next();
        }
        if track_path {
            trajectory_path.push(*end);
        }
        Ok(WalkTrajectory {
            path: trajectory_path,
            end: *end,
        })
    }

    /// Looks up the node pool index for a certain key,
    /// the key is an iterator over the prefixes. For instance,
    /// for strings this would be characters. This method forms the basis
    /// for [Trie::get] and [Trie::get_mut].
    fn lookup_key<I>(&self, key: I) -> Option<NodeIndex>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        Some(
            self.internal_walk(&mut key.into_iter().peekable(), false)
                .ok()?
                .end,
        )
    }

    /// Traverses all the possible completions recursively, building
    /// out all the possible values.
    fn traverse_completions<'a, J>(
        &'a self,
        pool: &mut CompletionIter<'a, K, J>,
        // The index of the prior completion root.
        current: usize,
        // The index of the current node.
        index: NodeIndex,
    ) {
        for link in &self.node[index].sub_keys {
            let key = self.node[*link].key().as_ref().unwrap();
            let cur = Completion {
                value: vec![key],
                previous: Some(current),
            };

            let cur_index = pool.completion.len();
            pool.completion.push(cur);

            if self.node[*link].value().is_some() {
                pool.traversal.push(cur_index);
            }

            self.traverse_completions(pool, cur_index, *link);
        }
    }

    /// Gets all the completions of a key. This will
    /// traverse the tree to the point at which the key
    /// diverges.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, ()>::new();
    /// tree.insert("hello".chars(), ());
    /// tree.insert("hey".chars(), ());
    ///
    ///
    /// println!("Tree: {:?}", tree);
    ///
    /// let mut values = tree.completions::<_, String>("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<_>>();
    ///
    /// values.sort();
    ///
    ///
    /// assert_eq!(values[0], "hello");
    /// assert_eq!(values[1], "hey");
    /// ```
    pub fn completions<'a, I, J>(&'a self, key: I) -> CompletionIter<'a, K, J>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
        J: FromIterator<K>,
    {
        let mut binding = key.into_iter().peekable();
        let Ok(result) = self.internal_walk(binding.by_ref(), true) else {
            // There were no additional paths.
            return CompletionIter {
                completion: vec![],
                _transform: PhantomData,
                traversal: vec![],
            };
        };

        let mut col = vec![];
        for i in result.path {
            let Some(word) = self.node[i].key() else {
                continue;
            };
            col.push(word);
        }

        let mut pool = CompletionIter {
            completion: vec![Completion {
                value: col,
                previous: None,
            }],
            _transform: PhantomData,
            traversal: vec![],
        };

        self.traverse_completions(&mut pool, 0, result.end);

        pool
    }

    /// Reconstructs a key by traversing up the [Trie] structure.
    ///
    /// This will reconstruct it into a new type that can be created from
    /// an iterator of the node key types.
    fn reconstruct_node_key<J>(&self, key: NodeIndex) -> Option<J>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        let mut node = &self.node[key];

        let mut deque = VecDeque::new();
        while node.parent().is_some() {
            deque.push_front(node.key().as_ref().unwrap());

            node = &self.node[node.parent().unwrap()];
        }

        Some(deque.into_iter().collect())
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure.
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.values();
    ///
    /// assert_eq!(values.next().cloned(), Some(1));
    /// assert_eq!(values.next().cloned(), Some(2));
    /// assert_eq!(values.next().cloned(), None);
    /// ```
    pub fn values<'a>(&'a self) -> ValueIterRef<'a, K, V> {
        ValueIterRef {
            values: self.node.slots.iter(),
        }
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure mutably.
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.values_mut();
    ///
    /// assert_eq!(values.next().cloned(), Some(1));
    ///
    /// // Mutate a value to verify mutabiltiy.
    /// *values.next().unwrap() = 3;
    ///
    ///
    /// assert_eq!(values.next().cloned(), None);
    ///
    ///
    /// // Check the value was properly mutate.
    /// assert_eq!(*tree.get("bye".chars()).unwrap(), 3);
    /// ```
    pub fn values_mut<'a>(&'a mut self) -> ValueIterMut<'a, K, V> {
        ValueIterMut {
            values: self.node.slots.iter_mut()
        }
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure mutably.
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.values_mut();
    ///
    /// assert_eq!(values.next().cloned(), Some(1));
    ///
    /// // Mutate a value to verify mutabiltiy.
    /// *values.next().unwrap() = 3;
    ///
    ///
    /// assert_eq!(values.next().cloned(), None);
    ///
    ///
    /// // Check the value was properly mutate.
    /// assert_eq!(*tree.get("bye".chars()).unwrap(), 3);
    /// ```
    pub fn into_values(mut self) -> ValueIter<V> {
        let values = self
            .node
            .drain()
            .map(|v| v.into_value())
            .filter(Option::is_some)
            .collect::<Vec<Option<V>>>();

        ValueIter {
            inner: iter::Iter::new(values),
        }
    }
    /// Returns an iterator over the keys of the [Trie]
    /// data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = tree.keys::<String>();
    ///
    /// assert_eq!(key_iter.next().unwrap(), "hello");
    /// assert_eq!(key_iter.next().unwrap(), "bye");
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn keys<J>(&self) -> KeyIter<J>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        let values = self
            .node
            .iter()
            .filter(|(_, v)| v.value().is_some())
            .map(|(k, _)| k)
            .map(|k| self.reconstruct_node_key::<J>(k))
            .collect::<Vec<Option<J>>>();

        KeyIter {
            inner: iter::Iter::new(values),
        }
    }

    fn collect_entry_partial<J>(&self) -> Vec<(J, NodeIndex)>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        self.node
            .iter()
            .filter(|(_, v)| v.value().is_some())
            .map(|(key, _)| {
                let construct = self.reconstruct_node_key::<J>(key);
                (construct.unwrap(), key)
            })
            .collect()
    }

    /// Returns an iterator over the entries of the [Trie]
    /// data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = tree.into_entries::<String>();
    ///
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), 4));
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), 3));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn into_entries<J>(mut self) -> EntryIter<J, V>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        // Convert into an owned iterator.
        let values = self
            .collect_entry_partial::<J>()
            .into_iter()
            .map(|(key, value)| Some((key, self.node[value].value_mut().take().unwrap())))
            .collect::<Vec<_>>();
        EntryIter {
            inner: iter::Iter::new(values),
        }
    }
    /// Returns an iterator over the entries of the [Trie]
    /// data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = tree.entries::<String>();
    ///
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), &4));
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), &3));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn entries<J>(&self) -> EntryIter<J, &V>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        // Convert into an owned iterator.
        let values = self
            .collect_entry_partial::<J>()
            .into_iter()
            .map(|(key, value)| Some((key, self.node[value].value().as_ref().unwrap())))
            .collect::<Vec<_>>();
        EntryIter {
            inner: iter::Iter::new(values),
        }
    }

    /// Returns an iterator over the entries of the [Trie]
    /// data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = tree.entries::<String>();
    ///
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), &4));
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), &3));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn entries_mut<'b, J>(&'b mut self) -> EntryIterMut<'b, K, V, J>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        let keys = self
            .node
            .iter()
            .map(|(k, _)| self.reconstruct_node_key::<J>(k))
            .collect::<Option<Vec<_>>>()
            .unwrap();

        EntryIterMut {
            inner: keys.into_iter().zip(self.node.node_iter_mut()),
            _type: PhantomData,
        }
    }

    /// Gets a value from the [Trie] according to
    /// the key. The key is an iterable that can be used
    /// to access the record.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// tree.insert("hello".chars(), "world");
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), "world");
    /// ```
    pub fn get<I>(&self, key: I) -> Option<&V>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        self.node[self.lookup_key(key)?].value().as_ref()
    }
    /// Tries to get the key map.
    ///
    /// # Errors
    /// If the keys are not disjoint, i.e, if they map to the same value.
    fn try_get_key_map<I, const N: usize>(
        &mut self,
        keys: [I; N],
    ) -> Result<[Option<NodeIndex>; N], NodeIndex>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        let mut array = [None::<NodeIndex>; N];

        let mut i = 0;
        for k in keys.into_iter() {
            let candidate = self.lookup_key(k);

            if candidate.is_some() && array.contains(&candidate) {
                return Err(candidate.unwrap());
            }
            array[i] = candidate;

            i += 1;
        }

        Ok(array)
    }
    /// Attempts to get many mutable references to the array with a set of
    /// valid keys. All keys must be disjoint and valid or else the entire
    /// set will return None.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 3);
    /// tree.insert("world".chars(), 4);
    ///
    /// let mut keys = tree.get_disjoint_mut([ "hello".chars(), "world".chars() ]);
    /// **keys[0].as_mut().unwrap() = 4;
    /// **keys[1].as_mut().unwrap() = 2;
    ///
    /// assert_eq!(tree.get("hello".chars()), Some(&4));
    /// assert_eq!(tree.get("world".chars()), Some(&2));
    /// ```
    pub fn get_disjoint_mut<I, const N: usize>(&mut self, keys: [I; N]) -> [Option<&mut V>; N]
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        self.try_get_disjoint_mut(keys)
            .expect("Keys were overlapping.")
    }

    /// Using an array of node keys, fetch many mutable pointers.
    fn get_many_mut_ptr<const N: usize>(
        &mut self,
        keys: [Option<NodeIndex>; N],
    ) -> [Option<NonNull<V>>; N] {
        let mut many: [Option<NonNull<V>>; N] = core::array::from_fn(|_| None);

        for i in 0..N {
            match keys[i] {
                Some(inner) => {
                    let fetched = self
                        .node[inner]
                        .value_mut()
                        .as_mut()
                        .unwrap() as *mut V;

                    // SAFETY: Pointer is not null.
                    many[i] = Some(unsafe { NonNull::new_unchecked(fetched) });
                }
                None => many[i] = None,
            }
        }
        many
    }
    /// Attempts to get many mutable references to the array with a set of
    /// valid keys. All keys must be disjoint and valid or else the entire
    /// set will return None.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 3);
    /// tree.insert("world".chars(), 4);
    ///
    /// let mut keys = tree.try_get_disjoint_mut([ "hello".chars(), "world".chars() ]).unwrap();
    /// **keys[0].as_mut().unwrap() = 4;
    /// **keys[1].as_mut().unwrap() = 2;
    ///
    /// assert_eq!(tree.get("hello".chars()), Some(&4));
    /// assert_eq!(tree.get("world".chars()), Some(&2));
    /// ```
    pub fn try_get_disjoint_mut<I, const N: usize>(
        &mut self,
        keys: [I; N],
    ) -> Result<[Option<&mut V>; N], ()>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        // Calculate the key map.
        let array = self.try_get_key_map(keys).map_err(|_| ())?;

        // Get the pointer map.
        let mut ptr_map: [Option<NonNull<V>>; N] = self.get_many_mut_ptr(array);

        // Create proper references from these.
        let mut resulting = core::array::from_fn(|_| None);
        for i in 0..N {
            // SAFETY: The pointer is not null and all values are disjoint by
            // the invariants of the the [Trie::try_get_key_map] call.
            resulting[i] = ptr_map[i].take().map(|mut i| unsafe { i.as_mut() });
        }

        Ok(resulting)
    }

    /// Gets a mutable reference of a value from the [Trie] according to
    /// the key. The key is an iterable that can be used
    /// to access the record.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// tree.insert("hello".chars(), "world");
    /// assert_eq!(*tree.get_mut("hello".chars()).unwrap(), "world");
    ///
    /// *tree.get_mut("hello".chars()).unwrap() = "world2";
    /// assert_eq!(*tree.get_mut("hello".chars()).unwrap(), "world2");
    /// ```
    pub fn get_mut<I>(&mut self, key: I) -> Option<&mut V>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        let index = self.lookup_key(key)?;
        self.node[index].value_mut().as_mut()
    }
    /// Returns the amount of records within
    /// the [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// tree.insert("hello".chars(), "world");
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.size
    }
    /// Returns true if the [Trie] is empty,
    /// else it will return false.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// assert!(tree.is_empty());
    ///
    /// tree.insert("hello".chars(), "world");
    /// assert!(!tree.is_empty());
    ///
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Will clear the [Trie] data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    /// assert!(tree.is_empty());
    ///
    /// tree.insert("hello".chars(), 0);
    /// assert!(!tree.is_empty());
    ///
    /// tree.clear();
    /// assert!(tree.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.node.clear();
        self.root = self.node.insert(Node::root());
        self.size = 0;
    }

    // /// Reserves capacity for at least additional more elements to be inserted in the [Trie].
    // /// The collection may reserve more space to avoid frequent reallocations.
    // pub fn reserve(&mut self, space: usize) {
    //     self.node.reserve(space);
    // }
    /// Deletes a record from the [Trie] according to the
    /// key. It will return the old value if it is present within the
    /// data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    /// tree.insert("hello".chars(), 12);
    ///
    /// assert_eq!(tree.remove("hello".chars()).unwrap(), 12);
    /// ```
    pub fn remove<I>(&mut self, master: I) -> Option<V>
    where
        I: IntoIterator<Item = K>,
        K: Ord + Debug,
    {
        let trajectory = self
            .internal_walk(master.into_iter().peekable().by_ref(), true)
            .ok()?;

        let mut first = false;
        let mut value = None;

        let mut previous = None;
        for iter in trajectory.path.iter().rev() {

            if previous.is_some() {
                println!("REMOVING: {:?}", previous);
                self.node.remove(previous.unwrap());
            }

            if !first {
                value = self.node[*iter].value_mut().take();

                first = true;
            } else if self.node[*iter].sub_key_len() == 1 && !self.node[*iter].is_root() {
                self.node.remove(*iter);
            } else {
                // self.node[*iter].remove(&previous_key.unwrap());
                println!("Current: {:?}", iter);
                remove_node_subkey_by_key(*iter, *previous.as_ref().unwrap(), &mut self.node);
                // remove_node_subkey(*iter, previous_key.as_ref().unwrap(), &mut self.node);
                break; // we are back in a consistent state.
            }

            previous = Some(*iter);
        }

        if value.is_some() {
            self.size -= 1;
        }
        value
    }
    /// Checks if the [Trie] contains a key. This operation
    /// occurs in the same time as [Trie::get].
    pub fn contains_key<I>(&self, master: I) -> bool
    where
        I: Iterator<Item = K>,
        K: Ord,
    {
        self.get(master).is_some()
    }
    /// Checks if the [Trie] contains a value. This operation will
    /// perform a linear search of the tree and thus [Trie::get] and
    /// [Trie::get_mut] should be used instead.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    /// tree.insert("hello".chars(), 12);
    ///
    /// assert!(tree.contains_value(&12));
    /// assert!(!tree.contains_value(&11));
    ///
    /// ```
    pub fn contains_value(&self, value: &V) -> bool
    where
        V: Eq,
    {
        self.node
            .iter()
            .find(|(_, v)| v.value().is_some() && v.value().as_ref().unwrap() == value)
            .is_some()
    }

    // /// Returns the values.
    // ///
    // pub fn values(&self) ->  {

    // }
    /// Puts a new record in the [Trie], returning the old value
    /// if there previously was a value present.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    /// tree.insert("hello".chars(), 1);
    ///
    /// // Verify the key is in the tree.
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), 1);
    ///
    /// // Verify the key replacement.
    /// assert_eq!(tree.insert("hello".chars(), 2).unwrap(), 1);
    /// ```
    pub fn insert<I>(&mut self, master: I, value: V) -> Option<V>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
    {
        let master = master.into_iter();

        self.size += 1;

        match self.internal_walk(master.peekable().by_ref(), false) {
            Ok(v) => {
                let current = self.node[v.end].value_mut().take();
                *self.node[v.end].value_mut() = Some(value);
                current
            }
            Err(WalkFailure {
                root,
                remainder,
                first,
            }) => {
                let mut previous = if let Some(value) = root {
                    // Node exists
                    value
                } else {
                    // Make a new node.
                    let new_node = Node::keyed(first.unwrap(), self.root);

                    
                    let new_key = self.node.insert(new_node);
                    insert_node_subkey(self.root, new_key, &mut self.node);
                    // self.node[self.root].insert(first.as_ref().unwrap(), new_key, &self.node);
                    new_key
                };

                let mut nk = None;
                for item in remainder {
                    nk = Some(self.node.insert(Node::keyed(item, previous)));
                    insert_node_subkey(previous,nk.unwrap(), &mut self.node);
                    // self.node[previous].insert(&item, nk.unwrap(), &self.node);

                    previous = nk.unwrap();
                }

                if let Some(n) = nk {
                    *self.node[n].value_mut() = Some(value);
                }

                None
            }
        }
        // }
    }
}

#[derive(Debug)]
pub struct CompletionIter<'a, K, J> {
    /// The list of completions, this functions as a bump
    /// allocator and is used to build out the list.
    completion: Vec<Completion<'a, K>>,

    traversal: Vec<usize>,

    /// The value to which the iterators will be transformed during iteration.
    _transform: PhantomData<J>,
}

#[derive(Debug)]
struct Completion<'a, K> {
    /// The actual underlying values.
    value: Vec<&'a K>,
    /// Stores the continuations.
    previous: Option<usize>,
}


fn insert_node_subkey<K: Ord, V>(
    source: NodeIndex,
    value: NodeIndex,
    buffer: &mut Slots<K, V>
) -> Option<NodeIndex> {
    match buffer[source].bin_search(value, buffer) {
        Ok(valid) => {
            let old = std::mem::replace(&mut buffer[source].sub_keys[valid], value);
            Some(old)
        }
        Err(invalid) => {
            buffer[source].sub_keys.insert(invalid, value);
            None
        }
    }

    // None
}

fn remove_node_subkey_by_key<K: Ord, V>(
    source: NodeIndex,
    to_remove: NodeIndex,
    buffer: &mut Slots<K, V>,
) -> Option<NodeIndex> {
    let result = buffer[source].sub_keys.iter().position(|s| *s == to_remove)?;

    let elem = buffer[source].sub_keys.remove(result);

    Some(elem)
}


impl<'a, K: Debug, J> Iterator for CompletionIter<'a, K, J>
where
    J: FromIterator<&'a K>,
{
    type Item = J;

    /// Gets the next completion. This is comptued lazily.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, ()>::new();
    /// tree.insert("hello".chars(), ());
    /// tree.insert("hey".chars(), ());
    ///
    ///
    /// let mut values = tree.completions::<_, String>("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<_>>();
    ///
    /// values.sort();
    ///
    ///
    /// assert_eq!(values[0], "hello");
    /// assert_eq!(values[1], "hey");
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let mut verbiage = VecDeque::new();

        let mut current = Some(self.traversal.pop()?);
        while current.is_some() {
            let current_node = &self.completion[current?];

            verbiage.extend(current_node.value.iter().rev());
            current = self.completion[current?].previous;
        }

        Some(verbiage.into_iter().rev().collect::<J>())
    }
}

impl<K, V, I> Index<I> for Trie<K, V>
where
    K: Ord,
    I: IntoIterator<Item = K>,
{
    type Output = V;

    /// Indexes into the [Trie] using an iterator index.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let tree = Trie::<char, i32>::from([
    ///     ("apple".chars(), 4)
    /// ]);
    ///
    /// assert_eq!(tree["apple".chars()], 4);
    /// ```
    fn index(&self, index: I) -> &Self::Output {
        self.get(index).expect("Invalid trie index")
    }
}

impl<K, V, I> IndexMut<I> for Trie<K, V>
where
    K: Ord,
    I: IntoIterator<Item = K>,
{
    /// Indexes mutably into the [Trie] using an iterator index.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, i32>::from([
    ///     ("apple".chars(), 4)
    /// ]);
    ///
    /// tree["apple".chars()] = 5;
    ///
    /// assert_eq!(tree["apple".chars()], 5);
    /// ```
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect("Invalid trie index")
    }
}

// An iterator created by consuming the [Trie].
// Contains all the entries of the [Trie].
pub struct EntryIterMut<'a, K, V, J> {
    inner: Zip<std::vec::IntoIter<J>, NodeIterMut<'a, K, V>>,
    _type: PhantomData<J>,
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct EntryIter<J, V> {
    inner: crate::iter::Iter<(J, V)>,
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the keys of the [Trie].
pub struct KeyIter<J> {
    inner: crate::iter::Iter<J>,
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the elements of the [Trie].
pub struct ValueIter<V> {
    inner: crate::iter::Iter<V>,
}

/// An iterator over the values of a [Trie].
pub struct ValueIterRef<'a, K, V> {
    values: std::slice::Iter<'a, Option<Node<K, V>>>,
}

/// An iterator over the values of a [Trie]
/// that provides mutable references.
pub struct ValueIterMut<'a, K, V> {
    values: std::slice::IterMut<'a, Option<Node<K, V>>>
}

impl<'a, K, V, J> Iterator for EntryIterMut<'a, K, V, J> {
    type Item = (J, &'a mut V);

    /// Iterates over the mutably borrowed entries within a [Trie].
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = tree.entries_mut::<String>();
    ///
    /// assert_eq!(*key_iter.next().unwrap().1, 4);
    /// assert_eq!(*key_iter.next().unwrap().1, 3);
    /// assert!(key_iter.next().is_none());
    ///
    /// let mut key_iter = tree.entries_mut::<String>();
    ///
    /// *key_iter.next().unwrap().1 = 5;
    ///
    ///
    /// let mut key_iter = tree.entries_mut::<String>();
    ///
    /// assert_eq!(*key_iter.next().unwrap().1, 5);
    /// assert_eq!(*key_iter.next().unwrap().1, 3);
    /// assert!(key_iter.next().is_none());
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (key, (_, value)) = self.inner.next()?;
        if value.value_mut().is_none() {
            self.next()
        } else {
            Some((key, value.value_mut().as_mut().unwrap()))
        }
    }
}

impl<K, V> PartialEq for Trie<K, V>
where 
    K: Eq
{
    /// Checks if two [Trie] are equal.
    fn eq(&self, other: &Self) -> bool {
        for node in self.node.slots.iter().filter(|f| f.is_some()).map(|f|f.as_ref().unwrap()) {
            if !other.node.slots.iter().filter(|f| f.is_some()).map(|f| f.as_ref().unwrap()).any(|o_node| {
                node.key() == o_node.key() // make the keys equal.
                && check_node_key_equivalences(node, o_node, &self.node, &other.node)
                && check_node_key_equivalences(o_node, node, &other.node, &self.node)
            }) {
                return false;
            }
        }

        true
    }
}


fn check_node_key_equivalences<K, V>(
    node_a: &Node<K, V>,
    node_b: &Node<K, V>,
    pool_a: &Slots<K, V>,
    pool_b: &Slots<K, V>
) -> bool
where
    K: Eq
{

    node_a.sub_keys.iter().all(|key_a| {
        // For every a key.
        let value = pool_a[*key_a].key();
        node_b.sub_keys.iter().any(|key_b| pool_b[*key_b].key() == value)
    });


    true

}

impl<J, V> Iterator for EntryIter<J, V> {
    type Item = (J, V);

    /// Iterates over the owned entries within a [Trie].
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = tree.into_entries::<String>();
    ///
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), 4));
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), 3));
    /// assert!(key_iter.next().is_none());
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<J> Iterator for KeyIter<J> {
    type Item = J;

    /// Iterates over the owned keys within a [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.keys::<String>();
    ///
    /// assert_eq!(values.next().unwrap(), "hello");
    /// assert_eq!(values.next().unwrap(), "bye");
    /// assert_eq!(values.next().as_ref(), None);
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<V> Iterator for ValueIter<V> {
    type Item = V;

    /// Iterates over the owned values within a [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.into_values();
    ///
    /// assert_eq!(values.next(), Some(1));
    /// assert_eq!(values.next(), Some(2));
    /// assert_eq!(values.next(), None);
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, K, V> Iterator for ValueIterMut<'a, K, V> {
    type Item = &'a mut V;

    /// Iterates over the values mutably within a [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.values_mut();
    ///
    /// assert_eq!(values.next().cloned(), Some(1));
    /// assert_eq!(values.next().cloned(), Some(2));
    /// assert_eq!(values.next().cloned(), None);
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let mut current = None;
        while current.is_none() {
            // The INITIAL option can be propagated because this is the iterator one,
            // so if this is none then we are done anyways.
            let node = self.values.next()?;
            // We need to extract the value, and we continue until the value is something.
            current = map_node_to_value_mut(node.as_mut());
        }

        current
    }
}

fn map_node_to_value_mut<K, V>(option: Option<&mut Node<K, V>>) -> Option<&mut V> {
    let val = option?;
    val.value_mut().as_mut()
}

fn map_node_to_value<K, V>(option: Option<&Node<K, V>>) -> Option<&V> {
    let val = option?;
    val.value().as_ref()
}

impl<'a, K, V> Iterator for ValueIterRef<'a, K, V> {
    type Item = &'a V;

    /// Iterates over the values within a [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.insert("hello".chars(), 1);
    /// tree.insert("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.values();
    ///
    /// assert_eq!(values.next().cloned(), Some(1));
    /// assert_eq!(values.next().cloned(), Some(2));
    /// assert_eq!(values.next().cloned(), None);
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let mut current = None;
        while current.is_none() {
            // The INITIAL option can be propagated because this is the iterator one,
            // so if this is none then we are done anyways.
            let node = self.values.next()?;
            // We need to extract the value, and we continue until the value is something.
            current = map_node_to_value(node.as_ref());
        }

        current
    }
}

impl<KP, K, V> Extend<(KP, V)> for Trie<K, V>
where
    K: Ord,
    KP: IntoIterator<Item = K>,
{
    /// Extends a [Trie] from an iterator of tuples. The tuples must contain
    /// a valid key element, that is, it can be converted to an iterator
    /// of the sub-element. For instance, for [str] this is usually [char].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree: Trie<char, &str> = Trie::new();
    /// assert_eq!(tree.len(), 0);
    ///
    /// tree.extend([ ("hello".chars(), "world") ].into_iter());
    ///
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), "world");
    ///
    /// ```
    #[inline]
    fn extend<T: IntoIterator<Item = (KP, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<KP, K, V> FromIterator<(KP, V)> for Trie<K, V>
where
    K: Ord,
    KP: IntoIterator<Item = K>,
{
    /// Creates a [Trie] from an iterator of tuples. The tuples must contain
    /// a valid key element, that is, it can be converted to an iterator
    /// of the sub-element. For instance, for [str] this is usually [char].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree: Trie<char, usize> = Trie::from_iter([
    ///     ("hello".chars(), 4)
    /// ].into_iter());
    ///
    /// assert_eq!(tree.len(), 1);
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), 4);
    /// ```
    fn from_iter<T: IntoIterator<Item = (KP, V)>>(iter: T) -> Self {
        let mut map = Trie::new();
        map.extend(iter);
        map
    }
}

impl<KP, K, V, const N: usize> From<[(KP, V); N]> for Trie<K, V>
where
    K: Ord,
    KP: IntoIterator<Item = K>,
{
    /// Creates a [Trie] from an array of tuples. The tuples must contain
    /// a valid key element, that is, it can be converted to an iterator
    /// of the sub-element. For instance, for [str] this is usually [char].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie: Trie<char, i32> = Trie::from([
    ///     ("hello".chars(), 4)
    /// ]);
    ///
    /// assert_eq!(trie.len(), 1);
    /// assert_eq!(*trie.get("hello".chars()).unwrap(), 4);
    /// ```
    fn from(arr: [(KP, V); N]) -> Self {
        Self::from_iter(arr)
    }
}

#[cfg(test)]
mod tests {

    use super::Trie;

    #[test]
    pub fn trie_from_tuples() {
        let trie: Trie<char, i32> = Trie::from([("hello".chars(), 4)]);
        assert_eq!(trie.get("hello".chars()), Some(&4));
    }

    #[test]
    pub fn trie_disjoint_mut() {
        let mut tree = Trie::<char, usize>::new();

        tree.insert("hello".chars(), 3);
        tree.insert("world".chars(), 4);

        let mut keys = tree.get_disjoint_mut(["hello".chars(), "world".chars()]);

        **keys[0].as_mut().unwrap() = 4;
        **keys[1].as_mut().unwrap() = 2;

        assert_eq!(tree.get("hello".chars()), Some(&4));
        assert_eq!(tree.get("world".chars()), Some(&2));
    }

    #[test]
    pub fn trie_into_values() {
        let mut tree = Trie::<char, usize>::new();

        tree.insert("hello".chars(), 1);
        tree.insert("bye".chars(), 2);

        let mut values = tree.into_values();

        assert_eq!(values.next(), Some(1));
        assert_eq!(values.next(), Some(2));
        assert_eq!(values.next(), None);
    }

    #[test]
    pub fn basic_trie_insert() {
        let mut tree: Trie<char, &str> = Trie::new();
        tree.insert("test".chars(), "sample_1");
        assert!(tree.contains_key("test".chars()));
        assert_eq!(*tree.get("test".chars()).unwrap(), "sample_1");
    }

    #[test]
    pub fn basic_trie_insert_multi_keys() {
        let mut tree: Trie<char, &str> = Trie::new();
        tree.insert("test".chars(), "sample_1");
        // tree.insert("george".chars(), "sample_2");

        println!("INSERTING TEA...");
        tree.insert("tea".chars(), "sample_3");

        for (key, value) in tree.node.iter() {
            println!("({key:?}) -> {value:?}");
        }

        // assert!(tree.contains_key("test".chars()));
        // assert!(tree.contains_key("george".chars()));
        println!("\n\n\nStarting Tea...");
        assert!(tree.contains_key("tea".chars()));
        assert_eq!(*tree.get("test".chars()).unwrap(), "sample_1");
        assert_eq!(*tree.get("tea".chars()).unwrap(), "sample_3");
    }

    #[test]
    pub fn basic_trie_deletion() {
        let mut tree: Trie<char, &str> = Trie::new();
        tree.insert("test".chars(), "sample_1");

        assert_eq!(tree.len(), 1);

        assert!(tree.contains_key("test".chars()));

        println!("Helo");

        tree.remove("test".chars());
        println!("Deleetign");
        assert!(!tree.contains_key("test".chars()));
        assert_eq!(tree.len(), 0);
    }

    #[test]
    pub fn trie_deletion_multikey() {
        let mut tree: Trie<char, &str> = Trie::new();
        tree.insert("test".chars(), "sample_1");
        tree.insert("tea".chars(), "sample_2");

        assert!(tree.contains_key("test".chars()));
        assert!(tree.contains_key("tea".chars()));

        tree.remove("tea".chars());
        assert!(!tree.contains_key("tea".chars()));
        assert!(tree.contains_key("test".chars()));
    }

    #[test]
    pub fn test_arbitrary_insert() {
        let mut tree: Trie<char, String> = Trie::new();
        arbtest::arbtest(|u| {
            let key: String = u.arbitrary::<[char; 32]>().unwrap().iter().collect();
            let value: String = u.arbitrary::<[char; 32]>().unwrap().iter().collect();
            tree.insert(key.chars(), value);

            assert!(tree.contains_key(key.chars()));

            Ok(())
        });
    }

    #[test]
    pub fn completions_test() {
        let mut tree = Trie::<char, ()>::new();
        tree.insert("hello".chars(), ());
        tree.insert("hey".chars(), ());

        let mut values = tree
            .completions::<_, String>("he".chars())
            .into_iter()
            .collect::<Vec<_>>();

        values.sort();

        assert_eq!(values[0], "hello");
        assert_eq!(values[1], "hey");
    }

    #[test]
    pub fn test_arbitrary_deletion() {
        let mut tree: Trie<char, String> = Trie::new();
        arbtest::arbtest(|u| {
            let key: String = u.arbitrary::<[char; 4]>().unwrap().iter().collect();
            let value: String = u.arbitrary::<[char; 4]>().unwrap().iter().collect();

            assert_eq!(
                tree.contains_key(key.chars()),
                tree.remove(key.chars()).is_some()
            );

            tree.insert(key.chars(), value);

            assert_eq!(
                tree.contains_key(key.chars()),
                tree.remove(key.chars()).is_some()
            );

            Ok(())
        });
    }
}
