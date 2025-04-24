use std::{
    collections::VecDeque, fmt::Debug, hash::{BuildHasher, Hash, RandomState}, iter::{Peekable, Zip}, marker::PhantomData, ops::{Index, IndexMut}, vec
};

use node::Node;
use slotmap::{
    SlotMap,
    basic::{Values, ValuesMut},
};

mod iter;

mod node;

slotmap::new_key_type! { struct NodeKey; }

#[derive(Debug, Clone)]
/// A data strucur
pub struct Trie<K, V, S = RandomState> {
    /// The node pool, this is where the internal nodes are actually store. This
    /// improves cache locality and ease of access while limiting weird lifetime errors.
    node: SlotMap<NodeKey, Node<K, V, S>>,
    /// The root node of the pool. This is where things actually start searching.
    root: NodeKey,
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
    root: Option<NodeKey>,
    /// The first node in the walk.
    first: Option<K>,
    /// The remainder of the path as an iterator.
    remainder: &'a mut I,
}

#[derive(Debug)]
struct WalkTrajectory {
    path: Vec<NodeKey>,
    end: NodeKey,
}

impl<K, V> Default for Trie<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> Trie<K, V, S>
where
    S: BuildHasher,
{
    /// Creates a new [Trie] with no keys and records. This will
    /// create a [Trie] with a capacity of zero using the [Trie::with_capacity] method.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// assert_eq!(tree.len(), 0);
    /// ```
    pub fn new() -> Self
    where
        S: Default,
    {
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
    pub fn with_capacity(slots: usize) -> Self
    where
        S: Default,
    {
        let mut node = SlotMap::with_capacity_and_key(slots);
        let root = node.insert(Node::root());
        Self {
            node,
            root,
            size: 0,
        }
    }
    /// Gets the entrypoint of the tree by traversing the root. In the case
    /// where the iterator is empty, we will just return the root.
    fn get_entrypoint<I>(&self, key: &mut I) -> (Option<&NodeKey>, Option<K>)
    where
        I: Iterator<Item = K>,
        K: Hash + Eq,
    {
        match key.next() {
            Some(first) => (self.node[self.root].get(&first), Some(first)),
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
        K: Hash + Eq,
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

            let slot = self.node[*end].get(&current);

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
    fn lookup_key<I>(&self, key: I) -> Option<NodeKey>
    where
        I: IntoIterator<Item = K>,
        K: Hash + Eq,
    {
        Some(
            self.internal_walk(&mut key.into_iter().peekable(), false)
                .ok()?
                .end,
        )
    }

    /// Reconstructs a key by traversing up the [Trie] structure.
    /// 
    /// This will reconstruct it into a new type that can be created from
    /// an iterator of the node key types.
    fn reconstruct_node_key<J>(&self, key: NodeKey) -> Option<J>
    where 
        for<'a> J: FromIterator<&'a K>
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
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
    ///
    ///
    /// let mut values  = tree.values();
    ///
    /// assert_eq!(values.next().cloned(), Some(1));
    /// assert_eq!(values.next().cloned(), Some(2));
    /// assert_eq!(values.next().cloned(), None);
    /// ```
    pub fn values<'a>(&'a self) -> ValueIterRef<'a, K, V, S> {
        ValueIterRef {
            values: self.node.values(),
        }
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure mutably.
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
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
    pub fn values_mut<'a>(&'a mut self) -> ValueIterMut<'a, K, V, S> {
        ValueIterMut {
            values: self.node.values_mut(),
        }
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure mutably.
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
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
            .map(|(_, v)| v.into_value())
            .filter(Option::is_some)
            .collect::<Vec<Option<V>>>();

        ValueIter {
            inner: iter::Iter::new(values)
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
        for<'a> J: FromIterator<&'a K>
    {
        let values = self
            .node
            .iter()
            .filter(|(_, v)| v.value().is_some())
            .map(|(k, _)| k)
            .map(|k| self.reconstruct_node_key::<J>(k))            
            .collect::<Vec<Option<J>>>();

        KeyIter {
            inner: iter::Iter::new(values)
        }
    }


    fn collect_entry_partial<J>(&self) -> Vec<(J, NodeKey)>
    where
        for<'a> J: FromIterator<&'a K>
    {

        self
            .node
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
        for<'a> J: FromIterator<&'a K>
    {

        // Convert into an owned iterator.
        let values = self.collect_entry_partial::<J>()
            .into_iter()
            .map(|(key, value)| Some((key, self.node[value].value_mut().take().unwrap())))
            .collect::<Vec<_>>();
        EntryIter {
            inner: iter::Iter::new(values)
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
        for<'a> J: FromIterator<&'a K>
    {

        // Convert into an owned iterator.
        let values = self.collect_entry_partial::<J>()
            .into_iter()
            .map(|(key, value)| Some((key, self.node[value].value().as_ref().unwrap())))
            .collect::<Vec<_>>();
        EntryIter {
            inner: iter::Iter::new(values)
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
    pub fn entries_mut<'b, J>(&'b mut self) -> EntryIterMut<'b, K, V, S, J>
    where 
        for<'a> J: FromIterator<&'a K>
    {

        // let wow = self.node.iter_mut()
        //     .filter(|(k, value)| value.value().is_some())
        //     .map(|(k, v)| v.)

        // // Convert into an owned iterator.
        // let values = self.collect_entry_partial::<J>()
        //     .into_iter()
        //     .map(|(key, value)| Some((key, self.node[value].value_mut().as_mut().unwrap())))
        //     .collect::<Vec<_>>();

        // let values = self.collect_entry_partial::<J>()
        //     .into_iter()
        //     .map(|(key, value)| (key, self.node[value]))
        //     .collect::<Vec<_>>();



        let keys = self.node.iter().map(|(k, _)| self.reconstruct_node_key::<J>(k)).collect::<Option<Vec<_>>>().unwrap();



        EntryIterMut {
            inner: keys.into_iter().zip(self.node.iter_mut()),
            _type: PhantomData
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
    /// tree.put("hello".chars(), "world");
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), "world");
    /// ```
    pub fn get<I>(&self, key: I) -> Option<&V>
    where
        I: IntoIterator<Item = K>,
        K: Hash + Eq,
    {
        self.node[self.lookup_key(key)?].value().as_ref()
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
    /// tree.put("hello".chars(), 3);
    /// tree.put("world".chars(), 4);
    ///
    /// let keys = tree.get_disjoint_mut([ "hello".chars(), "world".chars() ]).unwrap();
    /// *keys[0] = 4;
    /// *keys[1] = 2;
    ///
    /// assert_eq!(tree.get("hello".chars()), Some(&4));
    /// assert_eq!(tree.get("world".chars()), Some(&2));
    /// ```
    pub fn get_disjoint_mut<'a, I, const N: usize>(
        &'a mut self,
        keys: [I; N],
    ) -> Option<[&'a mut V; N]>
    where
        I: IntoIterator<Item = K>,
        K: Hash + Eq,
    {
        // Perform key lookup.
        let mut node_keys: [NodeKey; N] = [NodeKey::default(); N];
        let mut i = 0;
        for value in keys.into_iter() {
            node_keys[i] = self.lookup_key(value)?;
            i += 1;
        }

        // Acquire the selection of disjoint keys.
        let result = self.node.get_disjoint_mut(node_keys)?;

        Some(
            result
                .into_iter()
                .map(|f| f.value_mut().as_mut().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .ok()?,
        )
    }
    /// Gets a mutable reference of a value from the [Trie] according to
    /// the key. The key is an iterable that can be used
    /// to access the record.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, &str>::new();
    /// tree.put("hello".chars(), "world");
    /// assert_eq!(*tree.get_mut("hello".chars()).unwrap(), "world");
    ///
    /// *tree.get_mut("hello".chars()).unwrap() = "world2";
    /// assert_eq!(*tree.get_mut("hello".chars()).unwrap(), "world2");
    /// ```
    pub fn get_mut<I>(&mut self, key: I) -> Option<&mut V>
    where
        I: IntoIterator<Item = K>,
        K: Hash + Eq,
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
    /// tree.put("hello".chars(), "world");
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
    /// tree.put("hello".chars(), "world");
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
    /// tree.put("hello".chars(), 0);
    /// assert!(!tree.is_empty());
    ///
    /// tree.clear();
    /// assert!(tree.is_empty());
    /// ```
    pub fn clear(&mut self)
    where
        S: Default,
    {
        self.node.clear();
        self.root = self.node.insert(Node::root());
        self.size = 0;
    }

    /// Reserves capacity for at least additional more elements to be inserted in the [Trie].
    /// The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&mut self, space: usize) {
        self.node.reserve(space);
    }
    /// Deletes a record from the [Trie] according to the
    /// key. It will return the old value if it is present within the
    /// data structure.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    /// tree.put("hello".chars(), 12);
    ///
    /// assert_eq!(tree.remove("hello".chars()).unwrap(), 12);
    /// ```
    pub fn remove<I>(&mut self, master: I) -> Option<V>
    where
        I: IntoIterator<Item = K>,
        K: Hash + Eq + Copy,
        S: Default,
    {
        let trajectory = self
            .internal_walk(master.into_iter().peekable().by_ref(), true)
            .ok()?;

        let mut first = false;
        let mut value = None;

        let mut previous = None;
        let mut previous_key = None;
        for iter in trajectory.path.iter().rev() {
            let k_temp = self.node[*iter].key().clone();

            if previous.is_some() {
                self.node.remove(previous.unwrap());
            }

            if !first {
                value = self.node[*iter].value_mut().take();

                first = true;
            } else if self.node[*iter].sub_key_len() == 1 && !self.node[*iter].is_root() {
                self.node.remove(*iter);
            } else {
                self.node[*iter].remove(&previous_key.unwrap());
                break; // we are back in a consistent state.
            }

            previous = Some(*iter);
            previous_key = k_temp;
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
        K: Hash + Eq + Debug,
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
    /// tree.put("hello".chars(), 12);
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
    /// tree.put("hello".chars(), 1);
    ///
    /// // Verify the key is in the tree.
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), 1);
    ///
    /// // Verify the key replacement.
    /// assert_eq!(tree.put("hello".chars(), 2).unwrap(), 1);
    /// ```
    pub fn put<I>(&mut self, master: I, value: V) -> Option<V>
    where
        I: IntoIterator<Item = K>,
        K: Clone + Hash + Eq,
        S: Default,
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
                    let new_node = Node::keyed(first.clone().unwrap(), self.root);
            
                    let new_key = self.node.insert(new_node);
                    self.node[self.root]
                        
                        .insert(first.clone().unwrap(), new_key);
                    new_key
                };

                let mut nk = None;
                for item in remainder {
                    nk = Some(self.node.insert(Node::keyed(item.clone(), previous)));
                    self.node[previous].insert(item, nk.unwrap());

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



// impl<K, V, S> PartialEq for Trie<K, V, S> {
//     fn eq(&self, other: &Self) -> bool {

//         true
//     }
// }

impl<K, V, S, I> Index<I> for Trie<K, V, S>
where
    K: Hash + Eq,
    I: IntoIterator<Item = K>,
    S: BuildHasher + Default,
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

impl<K, V, S, I> IndexMut<I> for Trie<K, V, S>
where
    K: Hash + Eq,
    I: IntoIterator<Item = K>,
    S: BuildHasher + Default,
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

/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct EntryIterMut<'a, K, V, S, J> {
    inner: Zip<vec::IntoIter<J>, slotmap::basic::IterMut<'a, NodeKey, Node<K, V, S>>>,
    _type: PhantomData<J>
}




/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct EntryIter<J, V> {
    inner: crate::iter::Iter<(J, V)>
}


/// An iterator created by consuming the [Trie].
///
/// Contains all the keys of the [Trie].
pub struct KeyIter<J> {
    inner: crate::iter::Iter<J>
}


/// An iterator created by consuming the [Trie].
///
/// Contains all the elements of the [Trie].
pub struct ValueIter<V> {
    inner: crate::iter::Iter<V>
}

/// An iterator over the values of a [Trie].
pub struct ValueIterRef<'a, K, V, S> {
    values: Values<'a, NodeKey, Node<K, V, S>>,
}

/// An iterator over the values of a [Trie]
/// that provides mutable references.
pub struct ValueIterMut<'a, K, V, S> {
    values: ValuesMut<'a, NodeKey, Node<K, V, S>>,
}

impl<'a, K, V, S, J> Iterator for EntryIterMut<'a, K, V, S, J> {
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
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
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
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
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

impl<'a, K, V, S> Iterator for ValueIterMut<'a, K, V, S> {
    type Item = &'a mut V;

    /// Iterates over the values mutably within a [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
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
            let node = self.values.next()?;
            current = node.value_mut().as_mut();
        }

        current
    }
}

impl<'a, K, V, S> Iterator for ValueIterRef<'a, K, V, S> {
    type Item = &'a V;

    /// Iterates over the values within a [Trie].
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::new();
    ///
    /// tree.put("hello".chars(), 1);
    /// tree.put("bye".chars(), 2);
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
            let node = self.values.next()?;
            current = node.value().as_ref();
        }

        current
    }
}

impl<KP, K, V, S> Extend<(KP, V)> for Trie<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
    K: Hash + Eq + Clone,
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
            self.put(key, value);
        }
    }
}

impl<KP, K, V, S> FromIterator<(KP, V)> for Trie<K, V, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Default,
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

impl<KP, K, V, S, const N: usize> From<[(KP, V); N]> for Trie<K, V, S>
where
    K: Eq + Hash + Clone,
    KP: IntoIterator<Item = K>,
    S: BuildHasher + Default,
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

    use std::hash::RandomState;


    use super::Trie;


    #[test]
    pub fn trie_from_tuples() {
        let trie: Trie<char, i32> = Trie::from([("hello".chars(), 4)]);
        assert_eq!(trie.get("hello".chars()), Some(&4));
    }

    #[test]
    pub fn trie_into_values() {
        let mut tree = Trie::<char, usize>::new();

        tree.put("hello".chars(), 1);
        tree.put("bye".chars(), 2);

        let mut values = tree.into_values();

        assert_eq!(values.next(), Some(1));
        assert_eq!(values.next(), Some(2));
        assert_eq!(values.next(), None);
    }

    #[test]
    pub fn basic_trie_insert() {
        let mut tree: Trie<char, &str, RandomState> = Trie::new();
        tree.put("test".chars(), "sample_1");
        assert!(tree.contains_key("test".chars()));
        assert_eq!(*tree.get("test".chars()).unwrap(), "sample_1");
    }

    #[test]
    pub fn basic_trie_insert_multi_keys() {
        let mut tree: Trie<char, &str, RandomState> = Trie::new();
        tree.put("test".chars(), "sample_1");
        // tree.put("george".chars(), "sample_2");

        println!("INSERTING TEA...");
        tree.put("tea".chars(), "sample_3");

        for (key, value) in &tree.node {
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
        tree.put("test".chars(), "sample_1");

        assert_eq!(tree.len(), 1);

        assert!(tree.contains_key("test".chars()));

        tree.remove("test".chars());
        assert!(!tree.contains_key("test".chars()));
        assert_eq!(tree.len(), 0);
    }

    #[test]
    pub fn trie_deletion_multikey() {
        let mut tree: Trie<char, &str> = Trie::new();
        tree.put("test".chars(), "sample_1");
        tree.put("tea".chars(), "sample_2");

        assert!(tree.contains_key("test".chars()));
        assert!(tree.contains_key("tea".chars()));

        tree.remove("tea".chars());
        assert!(!tree.contains_key("tea".chars()));
        assert!(tree.contains_key("test".chars()));
    }

    // #[test]
    // pub fn parse_trie() {
    //     let mut tree: Trie<char, &'static str> = Trie::new();

    //     tree.put("test".chars(), "obama");
    //     tree.put("tesla".chars(), "obama3");

    //     println!("INITIAL:\n");
    //     for (key, value) in &tree.node {
    //         println!("({key:?}) -> {value:?}");
    //     }

    //     // tree.delete("test", |f| f.chars().collect());
    //     // tree.delete("tesla", |f| f.chars().collect());

    //     for (key, value) in &tree.node {
    //         println!("({key:?}) -> {value:?}");
    //     }

    //     let result = tree.get("test".chars());
    //     // let result2 = tree.get("tesla", |f| f.chars().collect());

    //     println!("Hello: {:?}", result);
    //     // println!("Structure: {:?}", tree.node);

    //     panic!("Ye");
    // }
}
