use std::{
    collections::VecDeque,
    fmt::Debug,
    iter::{Peekable, Zip},
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::NonNull,
    vec::{IntoIter, Vec},
};

use list::{NodeIndex, NodeIterMut, Slots};
use node::Node;

mod iter;
mod node;

mod list;

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
struct WalkFailure<'a, I> {
    /// The root of the walk.
    root: NodeIndex,
    /// The first node in the walk.
    // first: Option<K>,
    // first: K,
    /// The remainder of the path as an iterator.
    remainder: &'a mut I,
}

#[derive(Debug)]
struct WalkTrajectory {
    path: Vec<NodeIndex>,
    end: NodeIndex,
}

impl<K, V> Default for Trie<K, V> {
    /// Creates an empty [Trie] with the [Default] value for
    /// the hasher.
    ///
    /// # Example
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, ()>::default();
    /// assert_eq!(trie.capacity(), 0);
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct WalkCtx {
    pending: Vec<Vec<NodeIndex>>,
}

struct WalkIter<'a, K, V> {
    trie: &'a Trie<K, V>,
    context: WalkCtx,
}

impl WalkCtx {
    /// Drives this forward, this is similar to the
    pub fn drive<K, V>(&mut self, trie: &Trie<K, V>) -> Option<(NodeIndex, Vec<NodeIndex>)> {
        while let Some(pending) = self.pending.pop() {
            let current = pending.last().unwrap();
            if trie.node[*current].value().is_some() {
                // We have reached a root.
                return Some((*current, pending));
            }

            for &child in trie.node[*current].sub_keys.iter().rev() {
                // Create a new path.
                let mut new_path = pending.clone();
                new_path.push(child);
                self.pending.push(new_path);
            }
        }

        None
    }
}

impl<'a, K, V> Iterator for WalkIter<'a, K, V> {
    type Item = (NodeIndex, Vec<NodeIndex>);
    fn next(&mut self) -> Option<Self::Item> {
        self.context.drive(self.trie)
    }
}

// The following are the internal 'developer' functions of the [Trie].
impl<K, V> Trie<K, V> {
    fn collect_path_keys<'a, J>(&'a self, nodes: &[NodeIndex]) -> J
    where 
        J: FromIterator<&'a K>
    {
        nodes
            .iter()
            .map(|f| self.node[*f].key())
            .filter(|f| f.is_some())
            .map(|f| f.as_ref().unwrap())
            .collect()
    }


    /// Performs an internal wlak of the [Trie] but
    /// receives the last node.
    fn internal_walk_with_index<'a, I>(
        &self,
        remainder: &'a mut Peekable<I>,
    ) -> Result<NodeIndex, WalkFailure<'a, Peekable<I>>>
    where
        I: Iterator<Item = K>,
        K: Ord,
    {
        self.internal_walk_with_fn(remainder, |_| {})
    }

    /// Performs an internal walk of the [Trie] and obtains the path
    /// used to get there. Internally, this just calls the internal walk
    /// function and has it collect the nodes as we go.
    fn internal_walk_with_path<'a, I>(
        &self,
        remainder: &'a mut Peekable<I>,
    ) -> Result<WalkTrajectory, WalkFailure<'a, Peekable<I>>>
    where
        I: Iterator<Item = K>,
        K: Ord,
    {
        self.internal_walk_with_path_fn(remainder, |_| {})
    }

    /// Performs an internal walk of the [Trie] and obtains the path
    /// used to get there. Internally, this just calls the internal walk
    /// function and has it collect the nodes as we go.
    fn internal_walk_with_path_fn<'a, I, F>(
        &self,
        remainder: &'a mut Peekable<I>,
        mut functor: F,
    ) -> Result<WalkTrajectory, WalkFailure<'a, Peekable<I>>>
    where
        I: Iterator<Item = K>,
        K: Ord,
        F: FnMut(NodeIndex),
    {
        let mut trajectory_path = vec![];

        let end = self.internal_walk_with_fn(remainder, |node| {
            trajectory_path.push(node);
            functor(node);
        })?;

        Ok(WalkTrajectory {
            path: trajectory_path,
            end,
        })
    }

    /// Collects the key
    fn key_collect<'a>(&'a self, node: NodeIndex, array: &mut Vec<&'a K>) {
        if let Some(inner) = self.node[node].key() {
            array.push(inner);
        }
    }

    fn internal_walk_collect_key<'a, I, J>(
        &self,
        remainder: &'a mut Peekable<I>,
    ) -> Result<(J, NodeIndex), WalkFailure<'a, Peekable<I>>>
    where
        I: Iterator<Item = K>,
        K: Ord,
        for<'b> J: FromIterator<&'b K>,
    {
        let mut collector = vec![];
        let index =
            self.internal_walk_with_fn(remainder, |nk| self.key_collect(nk, &mut collector))?;
        Ok((collector.into_iter().collect::<J>(), index))
    }

    fn perform_walk(&self, root: NodeIndex) -> WalkCtx {
        // self.walk_path(self.root);
        WalkCtx {
            pending: vec![vec![root]],
        }
    }

    // fn walk_path(&self, current: NodeIndex) {

    //     println!("Walking from {current:?}");
    //     if self.node[current].value().is_some() {
    //         println!("TERMINAL");
    //     }
    //     for key in &self.node[current].sub_keys {
    //         println!("({current:?}) --> ({key:?})");
    //         self.walk_path(*key);
    //     }

    // }

    /// Performs an internal walk of the [Trie]. This is the most important function
    /// for the operation of the [Trie]. It takes in a visitor function that is called
    /// every time a nodes key is accessed, this allows for activities such as collecting
    /// the key.
    ///
    /// If the walk is fully succesful, as in we can propery traverse the path to the end, we
    /// return the [NodeIndex] of the last node.
    ///
    /// If the walk cannot be completed, we return a [WalkFailure] struct that contains the necessary
    /// information to perform insertion.
    fn internal_walk_with_fn<'a, I, F>(
        &self,
        remainder: &'a mut Peekable<I>,
        mut visitor_fn: F,
    ) -> Result<NodeIndex, WalkFailure<'a, Peekable<I>>>
    where
        I: Iterator<Item = K>,
        K: Ord,
        F: FnMut(NodeIndex),
    {
        // The root for the path.
        visitor_fn(self.root);

        let mut end = &self.root;
        loop {
            let Some(current) = remainder.peek() else {
                break;
            };

            // Call on the current node.
            visitor_fn(*end);

            if let Some(slot) = self.node[*end].get(&current, &self.node) {
                end = slot;
            } else {
                return Err(WalkFailure {
                    root: *end,
                    remainder,
                });
            }

            // Actually consume.
            remainder.next();
        }

        // Call on the very last index, if we reach this point this is indicatvie
        // of a succesful complete walk.
        visitor_fn(*end);

        Ok(*end)
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
    /// Returns the capacity of the underlying arena allocator.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, ()>::with_capacity(3);
    /// assert_eq!(trie.capacity(), 3);
    /// ```
    pub fn capacity(&self) -> usize {
        self.node.slots.capacity()
    }
    /// Creates a new [Trie] with a certain specified capacity.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, usize>::with_capacity(20);
    /// assert!(tree.is_empty());
    /// ```
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            node: Slots::with_capacity(slots),
            root: NodeIndex { position: 0 },
            size: 0,
        }
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
        self.skip_trie_consistency()?;
        Some(
            self.internal_walk_with_index(&mut key.into_iter().peekable())
                .ok()?,
        )
    }

    /// Checks if a key is a prefix within a [Trie]. This
    /// may or may not be an exact math.
    /// 
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    /// 
    /// let mut trie = Trie::<char, ()>::new();
    /// trie.insert("hello".chars(), ());
    /// trie.insert("hey".chars(), ());
    /// trie.insert("hero".chars(), ());
    /// 
    /// assert!(trie.is_prefix(['h', 'e']));
    /// assert!(trie.is_prefix(['h', 'e', 'r']));
    /// assert!(!trie.is_prefix(['h', 'e', 'r', 'y']));
    /// ```
    pub fn is_prefix<I>(&self, key: I) -> bool
    where 
        I: IntoIterator<Item = K>,
        K: Ord
    {
        self.skip_trie_consistency();
        self.internal_walk_with_index(&mut key.into_iter().peekable())
            .is_ok()

    }
    /// Gets all the completions of a key. This will
    /// traverse the tree to the point at which the key
    /// diverges. This is also called a 'common prefix search'.
    ///
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, ()>::new();
    /// tree.insert("hello".chars(), ());
    /// tree.insert("hey".chars(), ());
    /// tree.insert("james".chars(), ());
    ///
    ///
    /// println!("Tree: {:?}", tree);
    ///
    /// let mut values = tree.completions::<_, String>("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<_>>();
    /// assert_eq!(values.len(), 2);
    ///
    ///
    /// assert_eq!(values[0].0, "hello");
    /// assert_eq!(values[1].0, "hey");
    /// ```
    pub fn completions<I, J>(&self, key: I) -> CompletionIter<'_, K, V, J>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
        J: FromIterator<K>,
    {
        let mut collector = vec![];
        match self.internal_walk_with_fn(key.into_iter().peekable().by_ref(), |nk| {
            self.key_collect(nk, &mut collector)
        }) {
            Ok(inner) => CompletionIter {
                beginning: collector,
                inner: self.perform_walk(inner),
                trie: self,
                _transform: PhantomData,
            },
            Err(_) => {
                // Here we will create something that just will not iterate.
                CompletionIter {
                    beginning: vec![],
                    _transform: PhantomData,
                    inner: WalkCtx { pending: vec![] },
                    trie: self,
                }
            }
        }
    }

    /// Performs a postfix search of the tree, returning
    /// the postfixes.
    ///
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut tree = Trie::<char, ()>::new();
    /// tree.insert("hello".chars(), ());
    /// tree.insert("hey".chars(), ());
    /// tree.insert("james".chars(), ());
    ///
    /// let mut values = tree.postfix_search::<_, String>("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<_>>();
    /// assert_eq!(values.len(), 2);
    ///
    ///
    /// assert_eq!(values[0].0, "llo");
    /// assert_eq!(values[1].0, "y");
    /// ```
    pub fn postfix_search<I, J>(&self, key: I) -> PostfixIter<'_, K, V, J>
    where
        I: IntoIterator<Item = K>,
        K: Ord,
        J: FromIterator<K>,
    {
        let mut collector = vec![];
        match self.internal_walk_with_fn(key.into_iter().peekable().by_ref(), |nk| {
            self.key_collect(nk, &mut collector)
        }) {
            Ok(inner) => PostfixIter {
                inner: self.perform_walk(inner),
                trie: self,
                _transform: PhantomData,
            },
            Err(_) => {
                // Here we will create something that just will not iterate.
                PostfixIter {
                    _transform: PhantomData,
                    inner: WalkCtx { pending: vec![] },
                    trie: self,
                }
            }
        }
    }



    /// Returns an iterator over the values of the [Trie]
    /// data structure.
    ///
    /// # Examples
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
    pub fn values(&self) -> ValueIterRef<'_, K, V> {
        ValueIterRef {
            values: self.node.slots.iter(),
        }
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure mutably.
    ///
    /// # Examples
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
    pub fn values_mut(&mut self) -> ValueIterMut<'_, K, V> {
        ValueIterMut {
            values: self.node.slots.iter_mut(),
        }
    }

    /// Clears the [Trie], returning all key-value pairs as an
    /// iterator. Keeps the allocated memory for reuse.
    /// TODO: Is the root re-added after it is dropped?
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert(['a', 'b', 'c'], 1);
    /// trie.insert(['a', 'c'], 2);
    /// trie.insert(['b'], 3);
    ///
    /// let mut iter = trie.drain::<String>();
    /// assert_eq!(iter.next(), Some(("abc".to_string(), 1)));
    /// assert_eq!(iter.next(), Some(("ac".to_string(), 2)));
    /// assert_eq!(iter.next(), Some(("b".to_string(), 3)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn drain<J>(&mut self) -> Drain<'_, K, V, J>
    where
        for<'b> J: FromIterator<&'b K>,
    {
        let keys = self.keys::<J>().collect::<Vec<_>>();

        Drain {
            key_iter: keys.into_iter(),
            inner: self.node.slots.drain(0..self.node.slots.len()),
        }
    }

    /// Returns an iterator over the values of the [Trie]
    /// data structure mutably.
    ///
    ///
    /// # Examples
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
    ///
    /// # Examples
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
    /// assert_eq!(key_iter.next().unwrap(), "bye");
    /// assert_eq!(key_iter.next().unwrap(), "hello");
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn keys<'a, J>(&'a self) -> KeyIter<'a, K, V, J>
    where
        J: FromIterator<&'a K>,
    {
        KeyIter {
            inner: WalkIter {
                context: self.perform_walk(self.root),
                trie: self,
            },
            _type: PhantomData,
        }
    }

    /// Returns an iterator over the entries of the [Trie]
    /// data structure.
    ///
    ///
    /// # Examples
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
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), 3));
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), 4));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn into_entries<J>(self) -> IntoEntryIter<K, V, J>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        IntoEntryIter {
            inner: self.perform_walk(self.root),
            trie: self,
            _type: PhantomData
        }
    }
    /// Returns an iterator over the entries of the [Trie]
    /// data structure.
    ///
    ///
    /// # Examples
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
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), &3));
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), &4));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn entries<'a, J>(&'a self) -> EntryIterRef<'a, K, V, J>
    where
        J: FromIterator<&'a K>,
    {
        EntryIterRef {
            inner: WalkIter {
                context: self.perform_walk(self.root),
                trie: self
            },
            _type: PhantomData
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
    /// let mut key_iter = tree.entries_mut::<String>();
    ///
    /// assert_eq!(*key_iter.next().unwrap().1, 3);
    /// assert_eq!(*key_iter.next().unwrap().1, 4);
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn entries_mut<J>(&mut self) -> EntryIterMut<'_, K, V, J>
    where
        for<'a> J: FromIterator<&'a K>,
    {
        EntryIterMut {
            inner: self.perform_walk(self.root),
            trie: self,
            // slot: None,.
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
                    let fetched = self.node[inner].value_mut().as_mut().unwrap() as *mut V;

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
        self.size = 0;
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert(['a', 'b', 'c'], 1);
    ///
    /// assert_eq!(trie.get_key_value(['a', 'b', 'c']), Some(("abc".to_string(), &1)))
    ///
    /// ```
    pub fn get_key_value<I, J>(&self, key: I) -> Option<(J, &V)>
    where
        I: IntoIterator<Item = K>,
        for<'a> J: FromIterator<&'a K>,
        K: Ord,
    {
        self.skip_trie_consistency()?;
        let (key, value) = self
            .internal_walk_collect_key(&mut key.into_iter().peekable())
            .ok()?;
        Some((key, self.node[value].value().as_ref().unwrap()))
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert(['a'], 1);
    ///
    /// assert_eq!(trie.get_key_value_mut(['a']), Some(("a".to_string(), &mut 1)));
    ///
    /// *trie.get_key_value_mut::<_, String>(['a']).as_mut().unwrap().1 = 2;
    /// assert_eq!(trie.get_key_value_mut::<_, String>(['a']), Some(("a".to_string(), &mut 2)));
    ///
    ///
    /// ```
    pub fn get_key_value_mut<I, J>(&mut self, key: I) -> Option<(J, &mut V)>
    where
        I: IntoIterator<Item = K>,
        for<'a> J: FromIterator<&'a K>,
        K: Ord,
    {
        self.skip_trie_consistency()?;
        let (key, value) = self
            .internal_walk_collect_key(&mut key.into_iter().peekable())
            .ok()?;
        Some((key, self.node[value].value_mut().as_mut().unwrap()))
    }

    /// Removes a key from the [Trie], returning the computed key and value
    /// if the key was previously in the map.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert(['a'], 2);
    ///
    /// assert_eq!(trie.remove_entry::<_, String>(['a']), Some(("a".to_string(), 2)));
    ///
    /// ```
    ///
    pub fn remove_entry<I, J>(&mut self, key: I) -> Option<(J, V)>
    where
        I: IntoIterator<Item = K>,
        for<'a> J: FromIterator<&'a K>,
        K: Ord,
    {
        self.skip_trie_consistency()?;

        let mut path = vec![];
        let trajectory = self
            .internal_walk_with_path_fn(key.into_iter().peekable().by_ref(), |nk| {
                self.key_collect(nk, &mut path)
            })
            .ok()?;

        let mut traj_path = trajectory.path.iter().rev().peekable();

        if let Some(inner) = traj_path.peek() {
            let reconstruction = path.into_iter().collect::<J>();
            let value = self.remove_post_walk(trajectory.path.iter().rev())?;
            Some((reconstruction, value))
        } else {
            self.remove_post_walk(trajectory.path.iter().rev())?;
            None
        }
    }

    /// Reserves capacity for at least additional more elements to be inserted in the [Trie].
    /// The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&mut self, space: usize) {
        self.node.slots.reserve(space);
    }
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
        self.skip_trie_consistency()?;
        let trajectory = self
            .internal_walk_with_path(master.into_iter().peekable().by_ref())
            .ok()?;

        self.remove_post_walk(trajectory.path.iter().rev())
    }

    /// Performs the actual removal from the [Trie]. This will
    /// traverse back from the node using the calculated trajectory
    /// in order to remove the node from the [Trie].
    ///
    /// This method exists to allow us to easily implement remove entry
    /// without having to walk the tree a second time to reconstruct the [Trie].
    fn remove_post_walk<'a, T>(&'a mut self, path: T) -> Option<V>
    where
        T: Iterator<Item = &'a NodeIndex>,
        K: Ord,
    {
        let mut first = false;
        let mut value = None;

        let mut previous = None;
        for iter in path {
            if let Some(previous) = previous {
                self.node.remove(previous);
            }

            if !first {
                value = self.node[*iter].value_mut().take();

                first = true;
            } else if self.node[*iter].sub_key_len() == 1 && !self.node[*iter].is_root() {
                self.node.remove(*iter);
            } else {
                remove_node_subkey_by_key(*iter, *previous.as_ref().unwrap(), &mut self.node);
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

    /// If we do not have a root then we should just return.
    fn skip_trie_consistency(&self) -> Option<()> {
        if self.node.slots.is_empty() {
            None
        } else {
            Some(())
        }
    }

    fn make_trie_consistent(&mut self) {
        if self.node.slots.is_empty() {
            // No nodes.
            self.node.slots.push(Some(Node::root()));
        }
    }

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
        K: Debug,
        K: Ord,
    {
        self.make_trie_consistent();

        self.size += 1;

        match self.internal_walk_with_index(master.into_iter().peekable().by_ref()) {
            Ok(v) => {
                let current = self.node[v].value_mut().take();
                *self.node[v].value_mut() = Some(value);
                current
            }
            Err(WalkFailure {
                mut root,
                remainder,
            }) => {
                let mut nk = Some(root);
                for item in remainder {
                    nk = Some(self.node.insert(Node::keyed(item, root)));
                    insert_node_subkey(root, nk.unwrap(), &mut self.node);
                    root = nk.unwrap();
                }

                *self.node[nk.unwrap()].value_mut() = Some(value);

                None
            }
        }
        // }
    }
    /// This will remove any unused space in the underlying arena allocator and
    /// then shrink the arena to fit this data. This can help improve cache locality.
    ///
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert(['a'], 1);
    /// trie.insert(['d'], 2);
    /// trie.insert(['b'], 3);
    /// trie.insert(['b', 'c'], 4);
    ///
    /// trie.remove(['d']);
    /// trie.shrink_to_fit();
    ///
    /// assert_eq!(trie.get(['a']), Some(&1));
    /// assert_eq!(trie.get(['d']), None);
    /// assert_eq!(trie.get(['b']), Some(&3));
    /// assert_eq!(trie.get(['b', 'c']), Some(&4));
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.node.defragment();
        self.node.shrink_to_fit();
    }
}

// pub struct LazyKey<'a, K, J, V>
// where
//     for<'b> J: FromIterator<&'b K>
// {
//     inner: LazyKeyInner<'a, K, J, V>
// }

// enum LazyKeyInner<'a, K, J, V> {
//     Resolved(J),
//     Plan {
//         sequence: Vec<NodeIndex>,
//         map: &'a Trie<K, V>
//     }
// }

// impl<'a, K, J, V> LazyKeyInner<'a, K, J, V> {
//     pub fn key(self) -> LazyKeyInner<'a, K, J, V> {
//         match self {
//             LazyKeyInner::Resolved(inner) => Self::Resolved(inner),
//             LazyKeyInner::Plan { sequence, map } => {
//                 LazyKeyInner::Resolved(map.no)
//             }
//         }
//     }
// }

#[derive(Debug)]
pub struct CompletionIter<'a, K, V, J> {
    trie: &'a Trie<K, V>,
    /// The root path.
    beginning: Vec<&'a K>,
    /// The list of completions, this functions as a bump
    /// allocator and is used to build out the list.
    inner: WalkCtx,
    /// The value to which the iterators will be transformed during iteration.
    _transform: PhantomData<J>,
}

#[derive(Debug)]
pub struct PostfixIter<'a, K, V, J> {
    trie: &'a Trie<K, V>,
    /// The list of completions, this functions as a bump
    /// allocator and is used to build out the list.
    inner: WalkCtx,
    /// The value to which the iterators will be transformed during iteration.
    _transform: PhantomData<J>,
}

fn insert_node_subkey<K: Ord, V>(
    source: NodeIndex,
    value: NodeIndex,
    buffer: &mut Slots<K, V>,
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
    let result = buffer[source]
        .sub_keys
        .iter()
        .position(|s| *s == to_remove)?;

    let elem = buffer[source].sub_keys.remove(result);

    Some(elem)
}

impl<I, K, V> Index<I> for Trie<K, V>
where
    I: IntoIterator<Item = K>,
    K: Ord,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    /// Panics if the key is not present in the [Trie].
    ///
    /// # Example
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert(['a'], 1);
    ///
    /// assert_eq!(trie[['a']], 1);
    ///
    /// ```
    fn index(&self, index: I) -> &Self::Output {
        self.get(index).as_ref().unwrap()
    }
}

impl<'a, K: Debug, V, J> Iterator for CompletionIter<'a, K, V, J>
where
    J: FromIterator<&'a K>,
{
    type Item = (J, &'a V);

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
    /// assert_eq!(values[0].0, "hello");
    /// assert_eq!(values[1].0, "hey");
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (current, path) = self.inner.drive(&self.trie)?;

        let key = assemble_completion(&self.trie, &self.beginning, path);

        Some((key, self.trie.node[current].value().as_ref().unwrap()))
    }
}

impl<'a, K: Debug, V, J> Iterator for PostfixIter<'a, K, V, J>
where
    J: FromIterator<&'a K>,
{
    type Item = (J, &'a V);

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
    /// let mut values = tree.postfix_search::<_, String>("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<_>>();
    /// assert_eq!(values.len(), 2);
    ///
    ///
    /// assert_eq!(values[0].0, "llo");
    /// assert_eq!(values[1].0, "y");
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (current, path) = self.inner.drive(&self.trie)?;

        let key = path
            .iter()
            .map(|f| self.trie.node[*f].key())
            .filter(|f| f.is_some())
            .skip(1)
            .map(|f| f.as_ref().unwrap());
        Some((
            key.collect(),
            self.trie.node[current].value().as_ref().unwrap(),
        ))
    }
}

fn assemble_completion<'a, K, V, J>(
    trie: &'a Trie<K, V>,
    beginning: &Vec<&'a K>,
    walked: Vec<NodeIndex>,
) -> J
where
    J: FromIterator<&'a K>,
{
    let tail_end = walked
        .into_iter()
        .map(|nk| trie.node[nk].key())
        .filter(|f| f.is_some())
        .map(|f| f.as_ref().unwrap());

    let actual_key = beginning
        .iter()
        // The following lines just omit the very last element.
        .rev()
        .skip(1)
        .rev()
        .map(|k| *k)
        .chain(tail_end)
        .collect::<J>();

    actual_key
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
    // value: ValueIterMut<'a, K, V>,
    trie: &'a mut Trie<K, V>,
    inner: WalkCtx,
    _type: PhantomData<J>,
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct IntoEntryIter<K, V, J> {
    trie: Trie<K, V>,
    inner: WalkCtx,
    _type: PhantomData<J>
}


/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct EntryIterRef<'a, K, V, J> {
    inner: WalkIter<'a, K, V>,
    _type: PhantomData<J>
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the keys of the [Trie].
pub struct KeyIter<'a, K, V, J> {
    inner: WalkIter<'a, K, V>,
    _type: PhantomData<J>,
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
    values: std::slice::IterMut<'a, Option<Node<K, V>>>,
}

// impl<'a, K, V, J> Iterator for EntryIter<'a, K, V, J>
// where 
//     J: FromIterator<&'a K>
// {
//     type Item = (J, &'a V);

impl<K, V, J> Iterator for IntoEntryIter<K, V, J>
where 
     for<'b> J: FromIterator<&'b K>,
     
{
    type Item = (J, V);

    /// Iterates over the mutably borrowed entries within a [Trie].
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
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), 3));
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), 4));
    /// assert!(key_iter.next().is_none());
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (current, path) =  { self.inner.drive(&self.trie)? };

        
        let calced = self.trie.collect_path_keys::<J>(&path);
        


       
       Some((calced, self.trie.node[current].value_mut().take().unwrap()))
    }
}


impl<'a, K: Clone, V, J> Iterator for EntryIterMut<'a, K, V, J>
where 
     for<'b> J: FromIterator<&'b K>,
     
{
    type Item = (J, ValueSlot<'a, K, V>);

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
    /// assert_eq!(*key_iter.next().unwrap().1, 3);
    /// assert_eq!(*key_iter.next().unwrap().1, 4);
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
    /// assert_eq!(*key_iter.next().unwrap().1, 4);
    /// assert!(key_iter.next().is_none());
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (current, path) =  { self.inner.drive(&self.trie)? };

        
        let calced = self.trie.collect_path_keys::<J>(&path);
        


        // SAFETY: The borrow checker does not know that subsequent calls return distinct
        // nodes, and thus the aliasing rules are upheld and we only have a single mutable reference at a time.
        let candidate = unsafe { &mut *(&mut self.trie.node[current] as *mut Node<K, V>)  };


   
        Some((calced, ValueSlot {
            node: candidate
        }))
    }
}

pub struct ValueSlot<'a, K, V> {
    node: &'a mut Node<K, V>
}

impl<'a, K, V> Deref for ValueSlot<'a, K, V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        self.node.value().as_ref().unwrap()
    }
}

impl<'a, K, V> DerefMut for ValueSlot<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.node.value_mut_unchecked()
    }
}


impl<K, V> PartialEq for Trie<K, V>
where
    K: Eq,
{
    /// Checks if two [Trie] are equal.
    fn eq(&self, other: &Self) -> bool {
        for node in self
            .node
            .slots
            .iter()
            .filter(|f| f.is_some())
            .map(|f| f.as_ref().unwrap())
        {
            if !other
                .node
                .slots
                .iter()
                .filter_map(|f| f.as_ref())
                .any(|o_node| {
                    node.key() == o_node.key() // make the keys equal.
                && check_node_key_equivalences(node, o_node, &self.node, &other.node)
                && check_node_key_equivalences(o_node, node, &other.node, &self.node)
                })
            {
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
    pool_b: &Slots<K, V>,
) -> bool
where
    K: Eq,
{
    node_a.sub_keys.iter().all(|key_a| {
        // For every a key.
        let value = pool_a[*key_a].key();
        node_b
            .sub_keys
            .iter()
            .any(|key_b| pool_b[*key_b].key() == value)
    });

    true
}

impl<'a, K, V, J> Iterator for EntryIterRef<'a, K, V, J>
where 
    J: FromIterator<&'a K>
{
    type Item = (J, &'a V);

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
    /// 
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), 3));
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), 4));
    /// assert!(key_iter.next().is_none());
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (curr, path) = self.inner.next()?;

        Some((self.inner.trie.collect_path_keys(&path), self.inner.trie.node[curr].value().as_ref().unwrap()))
    }
}

impl<'a, K, V, J> Iterator for KeyIter<'a, K, V, J>
where
    J: FromIterator<&'a K>,
{
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
    /// assert_eq!(values.next().unwrap(), "bye");
    /// assert_eq!(values.next().unwrap(), "hello");
    /// assert_eq!(values.next().as_ref(), None);
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (_, path) = self.inner.next()?;
        Some(self.inner.trie.collect_path_keys(&path))
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
    K: Ord + Debug,
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

impl<'a, KP, K, V> Extend<(KP, &'a V)> for Trie<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone,
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
    fn extend<T: IntoIterator<Item = (KP, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value.clone());
        }
    }
}

impl<KP, K, V> FromIterator<(KP, V)> for Trie<K, V>
where
    K: Ord + Debug,
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
    K: Ord + Debug,
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

pub struct Drain<'a, K, V, J> {
    key_iter: std::vec::IntoIter<J>,
    inner: std::vec::Drain<'a, Option<Node<K, V>>>,
}

impl<K, V, J> Iterator for Drain<'_, K, V, J> {
    type Item = (J, V);
    fn next(&mut self) -> Option<Self::Item> {
        let mut current = None;
        while current.is_none() {
            // This is okay to propagate.
            if let Some(node) = self.inner.next()? {
                current = node.into_value();
            }
        }
        Some((self.key_iter.next().unwrap(), current.unwrap()))
    }
}

#[cfg(test)]
mod tests {

    use super::Trie;

    #[test]
    pub fn trie_get_kv_properly() {
        let mut trie: Trie<char, i32> = Trie::from([(['h', 'e', 'l', 'l', 'o'], 4)]);

        assert_eq!(
            trie.get_key_value::<_, String>("hello".chars()),
            Some(("hello".to_string(), &4))
        );
    }

    #[test]
    pub fn test_entry_mut() {

        let mut tree = Trie::<char, usize>::from([
            ("hello".chars(), 4),
            ("bye".chars(), 3)
        ]);
        
        let mut key_iter = tree.entries_mut::<String>();
        
        assert_eq!(*key_iter.next().unwrap().1, 3);
        assert_eq!(*key_iter.next().unwrap().1, 4);
        
        assert!(key_iter.next().is_none());
        
        let mut key_iter = tree.entries_mut::<String>();
        
        *key_iter.next().unwrap().1 = 5;
        
        let mut key_iter = tree.entries_mut::<String>();
        
        assert_eq!(*key_iter.next().unwrap().1, 5);
        assert_eq!(*key_iter.next().unwrap().1, 4);
        assert!(key_iter.next().is_none());
        
    }

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
    pub fn trie_test_defragment() {
        let mut trie = Trie::<char, usize>::new();

        trie.insert(['a'], 1);
        trie.insert(['d'], 2);
        trie.insert(['b'], 3);
        trie.insert(['b', 'c'], 4);

        //
        assert_eq!(trie.get(['a']), Some(&1));
        assert_eq!(trie.get(['d']), Some(&2));
        assert_eq!(trie.get(['b']), Some(&3));
        assert_eq!(trie.get(['b', 'c']), Some(&4));

        trie.remove(['d']);

        assert_eq!(trie.get(['a']), Some(&1));
        assert_eq!(trie.get(['d']), None);
        assert_eq!(trie.get(['b']), Some(&3));
        assert_eq!(trie.get(['b', 'c']), Some(&4));

        trie.shrink_to_fit();

        assert_eq!(trie.get(['a']), Some(&1));
        assert_eq!(trie.get(['d']), None);
        assert_eq!(trie.get(['b']), Some(&3));
        assert_eq!(trie.get(['b', 'c']), Some(&4));
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
    pub fn path_traversal_test() {
        let mut tree = Trie::<char, &str>::new();
        tree.insert(['t', 'e', 's', 't'], "sample_1");
        tree.insert("tea".chars(), "Sample_2");

        // for path in tree.complete_walk() {
        //     println!("Yielded, {:?}", path);
        // }
        // println!("TREE: {:?}", tree);

        // let mut tracker = vec![];

        // println!(
        //     "Traversal: {:?}",
        //     tree.internal_walk_with_fn(['t', 'e', 'a'].into_iter().peekable().by_ref(), |nk| {
        //         println!("Progressing @ {nk:?} {:?}", tree.node[nk]);
        //         println!("Value @ {nk:?} {:?}", tree.node[nk].get(&'t', &tree.node));
        //         tree.key_collect(nk, &mut tracker);
        //     })
        // );

        // panic!("Hello {:?}", 3);
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
        tree.insert("james".chars(), ());

        let mut values = tree
            .completions::<_, String>("he".chars())
            .into_iter()
            .collect::<Vec<_>>();

        values.sort();

        assert_eq!(values.len(), 2);

        assert_eq!(values[0].0, "hello");
        assert_eq!(values[1].0, "hey");

        assert_eq!(
            tree.completions::<_, String>([])
                .map(|(a, _)| a)
                .collect::<Vec<_>>(),
            vec![
                String::from("hello"),
                String::from("hey"),
                String::from("james")
            ]
        );
        assert_eq!(
            tree.completions::<_, String>(['h', 'e', 'l', 'l', 'o'])
                .map(|(a, _)| a)
                .collect::<Vec<_>>(),
            vec![String::from("hello")]
        );
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
