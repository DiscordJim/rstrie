//! A trie library for Rust that allows generic implementing a generic [Trie].
//!
//! A [Trie] actually does not store keys directly but supports storing the individual
//! fragments of what composes a key. For instance, a [String] trie will use [char] and will
//! have the signature `Trie::<char, ()>` ([Trie]).
//!
//! The [Trie] supports the entire API of the standard library's HashMap along with
//! various common algorithms that a [Trie] data structure should support. This [Trie] implementation
//! is optimized for extensibility and supporting many types and may not necessarily be
//! the most memory optimized.
//!
//! # Example
//! ```
//! use rstrie::Trie;
//!
//! let mut trie = Trie::<char, usize>::new();
//! trie.insert("hello".chars(), 4);
//! trie.insert("hey".chars(), 5);
//!
//! assert_eq!(trie.get("hello".chars()), Some(&4));
//! ```

use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::Debug,
    iter::Peekable,
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
    slice::GetDisjointMutError,
    vec::IntoIter,
};

use list::{DisjointMutIndices, Node, NodeIndex, Slots};

use crate::list::SlotsIterMut;
mod list;

/// A [Trie] that is implemented for [String] types. As textual
/// data is the most common usecase for the [Trie], they get some
/// special attention.
///
/// # Example
/// ```
/// use rstrie::StrTrie;
///
/// let trie = StrTrie::<usize>::new();
/// ```
pub type StrTrie<V> = Trie<char, V>;

/// A [Trie] is a data structure that is commonly used to
/// store and retrieve strings. Each node stores a character of the
/// string, which results in a very memory efficient storage of strings.
/// This crate takes a slightly different approach, allowing for the building
/// of Tries from arbitrary types as long as they can satisfy certain properties.
/// 
/// In our case, the key must implement the [Ord] type. This is because the nodes
/// store a list of keys that are sorted, and binary search is used for efficient lookup
/// of the storage index. 
/// 
/// # Example
/// ```
/// use rstrie::Trie;
/// 
/// let mut trie: Trie<char, usize> = Trie::new();
/// trie.insert("hello".chars(), 4);
/// trie.insert("hey".chars(), 5);
/// 
/// assert_eq!(trie.get("hello".chars()), Some(&4));
/// assert_eq!(trie.get("hey".chars()), Some(&5));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct Trie<K, V> {
    /// The node pool, this is where the internal nodes are actually store. This
    /// improves cache locality and ease of access while limiting weird lifetime errors.
    node: Slots<K, V>,
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
    /// The remainder of the path as an iterator.
    remainder: &'a mut I,
}

/// The walk trajectory. This is a list of node indexes that
/// is used to construct various types/various operations within the [Trie].
#[derive(Debug)]
struct WalkTrajectory {
    /// The list of indices.
    path: Vec<NodeIndex>,
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
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

/// The walk context object is a walk across the entire trie. It is
/// driven forward with the [WalkCtx::drive] method.
#[derive(Debug)]
struct WalkCtx {
    /// The pending paths to walk. Each of these is a stak.
    pending: Vec<Vec<NodeIndex>>,
}

/// Iterates automatically over the [WalkCtx] object by continually
/// driving it forward using the [Trie] provided.
struct WalkIter<'a, K, V> {
    trie: &'a Trie<K, V>,
    context: WalkCtx,
}

impl WalkCtx {
    /// Drives this forward, this is similar to the
    pub fn drive<K, V>(&mut self, trie: &Trie<K, V>) -> Option<(NodeIndex, Vec<NodeIndex>)> {
        while let Some(pending) = self.pending.pop() {
            let current = pending.last().unwrap();

            for &child in trie.node[*current].subkeys().rev() {
                // Create a new path.
                let mut new_path = pending.clone();
                new_path.push(child);
                self.pending.push(new_path);
            }

            if trie.node[*current].value().is_some() {
                // We have reached a root.
                return Some((*current, pending));
            }
        }

        None
    }
}

impl<K, V> Iterator for WalkIter<'_, K, V> {
    type Item = (NodeIndex, Vec<NodeIndex>);

    /// Returns the next item in the [WalkIter].
    fn next(&mut self) -> Option<Self::Item> {
        self.context.drive(self.trie)
    }
}

impl<K, V> Trie<K, V> {
    /// Given a path, traverses the [Trie], collecting them into the
    /// return type. Very common method when the user requires the key fragments
    /// to be combined into the entire key.
    fn collect_path_keys<'a, J>(&'a self, nodes: &[NodeIndex]) -> J
    where
        J: FromIterator<&'a K>,
    {
        nodes
            .iter()
            .map(|f| self.node[*f].key())
            .filter_map(<Option<K>>::as_ref)
            .collect()
    }

    /// Performs a standard internal walk of the [Trie] without any special
    /// functor. This will return the last node in the walk. It is a convienence
    /// method to ease the call signatures when we do not explicitly need to
    /// perform any collection on the [Trie].
    fn internal_walk_with_index<'b, I, B>(
        &self,
        remainder: &'b mut Peekable<I>,
    ) -> Result<NodeIndex, WalkFailure<'b, Peekable<I>>>
    where
        I: Iterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        self.internal_walk_with_fn(remainder, |_| {})
    }

    /// Performs an internal walk of the [Trie] and obtains the path
    /// used to get there. Internally, this just calls the internal walk
    /// function and has it collect the nodes as we go.
    fn internal_walk_with_path<'b, I, B>(
        &self,
        remainder: &'b mut Peekable<I>,
    ) -> Result<WalkTrajectory, WalkFailure<'b, Peekable<I>>>
    where
        I: Iterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        self.internal_walk_with_path_fn(remainder, |_| {})
    }

    /// Performs an internal walk of the [Trie] and obtains the path
    /// used to get there. Internally, this just calls the internal walk
    /// function and has it collect the nodes as we go.
    fn internal_walk_with_path_fn<'b, I, F, B>(
        &self,
        remainder: &'b mut Peekable<I>,
        mut functor: F,
    ) -> Result<WalkTrajectory, WalkFailure<'b, Peekable<I>>>
    where
        I: Iterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
        F: FnMut(NodeIndex),
    {
        let mut trajectory_path = vec![];

        self.internal_walk_with_fn(remainder, |node| {
            trajectory_path.push(node);
            functor(node);
        })?;

        Ok(WalkTrajectory {
            path: trajectory_path,
        })
    }

    /// Given a [NodeIndex] and an array, this will fold the key into the
    /// array ONLY if the key links to a non-root node (one where the key is not null).
    fn fold_in_key_optionally<'a>(&'a self, node: NodeIndex, array: &mut Vec<&'a K>) {
        if let Some(inner) = self.node[node].key() {
            array.push(inner);
        }
    }

    /// Performs an internal walk of the [Trie] but collects the key as we go.
    fn internal_walk_collect_key<'a, I, J, B>(
        &self,
        remainder: &'a mut Peekable<I>,
    ) -> Result<(J, NodeIndex), WalkFailure<'a, Peekable<I>>>
    where
        I: Iterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
        for<'b> J: FromIterator<&'b K>,
    {
        let mut collector = vec![];
        let index = self.internal_walk_with_fn(remainder, |nk| {
            self.fold_in_key_optionally(nk, &mut collector)
        })?;
        Ok((collector.into_iter().collect::<J>(), index))
    }

    /// Starts an iterative walk of the [Trie] starting at
    /// the specified root. This allows for efficient and truly
    /// lazy iteration over parts of the [Trie] in a no-copy way.
    fn perform_walk(&self, root: NodeIndex) -> WalkCtx {
        WalkCtx {
            pending: vec![vec![root]],
        }
    }
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
    fn internal_walk_with_fn<'b, I, F, B>(
        &self,
        remainder: &'b mut Peekable<I>,
        visitor_fn: F,
    ) -> Result<NodeIndex, WalkFailure<'b, Peekable<I>>>
    where
        I: Iterator<Item = B>,
        B: Borrow<K>,
        K: Ord,
        F: FnMut(NodeIndex),
    {
        self.internal_walk_with_fn_cmp(remainder, visitor_fn, |a, b| a.cmp(b.borrow()))
    }
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
    fn internal_walk_with_fn_cmp<'a, F, C, A, B>(
        &self,
        remainder: &'a mut Peekable<A>,
        mut visitor_fn: F,
        mut cmp_fn: C,
    ) -> Result<NodeIndex, WalkFailure<'a, Peekable<A>>>
    where
        A: Iterator<Item = B>,
        F: FnMut(NodeIndex),
        for<'b> C: FnMut(&'b K, &'b B) -> Ordering,
    {
        // The root for the path.
        visitor_fn(NodeIndex::ROOT);

        let mut end = &NodeIndex::ROOT;
        loop {
            let Some(current) = remainder.peek() else {
                break;
            };

            // Call on the current node.
            visitor_fn(*end);

            if let Some(slot) = self.node[*end].get_with(&self.node, |k| cmp_fn(k, current)) {
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
        self.node.capacity()
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
            size: 0,
        }
    }

    /// Looks up the node pool index for a certain key,
    /// the key is an iterator over the prefixes. For instance,
    /// for strings this would be characters. This method forms the basis
    /// for [Trie::get] and [Trie::get_mut].
    fn lookup_key<I, B>(&self, key: I) -> Option<NodeIndex>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        self.internal_walk_with_index(&mut key.into_iter().peekable())
            .ok()
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
    pub fn is_prefix<I, B>(&self, key: I) -> bool
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        self.internal_walk_with_index(&mut key.into_iter().peekable())
            .is_ok()
    }

    /// Performs a distance search with a custom function.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    /// use std::cmp::Reverse;
    ///
    /// let mut trie = Trie::<char, ()>::new();
    /// trie.insert("123".chars(), ());
    /// trie.insert("1".chars(), ());
    /// trie.insert("12".chars(), ());
    ///
    /// let longest = trie.search_with_score_fn(|c| c.len());
    /// assert_eq!(longest, Some(vec![ &'1', &'2', &'3' ]));
    ///
    /// let shortest = trie.search_with_score_fn(|c| Reverse(c.len()));
    /// assert_eq!(shortest, Some(vec![ &'1' ]));
    /// ```
    pub fn search_with_score_fn<F, B>(&self, mut distance: F) -> Option<Vec<&K>>
    where
        K: Ord,
        F: FnMut(&[&K]) -> B,
        B: Ord,
    {
        let full_walk = WalkIter {
            context: self.perform_walk(NodeIndex::ROOT),
            trie: self,
        };

        let mut best_score = None;
        let mut best_candidate = None;

        for (_, node_index) in full_walk {
            let collected = self.collect_path_keys::<Vec<_>>(&node_index);
            let score = distance(&collected);
            if best_score.is_none() || *best_score.as_ref().unwrap() < score {
                best_score = Some(score);
                best_candidate = Some(collected);
            }
        }

        best_candidate
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
    /// let mut values = tree.completions("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<(String, &())>>();
    /// assert_eq!(values.len(), 2);
    ///
    ///
    /// assert_eq!(values[0].0, "hello");
    /// assert_eq!(values[1].0, "hey");
    /// ```
    pub fn completions<'a, I, B, J>(&'a self, key: I) -> CompletionIter<'a, K, V, J>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        J: FromIterator<B>,
        B: Borrow<K>,
    {
        let mut collector = vec![];
        match self.internal_walk_with_fn(key.into_iter().peekable().by_ref(), |nk| {
            self.fold_in_key_optionally(nk, &mut collector)
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
    /// let mut values: Vec<(String, &())> = tree.postfix_search("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<_>>();
    /// assert_eq!(values.len(), 2);
    ///
    ///
    /// assert_eq!(values[0].0, "llo");
    /// assert_eq!(values[1].0, "y");
    /// ```
    pub fn postfix_search<'a, I, B, J>(&'a self, key: I) -> PostfixIter<'a, K, V, J>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        J: FromIterator<K>,
        B: Borrow<K>,
    {
        let mut collector = vec![];
        match self.internal_walk_with_fn(key.into_iter().peekable().by_ref(), |nk| {
            self.fold_in_key_optionally(nk, &mut collector)
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
            values: self.node.slot_iter(),
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
            values: self.node.slot_iter_mut(),
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
            inner: self.node.drain_slots(),
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
            .drain_slots()
            .flatten()
            .filter_map(Node::into_value)
            .collect::<Vec<V>>();

        ValueIter {
            inner: values.into_iter(),
        }
    }
    /// Returns the longest prefix match.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert("he".chars(), 1);
    /// trie.insert("hel".chars(), 2);
    /// trie.insert("hello".chars(), 3);
    ///
    /// assert_eq!(trie.longest_prefix_entry("hello".chars()), Some(("hello".to_string(), &3)));
    /// assert_eq!(trie.longest_prefix_entry("hellothere".chars()), Some(("hello".to_string(), &3)));
    /// ```
    pub fn longest_prefix_entry<I, J, B>(&self, key: I) -> Option<(J, &V)>
    where
        I: IntoIterator<Item = B>,
        B: Borrow<K>,
        for<'b> J: FromIterator<&'b K>,
        K: Ord,
    {
        let mut last_productive = NodeIndex::ROOT;
        let mut position = 0;
        let mut last_productive_position = 0;
        let mut collector = vec![];
        self.find_longest_prefix_fn(key, |nk, k| {
            if self.node[nk].value().is_some() {
                last_productive = nk;
                last_productive_position = position;
            }
            collector.push(k);
            position += 1;
        })?;
        let value = self.node[last_productive].value().as_ref()?;
        let key: J = collector[0..last_productive_position + 1]
            .iter()
            .copied()
            .collect();
        Some((key, value))
    }
    /// Returns the longest prefix match.
    ///
    /// # Examples
    /// ```
    /// use rstrie::Trie;
    ///
    /// let mut trie = Trie::<char, usize>::new();
    ///
    /// trie.insert("he".chars(), 1);
    /// trie.insert("hel".chars(), 2);
    /// trie.insert("hello".chars(), 3);
    ///
    /// assert_eq!(trie.longest_prefix("hello".chars()), Some(&3));
    /// assert_eq!(trie.longest_prefix("hellothere".chars()), Some(&3));
    /// ```
    pub fn longest_prefix<I, B>(&self, key: I) -> Option<&V>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        let mut last_productive = NodeIndex::ROOT;
        self.find_longest_prefix_fn(key, |nk, _| {
            if self.node[nk].value().is_some() {
                last_productive = nk;
            }
        })?;
        self.node[last_productive].value().as_ref()
    }

    fn find_longest_prefix_fn<'a, I, F, B>(&'a self, key: I, mut functor: F) -> Option<NodeIndex>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
        F: FnMut(NodeIndex, &'a K),
    {
        match self.internal_walk_with_fn(key.into_iter().peekable().by_ref(), |nk| {
            if let Some(inner) = self.node[nk].key() {
                functor(nk, inner);
            }
        }) {
            Ok(end) => Some(end),
            Err(WalkFailure { root, .. }) => {
                // The match was not an exact match (common case)
                Some(root)
            }
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
                context: self.perform_walk(NodeIndex::ROOT),
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
            inner: self.perform_walk(NodeIndex::ROOT),
            trie: self,
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
                context: self.perform_walk(NodeIndex::ROOT),
                trie: self,
            },
            _type: PhantomData,
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
            inner: self.perform_walk(NodeIndex::ROOT),
            trie: self,
            // slot: None,.
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
    /// tree.insert("hellooo".chars(), "world2");
    /// assert_eq!(*tree.get("hello".chars()).unwrap(), "world");
    /// ```
    pub fn get<I, B>(&self, key: I) -> Option<&V>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        self.node[self.lookup_key(key)?].value().as_ref()
    }
    /// Tries to get the key map.
    ///
    /// # Errors
    /// If the keys are not disjoint, i.e, if they map to the same value.
    fn try_get_key_map<I, B, const N: usize>(
        &mut self,
        keys: [I; N],
    ) -> Result<[NodeIndex; N], NodeIndex>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        let mut array = [NodeIndex::ROOT; N];

        for (i, k) in keys.into_iter().enumerate() {
            let candidate = self.lookup_key(k);

            if let Some(candidate) = candidate {
                if array.contains(&candidate) {
                    return Err(candidate);
                }
            }
            array[i] = candidate.unwrap();
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
    /// *keys[0].as_mut().unwrap() = 4;
    /// *keys[1].as_mut().unwrap() = 2;
    ///
    /// assert_eq!(tree.get("hello".chars()), Some(&4));
    /// assert_eq!(tree.get("world".chars()), Some(&2));
    /// ```
    pub fn get_disjoint_mut<I, B, const N: usize>(
        &mut self,
        keys: [I; N],
    ) -> DisjointMutIndices<'_, K, V, N>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        self.try_get_disjoint_mut(keys)
            .expect("Keys were overlapping.")
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
    /// *keys[0].as_mut().unwrap() = 4;
    /// *keys[1].as_mut().unwrap() = 2;
    ///
    /// assert_eq!(tree.get("hello".chars()), Some(&4));
    /// assert_eq!(tree.get("world".chars()), Some(&2));
    /// ```
    pub fn try_get_disjoint_mut<I, B, const N: usize>(
        &mut self,
        keys: [I; N],
    ) -> Result<DisjointMutIndices<K, V, N>, GetDisjointMutError>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        // Calculate the key map.
        let array = self
            .try_get_key_map(keys)
            .map_err(|_| GetDisjointMutError::OverlappingIndices)?;
        self.node.get_disjoint_mut(array)
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
    pub fn get_mut<I, B>(&mut self, key: I) -> Option<&mut V>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
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
    pub fn get_key_value<I, B, J>(&self, key: I) -> Option<(J, &V)>
    where
        I: IntoIterator<Item = B>,
        for<'a> J: FromIterator<&'a K>,
        K: Ord,
        B: Borrow<K>,
    {
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
    /// *trie.get_key_value_mut::<_, _, String>(['a']).as_mut().unwrap().1 = 2;
    /// assert_eq!(trie.get_key_value_mut::<_, _, String>(['a']), Some(("a".to_string(), &mut 2)));
    ///
    ///
    /// ```
    pub fn get_key_value_mut<I, B, J>(&mut self, key: I) -> Option<(J, &mut V)>
    where
        I: IntoIterator<Item = B>,
        for<'a> J: FromIterator<&'a K>,
        K: Ord,
        B: Borrow<K>,
    {
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
    /// assert_eq!(trie.remove_entry::<_, _, String>(['a']), Some(("a".to_string(), 2)));
    ///
    /// ```
    ///
    pub fn remove_entry<I, B, J>(&mut self, key: I) -> Option<(J, V)>
    where
        I: IntoIterator<Item = B>,
        for<'a> J: FromIterator<&'a K>,
        K: Ord,
        B: Borrow<K>,
    {
        let mut path = vec![];
        let trajectory = self
            .internal_walk_with_path_fn(key.into_iter().peekable().by_ref(), |nk| {
                self.fold_in_key_optionally(nk, &mut path)
            })
            .ok()?;

        let mut traj_path = trajectory.path.iter().rev().peekable();

        if traj_path.peek().is_some() {
            let reconstruction = path.into_iter().collect::<J>();
            let value = self.remove_post_walk(&trajectory.path)?;
            Some((reconstruction, value))
        } else {
            self.remove_post_walk(&trajectory.path)?;
            None
        }
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
    /// tree.insert("hello".chars(), 12);
    ///
    /// assert_eq!(tree.remove("hello".chars()).unwrap(), 12);
    /// ```
    pub fn remove<I, B>(&mut self, master: I) -> Option<V>
    where
        I: IntoIterator<Item = B>,
        K: Ord,
        B: Borrow<K>,
    {
        // Calculate the removal trajectory.
        let trajectory = self
            .internal_walk_with_path(master.into_iter().peekable().by_ref())
            .ok()?;

        // Remove and then correct the removal.
        self.remove_post_walk(&trajectory.path)
            .inspect(|_| self.size -= 1)
    }

    /// Detaches a node and its subkeys recursively. Since subkeys are
    /// always dependent on their parent, this is equivalent to just deleting
    /// a subtree.
    fn detach_node(&mut self, source: NodeIndex) {
        let keys = self.node[source].subkeys().copied().collect::<Vec<_>>();
        for i in keys {
            self.detach_node(i);
        }
        self.node.remove(source);
    }
    /// Performs the actual removal from the [Trie]. This will
    /// traverse back from the node using the calculated trajectory
    /// in order to remove the node from the [Trie].
    ///
    /// This method exists to allow us to easily implement remove entry
    /// without having to walk the tree a second time to reconstruct the [Trie].
    fn remove_post_walk(&mut self, path: &[NodeIndex]) -> Option<V>
    where
        K: Ord,
    {
        let internal_value = self.node[*path.last()?].value_mut().take();

        let mut sub_index: isize = (path.len() - 1) as isize;

        // Starting at the very end, this travels up the tree, removing
        // all nodes attached to the tail that are no longer valid.
        while sub_index >= 0 {
            // Selected index.
            let sh_index = path[sub_index as usize];

            // If we do not have children...
            // WHY? Because if we have children, clearly this forms
            // part of a superkey, and thus removing it would then remove
            // that as well. Consider:
            //
            // "Yes"
            // "Yesman"
            //
            // Both of these share a subtree, but deleting "Yes" should
            // not also delete "Yesman".
            if self.node[sh_index].sub_key_len() == 0 {
                // Detach the node....
                self.detach_node(sh_index);

                // Then if there is something that is above it, remove it.
                if sub_index > 0 {
                    let above = path[(sub_index - 1) as usize];
                    self.node[above].remove_subkey(sh_index);
                }
            } else {
                // If we have children, then clearly this is a valid path, so we will
                // not detach it and thus the path that preceeds this node is still valid,
                // so there is no purpose in continuation.
                break;
            }
            sub_index -= 1;
        }

        internal_value
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
            .any(|(_, v)| v.value().is_some() && v.value().as_ref().unwrap() == value)
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
        K: Ord,
    {
        match self.internal_walk_with_index(master.into_iter().peekable().by_ref()) {
            Ok(v) => {
                // This node already exists, so replace it and then return
                // the previous value.
                let current = self.node[v].value_mut().take();
                if current.is_none() {
                    self.size += 1;
                }
                *self.node[v].value_mut() = Some(value);

                current
            }
            Err(WalkFailure {
                mut root,
                remainder,
            }) => {
                let mut nk = Some(root);
                for item in remainder {
                    nk = Some(self.node.insert(Node::keyed(item)));
                    self.node.insert_subkey(root, nk.unwrap());
                    root = nk.unwrap();
                }
                *self.node[nk.unwrap()].value_mut() = Some(value);
                self.size += 1;
                None
            }
        }
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

impl<'a, K, V, J> Iterator for CompletionIter<'a, K, V, J>
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
    /// let mut values = tree.completions("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<(String, &())>>();
    ///
    /// values.sort();
    ///
    ///
    /// assert_eq!(values[0].0, "hello");
    /// assert_eq!(values[1].0, "hey");
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (current, path) = self.inner.drive(self.trie)?;

        let key = assemble_completion(self.trie, &self.beginning, path);

        Some((key, self.trie.node[current].value().as_ref().unwrap()))
    }
}

impl<'a, K, V, J> Iterator for PostfixIter<'a, K, V, J>
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
    /// let mut values = tree.postfix_search("he".chars())
    ///     .into_iter()
    ///     .collect::<Vec<(String, &())>>();
    /// assert_eq!(values.len(), 2);
    ///
    ///
    /// assert_eq!(values[0].0, "llo");
    /// assert_eq!(values[1].0, "y");
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        let (current, path) = self.inner.drive(self.trie)?;

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
        .filter_map(<Option<K>>::as_ref);

    beginning
        .iter()
        // The following lines just omit the very last element.
        .rev()
        .skip(1)
        .rev()
        .copied()
        .chain(tail_end)
        .collect::<J>()
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

/// An iterator created by consuming the [Trie].
/// Contains all the entries of the [Trie].
pub struct EntryIterMut<'a, K, V, J> {
    /// The internal reference to the [Trie].
    trie: &'a mut Trie<K, V>,
    /// The walk context for iterating over the fields.
    inner: WalkCtx,
    /// The type that we will collect into. Although technically
    /// we could collect into a different type on each iteration, we
    /// want to avoid this type of strange behaviour and streamline
    /// the process.
    _type: PhantomData<J>,
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct IntoEntryIter<K, V, J> {
    trie: Trie<K, V>,
    inner: WalkCtx,
    _type: PhantomData<J>,
}

/// An iterator created by consuming the [Trie].
///
/// Contains all the entries of the [Trie].
pub struct EntryIterRef<'a, K, V, J> {
    inner: WalkIter<'a, K, V>,
    _type: PhantomData<J>,
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
    inner: IntoIter<V>,
}

/// An iterator over the values of a [Trie].
pub struct ValueIterRef<'a, K, V> {
    values: std::slice::Iter<'a, Option<Node<K, V>>>,
}

/// An iterator over the values of a [Trie]
/// that provides mutable references.
pub struct ValueIterMut<'a, K, V> {
    values: SlotsIterMut<'a, K, V>,
}

impl<V> Trie<char, V> {
    /// A special implementation of [Trie::insert] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    /// ```
    pub fn insert_str(&mut self, key: &str, value: V) -> Option<V> {
        self.insert(key.chars(), value)
    }
    /// A special implementation of [Trie::remove] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// assert!(trie.remove_str("hello").is_none());
    /// ```
    pub fn remove_str(&mut self, key: &str) -> Option<V> {
        self.remove(key.chars())
    }
    /// A special implementation of [Trie::contains_key] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    ///
    /// assert!(trie.contains_key_str("hello"));
    /// assert!(!trie.contains_key_str("hey"));
    ///
    ///
    /// ```
    pub fn contains_key_str(&self, key: &str) -> bool {
        self.contains_key(key.chars())
    }
    /// A special implementation of [Trie::is_prefix] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    ///
    /// assert!(trie.is_prefix_str("hello"));
    /// assert!(trie.is_prefix_str("he"));
    /// assert!(!trie.is_prefix_str("hellop"));
    ///
    ///
    /// ```
    pub fn is_prefix_str(&self, key: &str) -> bool {
        self.is_prefix(key.chars())
    }
    /// A special implementation of [Trie::completions] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    /// trie.insert_str("hey", 1);
    /// trie.insert_str("james", 2);
    ///
    /// let mut values = trie.completions_str("he");
    /// assert_eq!(values.next().unwrap().0.as_str(), "hello");
    /// assert_eq!(values.next().unwrap().0.as_str(), "hey");
    /// ```
    pub fn completions_str(&self, key: &str) -> CompletionIter<'_, char, V, String> {
        self.completions(key.chars())
    }
    /// A special implementation of [Trie::postfix_search] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    /// trie.insert_str("hey", 1);
    /// trie.insert_str("james", 2);
    ///
    /// let mut values = trie.postfix_search_str("he");
    /// assert_eq!(values.next().unwrap().0.as_str(), "llo");
    /// assert_eq!(values.next().unwrap().0.as_str(), "y");
    /// ```
    pub fn postfix_search_str(&self, key: &str) -> PostfixIter<'_, char, V, String> {
        self.postfix_search(key.chars())
    }
    /// A special implementation of [Trie::longest_prefix_entry] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    /// trie.insert_str("hey", 1);
    /// trie.insert_str("james", 2);
    ///
    /// assert_eq!(trie.longest_prefix_entry_str("hello"), Some(("hello".to_string(), &3)));
    /// assert_eq!(trie.longest_prefix_entry_str("hellothere"), Some(("hello".to_string(), &3)));
    /// ```
    pub fn longest_prefix_entry_str(&self, key: &str) -> Option<(String, &V)> {
        self.longest_prefix_entry(key.chars())
    }
    /// A special implementation of [Trie::longest_prefix] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    /// trie.insert_str("hello", 3);
    /// trie.insert_str("hey", 1);
    /// trie.insert_str("james", 2);
    ///
    /// assert_eq!(trie.longest_prefix_str("hello"), Some(&3));
    /// assert_eq!(trie.longest_prefix_str("hellothere"), Some(&3));
    /// ```
    pub fn longest_prefix_str(&self, key: &str) -> Option<&V> {
        self.longest_prefix(key.chars())
    }
    /// A special implementation of [Trie::keys] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = trie.keys_str();
    /// assert_eq!(key_iter.next().unwrap(), "bye");
    /// assert_eq!(key_iter.next().unwrap(), "hello");
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn keys_str(&self) -> KeyIter<'_, char, V, String> {
        self.keys()
    }
    /// A special implementation of [Trie::into_entries] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = trie.into_entries_str();
    ///
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), 3));
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), 4));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn into_entries_str(self) -> IntoEntryIter<char, V, String> {
        self.into_entries()
    }
    /// A special implementation of [Trie::entries] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = trie.entries_str();
    ///
    /// assert_eq!(key_iter.next().unwrap(), ("bye".to_string(), &3));
    /// assert_eq!(key_iter.next().unwrap(), ("hello".to_string(), &4));
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn entries_str(&self) -> EntryIterRef<'_, char, V, String> {
        self.entries()
    }
    /// A special implementation of [Trie::entries_mut] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::from([
    ///     ("hello".chars(), 4),
    ///     ("bye".chars(), 3)
    /// ]);
    ///
    /// let mut key_iter = trie.entries_mut_str();
    ///
    /// assert_eq!(*key_iter.next().unwrap().1, 3);
    /// assert_eq!(*key_iter.next().unwrap().1, 4);
    /// assert!(key_iter.next().is_none());
    /// ```
    pub fn entries_mut_str(&mut self) -> EntryIterMut<'_, char, V, String> {
        self.entries_mut()
    }
    /// A special implementation of [Trie::get] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut tree = StrTrie::<&str>::new();
    /// tree.insert_str("hello", "world");
    /// tree.insert_str("hellooo", "world2");
    /// assert_eq!(*tree.get_str("hello").unwrap(), "world");
    /// ```
    pub fn get_str(&self, key: &str) -> Option<&V> {
        self.get(key.chars())
    }
    /// A special implementation of [Trie::get_mut] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut tree = StrTrie::<&str>::new();
    /// tree.insert_str("hello", "world");
    /// assert_eq!(*tree.get_mut_str("hello").unwrap(), "world");
    ///
    /// *tree.get_mut_str("hello").unwrap() = "world2";
    /// assert_eq!(*tree.get_mut_str("hello").unwrap(), "world2");
    /// ```
    pub fn get_mut_str(&mut self, key: &str) -> Option<&mut V> {
        self.get_mut(key.chars())
    }
    /// A special implementation of [Trie::get_disjoint_mut] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut tree = StrTrie::<usize>::new();
    ///
    /// tree.insert_str("hello", 3);
    /// tree.insert_str("world", 4);
    ///
    /// let mut keys = tree.get_disjoint_mut_str([
    ///     "hello",
    ///     "world"
    /// ]);
    /// *keys[0].as_mut().unwrap() = 4;
    /// *keys[1].as_mut().unwrap() = 2;
    ///
    /// assert_eq!(tree.get_str("hello"), Some(&4));
    /// assert_eq!(tree.get_str("world"), Some(&2));
    /// ```
    pub fn get_disjoint_mut_str<const N: usize>(
        &mut self,
        key: [&str; N],
    ) -> DisjointMutIndices<'_, char, V, N> {
        let modded = core::array::from_fn(|i| key[i].chars());
        self.get_disjoint_mut(modded)
    }
    /// A special implementation of [Trie::try_get_disjoint_mut] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut tree = StrTrie::<usize>::new();
    ///
    /// tree.insert_str("hello", 3);
    /// tree.insert_str("world", 4);
    ///
    /// let mut keys = tree.try_get_disjoint_mut_str([
    ///     "hello",
    ///     "world"
    /// ]).unwrap();
    /// *keys[0].as_mut().unwrap() = 4;
    /// *keys[1].as_mut().unwrap() = 2;
    ///
    /// assert_eq!(tree.get_str("hello"), Some(&4));
    /// assert_eq!(tree.get_str("world"), Some(&2));
    /// ```
    pub fn try_get_disjoint_mut_str<const N: usize>(
        &mut self,
        key: [&str; N],
    ) -> Result<DisjointMutIndices<'_, char, V, N>, GetDisjointMutError> {
        let modded = core::array::from_fn(|i| key[i].chars());
        self.try_get_disjoint_mut(modded)
    }
    /// A special implementation of [Trie::get_key_value] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    ///
    /// trie.insert_str("abc", 1);
    ///
    /// assert_eq!(trie.get_key_value_str("abc"), Some(("abc".to_string(), &1)))
    ///
    /// ```
    pub fn get_key_value_str(&self, key: &str) -> Option<(String, &V)> {
        self.get_key_value(key.chars())
    }
    /// A special implementation of [Trie::get_key_value_mut] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    ///
    /// trie.insert_str("a", 1);
    ///
    /// assert_eq!(trie.get_key_value_mut_str("a"), Some(("a".to_string(), &mut 1)));
    ///
    /// *trie.get_key_value_mut_str("a").as_mut().unwrap().1 = 2;
    /// assert_eq!(trie.get_key_value_mut_str("a"), Some(("a".to_string(), &mut 2)));
    ///
    ///
    /// ```
    pub fn get_key_value_mut_str(&mut self, key: &str) -> Option<(String, &mut V)> {
        self.get_key_value_mut(key.chars())
    }
    /// A special implementation of [Trie::remove_entry] designed specifically for
    /// strings.
    ///
    /// NOTE: This is just a helper method. There are no string-based optimizations
    /// going on here.
    ///
    /// # Example
    /// ```
    /// use rstrie::StrTrie;
    ///
    /// let mut trie = StrTrie::<usize>::new();
    ///
    /// trie.insert_str("a", 2);
    ///
    /// assert_eq!(trie.remove_entry_str("a"), Some(("a".to_string(), 2)));
    ///
    /// ```
    pub fn remove_entry_str(&mut self, key: &str) -> Option<(String, V)> {
        self.remove_entry(key.chars())
    }
}

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
        let (current, path) = { self.inner.drive(&self.trie)? };

        let calced = self.trie.collect_path_keys::<J>(&path);

        Some((calced, self.trie.node[current].value_mut().take().unwrap()))
    }
}

impl<'a, K, V, J> Iterator for EntryIterMut<'a, K, V, J>
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
        let (current, path) = { self.inner.drive(self.trie)? };

        let calced = self.trie.collect_path_keys::<J>(&path);

        // SAFETY: The borrow checker does not know that subsequent calls return distinct
        // nodes, and thus the aliasing rules are upheld and we only have a single mutable reference at a time.
        let candidate = unsafe { &mut *(&mut self.trie.node[current] as *mut Node<K, V>) };

        Some((calced, ValueSlot { node: candidate }))
    }
}

pub struct ValueSlot<'a, K, V> {
    node: &'a mut Node<K, V>,
}

impl<K, V> Deref for ValueSlot<'_, K, V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        self.node.value().as_ref().unwrap()
    }
}

impl<K, V> DerefMut for ValueSlot<'_, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.node.value_mut_unchecked()
    }
}

impl<K, V> PartialEq for Trie<K, V>
where
    K: PartialEq,
    V: PartialEq,
{
    /// Checks if two [Trie] are equal.
    fn eq(&self, other: &Self) -> bool {
        self.node.eq(&other.node)
    }
}

impl<'a, K, V, J> Iterator for EntryIterRef<'a, K, V, J>
where
    J: FromIterator<&'a K>,
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

        Some((
            self.inner.trie.collect_path_keys(&path),
            self.inner.trie.node[curr].value().as_ref().unwrap(),
        ))
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
            current = node.value_mut().as_mut();
        }

        current
    }
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

impl<'a, KP, K, V> Extend<(KP, &'a V)> for Trie<K, V>
where
    K: Ord,
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

#[cfg(feature = "arbitrary")]
mod arbitrary_trie {
    use core::f32;
    use arbitrary::{Arbitrary, Unstructured};

    use crate::Trie;

    fn random_norm_float(u: &mut Unstructured<'_>) -> arbitrary::Result<f32> {
        Ok(f32::from_bits(u32::arbitrary(u)?).abs() / f32::MAX)
    }

    impl<'a, K, V> Arbitrary<'a> for Trie<K, V>
    where
        K: Arbitrary<'a> + Ord + Clone,
        V: Arbitrary<'a>,
    {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            let mut trie = Trie::<K, V>::new();
            let trie_length = u16::arbitrary(u)? as usize;
            let mut buffer = Vec::new();
            for _ in 0..trie_length {
                let vector: Vec<K> = Vec::arbitrary(u)?;
                let value = V::arbitrary(u)?;

                buffer.push(Some(vector.clone()));

                trie.insert(vector.into_iter(), value);
            }

            let thres = random_norm_float(u)?;
            for i in 0..buffer.len() {
                if random_norm_float(u)? < thres {
                    let key = buffer.get_mut(i).unwrap().take().unwrap();
                    trie.remove(key.into_iter());
                }
            }

            Ok(trie)
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{cmp::Reverse, time::Duration};

    use super::Trie;

    #[test]
    #[cfg(feature = "arbitrary")]
    pub fn test_arbitrary_crate() {
        use arbitrary::{Arbitrary, Unstructured};
        use rand::Rng;

        let val: [u8; 12] = rand::rng().random();

        let trie: Trie<char, usize> = Trie::arbitrary(&mut Unstructured::new(&val)).unwrap();
        let trie2: Trie<char, usize> = Trie::arbitrary(&mut Unstructured::new(&val)).unwrap();
        assert_eq!(trie, trie2);
    }

    #[test]
    pub fn test_root_len_mod() {
        let mut trie = Trie::<char, usize>::new();

        assert_eq!(trie.len(), 0);
        trie.insert_str("", 3);
        assert_eq!(trie.len(), 1);

        trie.insert_str("", 4);
        assert_eq!(trie.len(), 1);

        trie.remove_str("");
        assert_eq!(trie.len(), 0);
    }

    #[test]
    pub fn test_size_duplicates() {
        let mut trie = Trie::<char, usize>::from([
            ("hello".chars(), 4),
            ("hey".chars(), 3),
            ("hamburger".chars(), 10),
        ]);

        assert_eq!(trie.len(), 3);
        trie.insert("hello".chars(), 4);
        assert_eq!(trie.len(), 3);

        trie.insert("heman".chars(), 10);
        assert_eq!(trie.len(), 4);

        trie.remove("heman".chars());
        assert_eq!(trie.len(), 3);

        trie.remove("heman".chars());
        assert_eq!(trie.len(), 3);

        trie.remove("albert".chars());
        assert_eq!(trie.len(), 3);
    }

    #[test]
    #[cfg(all(feature = "arbitrary", feature = "serde"))]
    pub fn arbtest_arb_impl() {
        use arbitrary::Arbitrary;

        arbtest::arbtest(|u| {
            let standard = Trie::<char, usize>::arbitrary(u)?;
            let reversed: Trie<char, usize> =
                serde_json::from_slice(&serde_json::to_vec(&standard).unwrap()).unwrap();
            assert_eq!(standard, reversed);
            Ok(())
        })
        .budget(Duration::from_secs(5));
    }

    #[test]
    pub fn test_delete_root_vector() {
        let mut root = Trie::<char, usize>::from([("hello".chars(), 4)]);
        root.insert([], 5);
        root.remove::<[char; 0], char>([]);
        root.remove::<[char; 0], char>([]);
    }

    #[test]
    #[cfg(all(feature = "rkyv", feature = "serde"))]
    pub fn arbtest_arb_impl_rkyv() {
        use arbitrary::Arbitrary;
        use rkyv::rancor::Error;

        arbtest::arbtest(|u| {
            let standard = Trie::<char, u64>::arbitrary(u)?;

            // println!("STADNAR: {:?}", standard);

            let archived = rkyv::to_bytes::<Error>(&standard).unwrap();

            let archived: Trie<char, u64> = rkyv::from_bytes::<_, Error>(&archived).unwrap();

            // let reversed: Trie<char, usize> = serde_json::from_slice(&serde_json::to_vec(&standard).unwrap()).unwrap();
            assert_eq!(standard, archived);
            Ok(())
        })
        .budget(Duration::from_secs(5));
    }

    #[test]
    #[cfg(feature = "serde")]
    pub fn test_serde_serialize_map() {
        let tree = Trie::<char, usize>::from([("hello".chars(), 4), ("hey".chars(), 5)]);

        let val = serde_json::to_vec(&tree).unwrap();
        let tree2: Trie<char, usize> = serde_json::from_slice(&val).unwrap();
        assert_eq!(tree, tree2);
    }

    #[test]
    pub fn trie_get_kv_properly() {
        let trie: Trie<char, i32> = Trie::from([(['h', 'e', 'l', 'l', 'o'], 4)]);

        assert_eq!(
            trie.get_key_value::<_, _, String>("hello".chars()),
            Some(("hello".to_string(), &4))
        );
    }

    #[test]
    pub fn test_entry_mut() {
        let mut tree = Trie::<char, usize>::from([("hello".chars(), 4), ("bye".chars(), 3)]);

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

        *keys[0].as_mut().unwrap() = 4;
        *keys[1].as_mut().unwrap() = 2;

        assert_eq!(tree.get("hello".chars()), Some(&4));
        assert_eq!(tree.get("world".chars()), Some(&2));
    }

    #[test]
    pub fn test_trie_equality_simple() {
        let trie_1 = Trie::from([("hello".chars(), 3), ("good".chars(), 2)]);
        let trie_2 = Trie::from([("hello".chars(), 3), ("good".chars(), 2)]);
        let trie_3 = Trie::from([("hello".chars(), 3), ("good".chars(), 3)]);
        assert_eq!(trie_1, trie_2);
        assert_ne!(trie_1, trie_3);
    }

    #[test]
    pub fn test_into_values() {
        let trie: Trie<char, usize> = Trie::from([("".chars(), 1), ("tra".chars(), 2)]);

        let mut values = trie.into_values();
        assert_eq!(values.next(), Some(1));
        assert_eq!(values.next(), Some(2));
    }

    #[test]
    pub fn test_removal_subword() {
        let mut trie_1 = Trie::from([("hello".chars(), 3), ("good".chars(), 2)]);

        trie_1.insert("go".chars(), 8);
        assert_eq!(trie_1.remove("go".chars()), Some(8));
        assert!(trie_1.contains_key("good".chars()));

        // panic!("ye");
    }

    #[test]
    pub fn test_trie_equality_complex() {
        let mut trie_1 = Trie::from([("hello".chars(), 3), ("good".chars(), 2)]);
        let mut trie_2 = Trie::from([("hello".chars(), 3), ("good".chars(), 2)]);
        trie_2.remove("hello".chars());
        trie_2.insert("hey".chars(), 12);
        trie_2.insert("hello".chars(), 3);
        // trie_2.remove("hey".chars());

        trie_1.remove("good".chars());
        trie_1.insert("go".chars(), 8);
        trie_1.insert("hey".chars(), 12);

        assert!(trie_1.contains_key("hey".chars()));
        trie_1.insert("good".chars(), 2);
        trie_1.remove("go".chars());
        assert!(trie_1.contains_key("good".chars()));

        assert_eq!(trie_1, trie_2);
        // assert_ne!(trie_1, trie_3);
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
    }

    #[test]
    pub fn basic_trie_insert() {
        let mut tree: Trie<char, &str> = Trie::new();
        tree.insert("test".chars(), "sample_1");
        tree.insert("testttt".chars(), "sample_2");
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
            .completions::<_, _, String>("he".chars())
            .into_iter()
            .collect::<Vec<_>>();

        values.sort();

        assert_eq!(values.len(), 2);

        assert_eq!(values[0].0, "hello");
        assert_eq!(values[1].0, "hey");

        assert_eq!(
            tree.completions::<_, char, String>([])
                .map(|(a, _)| a)
                .collect::<Vec<_>>(),
            vec![
                String::from("hello"),
                String::from("hey"),
                String::from("james")
            ]
        );
        assert_eq!(
            tree.completions::<_, _, String>(['h', 'e', 'l', 'l', 'o'])
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
        })
        .budget(Duration::from_secs(3));
    }

    #[test]
    pub fn test_longest_prefix() {
        let mut trie = Trie::<char, usize>::new();

        trie.insert("he".chars(), 1);
        trie.insert("hel".chars(), 2);
        trie.insert("hello".chars(), 3);

        assert_eq!(
            trie.longest_prefix_entry("hello".chars()),
            Some(("hello".to_string(), &3))
        );
        assert_eq!(
            trie.longest_prefix_entry("hellothere".chars()),
            Some(("hello".to_string(), &3))
        );
    }

    #[test]
    pub fn test_entry_iter_mut_soundness() {
        let mut trie = Trie::from([("hello".chars(), 42), ("hey".chars(), 55)]);

        let wow = trie.entries_mut::<String>();
        for (_, mut val) in wow {
            *val = 23;
        }

        for (_, val) in trie.entries::<String>() {
            assert_eq!(*val, 23);
        }
    }

    #[test]
    pub fn test_longest_prefix_routing_table() {
        let mut trie = Trie::<bool, MatchRule>::new();

        fn convert_to_bits(val: u8) -> [bool; 8] {
            let mut result = [false; 8];
            for i in 0..8 {
                result[7 - i] = (val & (1 << i)) != 0;
                // println!("Value: {:08b}", );
            }
            result
        }

        #[derive(PartialEq, Eq, Debug)]
        enum MatchRule {
            Forward(u8),
            Delete,
        }

        // All addresses that start with 101 get forwarded to 3.
        trie.insert([true, false, true], MatchRule::Forward(3));

        // All addresses that start with 1010 get forwarded to 4;
        trie.insert([true, false, true, false], MatchRule::Forward(4));

        // All addresses that start with 1010001 get deleted
        trie.insert(
            [true, false, true, false, false, false, true],
            MatchRule::Delete,
        );

        assert_eq!(
            trie.longest_prefix(convert_to_bits(0b10100000u8)),
            Some(&MatchRule::Forward(4))
        );
        assert_eq!(
            trie.longest_prefix(convert_to_bits(0b10110000u8)),
            Some(&MatchRule::Forward(3))
        );
        assert_eq!(
            trie.longest_prefix(convert_to_bits(0b10100010u8)),
            Some(&MatchRule::Delete)
        );
    }

    #[test]
    pub fn test_scoring_fn() {
        let mut trie = Trie::<char, ()>::new();
        trie.insert("hello".chars(), ());
        trie.insert("he".chars(), ());
        trie.insert("hema".chars(), ());

        assert_eq!(
            trie.search_with_score_fn(|f| f.len()),
            Some(vec![&'h', &'e', &'l', &'l', &'o'])
        );
        assert_eq!(
            trie.search_with_score_fn(|f| Reverse(f.len())),
            Some(vec![&'h', &'e'])
        );
    }

    #[test]
    pub fn test_scoring_fn_levinstein() {
        let mut trie = Trie::<char, ()>::new();
        trie.insert("hello".chars(), ());
        trie.insert("he".chars(), ());
        trie.insert("hema".chars(), ());
        trie.insert("racecar".chars(), ());
        trie.insert("nissan".chars(), ());

        let lev = |target| {
            let lev_result = trie.search_with_score_fn(|f| {
                let string = f.iter().copied().collect::<String>();
                let distance = edit_distance::edit_distance(target, &string);
                Reverse(distance)
            });

            lev_result.map(|f| f.into_iter().copied().collect::<String>())
        };

        assert_eq!(lev("hello"), Some("hello".to_string()));
        assert_eq!(lev("niwan"), Some("nissan".to_string()));

        // assert_eq!(trie.search_with_score_fn(|f| f.len()), Some(vec![ &'h', &'e', &'l', &'l', &'o' ]));
        // assert_eq!(trie.search_with_score_fn(|f| Reverse(f.len())), Some(vec![ &'h', &'e' ]));
    }
}
