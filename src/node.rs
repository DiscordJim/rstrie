use std::{collections::HashMap, hash::BuildHasher};
use core::hash::Hash;

use crate::NodeKey;

#[derive(Debug, Default, Clone)]
pub(crate) struct Node<K, V, S> {
    key: Option<K>,
    sub_keys: HashMap<K, NodeKey, S>,
    value: Option<V>,
    parent: Option<NodeKey>,
}

impl<K, V, S> Node<K, V, S>
where
    S: BuildHasher + Default,
{
    pub fn root() -> Self {
        Node {
            key: None,
            sub_keys: HashMap::with_hasher(S::default()),
            value: None,
            parent: None,
        }
    }
    pub fn keyed(key: K, parent: NodeKey) -> Self {
        Self {
            sub_keys: HashMap::with_hasher(S::default()),
            value: None,
            key: Some(key),
            parent: Some(parent),
        }
    }
    
}

impl<K, V, S> Node<K, V, S> {
    pub fn key(&self) -> &Option<K> {
        &self.key
    }
    pub fn is_root(&self) -> bool {
        self.key.is_none()
    }
    pub fn parent(&self) -> Option<NodeKey> {
        self.parent.clone()
    }
    pub fn get(&self, key: &K) -> Option<&NodeKey>
    where 
        K: Eq + Hash,
        S: BuildHasher
    {
        self.sub_keys.get(key)
    }
    pub fn insert(&mut self, key: K, value: NodeKey) -> Option<NodeKey>
    where 
        K: Eq + Hash,
        S: BuildHasher
    {
        self.sub_keys.insert(key, value)
    }
    pub fn remove(&mut self, key: &K) -> Option<NodeKey>
    where 
        K: Eq + Hash,
        S: BuildHasher
    {
        self.sub_keys.remove(key)
    } 
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
}