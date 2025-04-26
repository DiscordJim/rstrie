use std::{collections::HashMap, hash::BuildHasher, marker::PhantomData};
use core::hash::Hash;

use crate::NodeKey;

#[derive(Debug, Default, Clone)]
pub(crate) struct Node<K, V> {
    key: Option<K>,
    pub(crate) sub_keys: Vec<(K, NodeKey)>,
    value: Option<V>,
    parent: Option<NodeKey>
}

impl<K, V> Node<K, V>
{
    pub fn root() -> Self {
        Node {
            key: None,
            sub_keys: Vec::default(),
            value: None,
            parent: None
        }
    }
    pub fn keyed(key: K, parent: NodeKey) -> Self {
        Self {
            sub_keys: Vec::default(),
            value: None,
            key: Some(key),
            parent: Some(parent)
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
    pub fn parent(&self) -> Option<NodeKey> {
        self.parent.clone()
    }
    pub fn get(&self, key: &K) -> Option<&NodeKey>
    where 
        K: Ord
    {
        let val = self.sub_keys.iter().find(|(k, _)| key == k);
        Some(&val?.1)
    }
  
    pub fn insert(&mut self, key: K, value: NodeKey) -> Option<NodeKey>
    where 
        K: Ord
    {

        match self.bin_search(&key) {
            Ok(found) => {
             
                let (_ , old) = std::mem::replace(&mut self.sub_keys[found], (key, value));
                Some(old)
            }

            Err(found) => {
                self.sub_keys.insert(found, (key, value));

                None
            }
        }

        // let old = self.remove(&key);

        // self.sub_keys.push((key, value));

        // old
    }
    fn bin_search(&self, key: &K) -> Result<usize, usize>
    where 
        K: Ord
    {
        self.sub_keys.binary_search_by(|(k, _)| k.cmp(key))
    }
    pub fn remove(&mut self, key: &K) -> Option<NodeKey>
    where 
        K: Ord
    {
        Some(self.sub_keys.remove(self.bin_search(key).ok()?).1)
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


#[cfg(test)]
mod tests {
    use std::hash::RandomState;

    use crate::Node;



    // #[test]
    // pub fn t() {
    //     panic!("KEY: {}", size_of::<Node<u8, u8, RandomState>>());
    // }
}