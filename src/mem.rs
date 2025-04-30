use super::up::UPSafeCell;
use lazy_static::lazy_static;
use llvm_ir::{Function, Name};
use std::collections::HashMap;
use std::sync::Arc;

pub const FUNC_MAX_NUM: usize = 10000;

/// table of function pointers
#[unsafe(no_mangle)]
pub static mut FUNC_TABLE: [u64; FUNC_MAX_NUM] = [0; FUNC_MAX_NUM];

lazy_static! {
    pub static ref FUNC: UPSafeCell<Vec<Arc<Function>>> = unsafe { UPSafeCell::new(Vec::new()) };
    pub static ref FUNC_NAME_RNK: UPSafeCell<HashMap<String, usize>> =
        unsafe { UPSafeCell::new(HashMap::new()) };
    pub static ref GLOBAL_PTR: UPSafeCell<HashMap<Name, u64>> =
        unsafe { UPSafeCell::new(HashMap::new()) };
    pub static ref HOTNESS: UPSafeCell<HashMap<u64, i64>> =
        unsafe { UPSafeCell::new(HashMap::new()) };
}

pub fn get_local_fn_by_name(name: &str) -> Option<Arc<Function>> {
    let func_name_inner = FUNC_NAME_RNK.exclusive_access();
    let index = func_name_inner.get(name)?;
    let func_inner = FUNC.exclusive_access();
    func_inner.get(*index).cloned()
}
pub fn get_local_fn_by_rnk(rnk: usize) -> Arc<Function> {
    let func_inner = FUNC.exclusive_access();
    func_inner.get(rnk).cloned().unwrap()
}

pub fn get_local_rnk(name: &str) -> Option<usize> {
    let func_name_inner = FUNC_NAME_RNK.exclusive_access();
    let index = func_name_inner.get(name)?;
    Some(*index)
}
