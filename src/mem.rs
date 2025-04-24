use std::collections::HashMap;
use super::up::UPSafeCell;
use llvm_ir::Function;
use lazy_static::lazy_static;

const FUNC_MAX_NUM: usize = 10000;

/// table of function pointers
#[unsafe(no_mangle)]
static mut FUNC_TABLE: [usize; FUNC_MAX_NUM] = [0; FUNC_MAX_NUM];

lazy_static! {
    pub static ref FUNC: UPSafeCell<Vec<Function>> =
        unsafe { UPSafeCell::new(Vec::new()) };
    pub static ref FUNC_NAME_RNK: UPSafeCell<HashMap<String, usize>> =
        unsafe { UPSafeCell::new(HashMap::new()) };
}

pub fn init() {}
