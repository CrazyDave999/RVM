use super::up::UPSafeCell;
use inkwell::values::FunctionValue;
use lazy_static::lazy_static;

const FUNC_MAX_NUM: usize = 10000;

/// table of function pointers
#[no_mangle]
static mut FUNC_TABLE: [usize; FUNC_MAX_NUM] = [0; FUNC_MAX_NUM];

/// some info of kernel
#[no_mangle]
pub static mut KERNEL_CTX: [usize; 32] = [0; 32];

lazy_static! {
    pub static ref FUNC: UPSafeCell<Vec<FunctionValue<'static>>> =
        unsafe { UPSafeCell::new(Vec::new()) };
}

pub fn init() {}
