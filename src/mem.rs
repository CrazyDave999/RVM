const FUNC_MAX_NUM: usize = 10000;

/// table of function pointers
#[no_mangle]
static mut FUNC_TABLE: [usize; FUNC_MAX_NUM] = [0; FUNC_MAX_NUM];

const KERNEL_INFO_SIZE: usize = 4;
/// some info of kernel
#[no_mangle]
pub static mut KERNEL_INFO: [usize; KERNEL_INFO_SIZE] = [0; KERNEL_INFO_SIZE];


pub fn init() {

}