const FUNC_MAX_NUM: usize = 10000;

/// table of function pointers
#[no_mangle]
static mut FUNC_TABLE: [u64; FUNC_MAX_NUM] = [0; FUNC_MAX_NUM];


pub fn init() {

}