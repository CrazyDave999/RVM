use crate::alloc::{alloc_stack, dealloc_stack};
use crate::interpreter::{interpret_func};
use crate::mem::{FUNC};

#[unsafe(no_mangle)]
pub extern "C" fn asm_call_fn_handler(fn_index: usize, ctx: *mut usize) {
    // todo
    let func = &FUNC.exclusive_access()[fn_index];
    let ret = interpret_func(func);
    unsafe extern "C" {
        fn __interpreter_ret_asm(ret: i64, ctx: *mut usize);
    }
    unsafe {

        __interpreter_ret_asm(ret, ctx);
    }
    panic!("should not reach here");
}

pub fn interpreter_call_asm(fn_index: usize, para_num: usize, para_ptr: usize) -> isize {
    let new_sp = alloc_stack();
    unsafe extern "C" {
        fn __interpreter_call_asm(
            fn_ptr: usize,
            para_num: usize,
            para_ptr: usize,
            new_sp: usize,
        ) -> isize;
    }
    let ret = unsafe {
        __interpreter_call_asm(fn_index, para_num, para_ptr, new_sp)
    };
    dealloc_stack();
    ret
}
