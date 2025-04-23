use crate::alloc::{alloc_stack, dealloc_stack};
use core::arch::asm;
pub fn interpret_func() -> isize {
    0
}

#[no_mangle]
pub extern "C" fn asm_call_fn_handler(fn_index: usize, ctx: *mut usize) {
    // todo

    let ret = interpret_func();
    extern "C" {
        fn __interpreter_ret_asm(ret: isize, ctx: *mut usize);
    }
    unsafe {
        __interpreter_ret_asm(ret, ctx);
    }
    panic!("should not reach here");
}

pub fn interpreter_call_asm(fn_index: usize, para_num: usize, para_ptr: usize) -> isize {
    extern "C" {
        fn __interpreter_call_asm(
            fn_ptr: usize,
            para_num: usize,
            para_ptr: usize,
            new_sp: usize,
        ) -> isize;
    }

    let new_sp = alloc_stack();
    let ret = unsafe { __interpreter_call_asm(fn_index, para_num, para_ptr, new_sp) };
    dealloc_stack();
    ret
}
