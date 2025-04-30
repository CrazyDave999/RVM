use std::arch::global_asm;
use crate::alloc::{alloc_stack, dealloc_stack};
use crate::interpreter::{interpret_func, InterpreterContext};
use crate::mem::{get_local_fn_by_rnk};

global_asm!(include_str!("interpreter_call.S"));

#[unsafe(no_mangle)]
pub extern "C" fn asm_call_fn_handler(fn_index: u64, ctx: *mut u64) {
    let func = get_local_fn_by_rnk(fn_index as usize);
    let mut int_ctx = InterpreterContext::new();
    let ret = interpret_func(func, &mut int_ctx);

    unsafe extern "C" {
        fn __interpreter_ret_asm(ret: i64, ctx: *mut u64);
    }
    unsafe {
        __interpreter_ret_asm(ret, ctx);
    }
    panic!("should not reach here");
}

pub fn interpreter_call_asm(fn_ptr: u64, para_num: u64, para_ptr: u64) -> i64 {
    let new_sp = alloc_stack();
    unsafe extern "C" {
        fn __interpreter_call_asm(
            fn_ptr: u64,
            para_num: u64,
            para_ptr: u64,
            new_sp: u64,
        ) -> i64;
    }
    let ret = unsafe {
        __interpreter_call_asm(fn_ptr, para_num, para_ptr, new_sp)
        // 0
    };
    dealloc_stack();
    ret
}
