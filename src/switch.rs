use crate::alloc::{alloc_stack, dealloc_stack};
use crate::interpreter::{interpret_func, InterpreterContext};
use crate::mem::{FUNC};

#[unsafe(no_mangle)]
pub extern "C" fn asm_call_fn_handler(fn_index: u64, ctx: *mut u64) {
    // todo
    let func = &FUNC.exclusive_access()[fn_index as usize];
    let mut int_ctx = InterpreterContext::new();
    let ret = interpret_func(func.clone(), &mut int_ctx);
    
    unsafe extern "C" {
        fn __interpreter_ret_asm(ret: i64, ctx: *mut u64);
    }
    unsafe {
        __interpreter_ret_asm(ret, ctx);
    }
    panic!("should not reach here");
}

pub fn interpreter_call_asm(fn_index: u64, para_num: u64, para_ptr: u64) -> i64 {
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
        __interpreter_call_asm(fn_index, para_num, para_ptr, new_sp)
    };
    dealloc_stack();
    ret
}
