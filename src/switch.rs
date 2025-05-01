use crate::alloc::{STACK_SIZE, alloc_stack, dealloc_stack};
use crate::interpreter::{InterpreterContext, interpret_func};
use crate::mem::get_local_fn_by_rnk;
use std::arch::global_asm;

global_asm!(include_str!("interpreter_call.S"));

#[unsafe(no_mangle)]
pub extern "C" fn asm_call_fn_handler(fn_index: u64, ctx: *mut u64) {
    println!("asm call interpreter: {}, ctx: {:#x}", fn_index, ctx as u64);
    let func = get_local_fn_by_rnk(fn_index as usize);
    let mut int_ctx = InterpreterContext::new();

    // prepare paras
    func.parameters.iter().enumerate().for_each(|(i, para)| {
        let offset = if i < 8 {
            i + 10
        } else {
            i - 8 + 32
        };
        int_ctx.virt_regs.insert(para.name.clone(), unsafe {
            *(ctx as *const i64).add(offset)
        });
    });

    let ret = interpret_func(func, &mut int_ctx);

    println!("interpreter ret asm: ret: {}", ret);

    unsafe extern "C" {
        fn __interpreter_ret_asm(ret: i64, ctx: *mut u64);
    }
    unsafe {
        __interpreter_ret_asm(ret, ctx);
    }
    panic!("should not reach here");
}

pub fn interpreter_call_asm(fn_ptr: u64, para_num: u64, para_ptr: u64) -> i64 {
    let new_sp = alloc_stack() + STACK_SIZE as u64;

    println!(
        "interpreter call asm: fn_ptr: {:#x}, para_num: {}",
        fn_ptr, para_num
    );

    unsafe extern "C" {
        fn __interpreter_call_asm(fn_ptr: u64, para_num: u64, para_ptr: u64, new_sp: u64) -> i64;
    }
    let ret = unsafe {
        __interpreter_call_asm(fn_ptr, para_num, para_ptr, new_sp)
        // 0
    };
    println!("asm ret interpreter: ret: {}", ret);
    dealloc_stack();
    ret
}
