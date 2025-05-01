mod alloc;
mod asm_builder;
mod console;
mod interpreter;
pub mod mem;
mod switch;
pub mod up;

use crate::interpreter::sign_extend;
use crate::mem::{HOTNESS, get_local_fn_by_name, get_local_rnk};
use interpreter::interpret_func;
use llvm_ir::module::Module;
use llvm_ir::{Constant, Instruction, Type};
use mem::{FUNC, FUNC_NAME_RNK};
use std::path::Path;
use std::sync::Arc;
use asm_builder::__asm_call_fn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let module = Module::from_ir_path(Path::new("test.ll"))?;
    for (i, func) in module.functions.iter().enumerate() {
        FUNC.exclusive_access().push(Arc::from(func.clone()));
        FUNC_NAME_RNK
            .exclusive_access()
            .insert(func.name.clone(), i);
    }
    println!("Functions: ");
    for (i, func) in module.functions.iter().enumerate() {
        println!(
            "{}, {}, {}",
            i,
            func.name,
            get_local_rnk(&func.name).unwrap()
        );
    }
    println!();

    // calculate total mem size that global vars need
    let mut global_var_size = 0usize;
    for var in module.global_vars.iter() {
        match *var.ty {
            Type::IntegerType { .. } => {
                global_var_size += 8;
            }
            Type::PointerType { .. } => {
                if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Int { .. } => {
                            global_var_size += 8;
                        }
                        Constant::Array { elements, .. } => {
                            global_var_size += elements.len() * 8;
                        }
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    global_var_size += 8;
                };
            }
            Type::ArrayType { num_elements, .. } => {
                global_var_size += num_elements * 8;
            }
            _ => panic!("Unsupported global variable type"),
        }
    }

    // allocate mem for global vars
    let mut global_var_ptr = alloc::alloc_mem(global_var_size);
    println!(
        "Global var start at: {:x}, size: {}",
        global_var_ptr, global_var_size
    );
    
    println!("__asm_call_fn is at: {:#x}", __asm_call_fn as u64);
    println!();
    

    // init global vars
    let mut global_inner = mem::GLOBAL_PTR.exclusive_access();
    for var in module.global_vars.iter() {
        global_inner.insert(var.name.clone(), global_var_ptr);
        match *var.ty {
            Type::IntegerType { .. } => {
                let init_val = if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Int { bits, value } => sign_extend(*bits, *value),
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    0
                };

                unsafe {
                    *(global_var_ptr as *mut i64) = init_val;
                }
                global_var_ptr += 8;
            }
            Type::PointerType { .. } => {
                if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Int { bits, value } => {
                            unsafe {
                                *(global_var_ptr as *mut i64) = sign_extend(*bits, *value);
                            }
                            global_var_ptr += 8;
                        }
                        Constant::Array { elements, .. } => {
                            for e in elements.iter() {
                                match &**e {
                                    Constant::Int { bits, value } => {
                                        // let c = sign_extend(*bits, *value) as u8 as char;
                                        unsafe {
                                            *(global_var_ptr as *mut i64) =
                                                sign_extend(*bits, *value);
                                        }
                                        global_var_ptr += 8;
                                    }
                                    _ => panic!("Unsupported global var initializer"),
                                }
                            }
                        }
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    unsafe {
                        *(global_var_ptr as *mut i64) = 0;
                    }
                    global_var_ptr += 8;
                };
            }
            Type::ArrayType { num_elements, .. } => {
                if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Array { elements, .. } => {
                            for e in elements.iter() {
                                match &**e {
                                    Constant::Int { bits, value } => {
                                        unsafe {
                                            *(global_var_ptr as *mut i64) =
                                                sign_extend(*bits, *value);
                                        }
                                        global_var_ptr += 8;
                                    }
                                    _ => panic!("Unsupported global var initializer"),
                                }
                            }
                        }
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    for _ in 0..num_elements {
                        unsafe {
                            *(global_var_ptr as *mut i64) = 0;
                        }
                        global_var_ptr += 8;
                    }
                };
            }
            _ => panic!("Unsupported global var type"),
        }
    }
    drop(global_inner);

    let mut ctx = interpreter::InterpreterContext::new();

    // prepare hotness, those with negative hotness will never be compiled
    // 2 kinds of functions cannot be compiled
    // 1. functions with phi/select
    // 2. functions that call extern functions
    let mut hotness = HOTNESS.exclusive_access();
    for func in module.functions.iter() {
        let rnk = get_local_rnk(&func.name).unwrap();
        if &func.name == "main" {
            hotness.insert(rnk as u64, -1);
            continue;
        }
        let mut can_compile = true;
        for basic_block in func.basic_blocks.iter() {
            for inst in basic_block.instrs.iter() {
                match inst {
                    Instruction::Phi(_) | Instruction::Select(_) | Instruction::Mul(_) | Instruction::SDiv(_) | Instruction::UDiv(_)  => {
                        can_compile = false;
                        break;
                    }
                    Instruction::Call(call) => {
                        let func_name = &call.function.clone().right().unwrap().to_string()[1..];
                        if let Some(_) = get_local_rnk(func_name) {
                        } else {
                            can_compile = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if !can_compile {
                break;
            }
        }
        if can_compile {
            hotness.insert(rnk as u64, 0);
        } else {
            hotness.insert(rnk as u64, -1);
        }
    }
    drop(hotness);

    let ret = interpret_func(get_local_fn_by_name("main").unwrap(), &mut ctx);
    println!("[RVM] main return: {:?}", ret);
    Ok(())
}
