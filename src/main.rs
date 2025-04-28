mod alloc;
mod asm_builder;
mod console;
mod interpreter;
pub mod mem;
mod switch;
pub mod up;

use crate::interpreter::sign_extend;
use crate::mem::get_local_fn_by_name;
use interpreter::interpret_func;
use llvm_ir::module::Module;
use llvm_ir::{Constant, Type, TypeRef};
use mem::{FUNC, FUNC_NAME_RNK};
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let module = Module::from_ir_path(Path::new("test.ll"))?;
    for func in &module.functions {
        let cur_size = FUNC.exclusive_access().len();
        FUNC.exclusive_access().push(Arc::from(func.clone()));
        FUNC_NAME_RNK
            .exclusive_access()
            .insert(func.name.clone(), cur_size);
    }

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
    println!("Global var start at: {:x}, size: {}", global_var_ptr, global_var_size);

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
    let ret = interpret_func(get_local_fn_by_name("main").unwrap(), &mut ctx);
    println!("[RVM] main return: {:?}", ret);
    Ok(())
}
