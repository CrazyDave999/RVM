mod console;
mod asm_builder;
pub mod mem;
mod interpreter;
pub mod up;
mod alloc;
mod switch;

use llvm_ir::module::Module;
use std::path::Path;
use std::sync::Arc;
use llvm_ir::{Constant, Type, TypeRef};
use mem::{FUNC_NAME_RNK, FUNC};
use interpreter::interpret_func;
use crate::interpreter::sign_extend;
use crate::mem::get_local_fn_by_name;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let module = Module::from_ir_path(Path::new("test.ll"))?;
    for func in &module.functions {
        let cur_size = FUNC.exclusive_access().len();
        FUNC.exclusive_access().push(Arc::from(func.clone()));
        FUNC_NAME_RNK
            .exclusive_access()
            .insert(func.name.clone(), cur_size);
    }
    // Global ptr
    let mut global_inner = mem::GLOBAL_PTR.exclusive_access();
    for var in module.global_vars.iter() {
        let name = &var.name.to_string()[1..];
        match *var.ty {
            Type::IntegerType { .. } => {
                let init_val = if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Int { bits, value } => {
                            sign_extend(*bits, *value)
                        }
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    0
                };
                global_inner.insert(name.to_string(), vec![init_val]);
            }
            Type::PointerType { .. } => {
                let init_vec = if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Int { bits, value } => {
                            vec![sign_extend(*bits, *value)]
                        }
                        Constant::Array { elements, .. } => {
                            let mut vec = Vec::new();
                            for e in elements.iter() {
                                match &**e {
                                    Constant::Int { bits, value } => {
                                        let c = sign_extend(*bits, *value) as u8 as char;
                                        vec.push(sign_extend(*bits, *value));
                                    }
                                    _ => panic!("Unsupported global var initializer"),
                                }
                            }
                            vec
                        }
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    vec![0]
                };
                global_inner.insert(name.to_string(), init_vec);
            }
            Type::ArrayType { num_elements, .. } => {
                let init_vec = if let Some(val) = &var.initializer {
                    match &**val {
                        Constant::Array { elements,..} => {
                            let mut vec = Vec::new();
                            for e in elements.iter() {
                                match &**e {
                                    Constant::Int { bits, value } => {
                                        vec.push(sign_extend(*bits, *value));
                                    }
                                    _ => panic!("Unsupported global var initializer"),
                                }
                            }
                            vec
                        }
                        _ => panic!("Unsupported global var initializer"),
                    }
                } else {
                    vec![0; num_elements]
                };
                global_inner.insert(name.to_string(), init_vec);
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
