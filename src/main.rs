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
use mem::{FUNC_NAME_RNK, FUNC};
use interpreter::interpret_func;
use crate::mem::get_fn_by_name;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let module = Module::from_ir_path(Path::new("test.ll"))?;
    for func in &module.functions {
        let cur_size = FUNC.exclusive_access().len();
        FUNC.exclusive_access().push(Arc::from(func.clone()));
        FUNC_NAME_RNK
            .exclusive_access()
            .insert(func.name.clone(), cur_size);
    }
    let mut ctx = interpreter::InterpreterContext::new();
    let ret = interpret_func(get_fn_by_name("main"),&mut ctx);
    println!("main return: {:?}", ret);
    Ok(())
}
