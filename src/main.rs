mod console;
mod asm_builder;
pub mod mem;
mod interpreter;
pub mod up;
mod alloc;
mod switch;

use llvm_ir::module::Module;
use std::path::Path;


use mem::{FUNC_NAME_RNK, FUNC};
use interpreter::interpret_func;



fn main() -> Result<(), Box<dyn std::error::Error>> {
    let module = Module::from_ir_path(Path::new("test.ll"))?;
    for func in &module.functions {
        let cur_size = FUNC.exclusive_access().len();
        FUNC.exclusive_access().push(func.clone());
        FUNC_NAME_RNK
            .exclusive_access()
            .insert(func.name.clone(), cur_size);
    }
    interpret_func(&module.functions[0]);
    Ok(())
}
