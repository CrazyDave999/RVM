mod console;
mod asm_builder;
pub mod mem;
mod interpreter;
pub mod up;
mod alloc;

use inkwell::module::Module;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::execution_engine::JitFunction;
use inkwell::OptimizationLevel;

use mem::FUNC;

type MainFunction = unsafe extern "C" fn() -> i32;


fn main() {
    let ctx = Context::create();
    // let ir_code = include_str!("../test.ll");
    // let buffer = MemoryBuffer::create_from_memory_range(ir_code.as_bytes(), "test.ll");
    // let module = ctx.create_module_from_ir(buffer).unwrap();
    let buffer = MemoryBuffer::create_from_file("test.bc".as_ref()).unwrap();
    let module = Module::parse_bitcode_from_buffer(&buffer, &ctx).unwrap();

    for func in module.get_functions() {
        FUNC.exclusive_access().push(func);
        for bb in func.get_basic_blocks() {
            for instr in bb.get_instructions() {

            }
        }
    }



    unsafe {
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let main_func: JitFunction<MainFunction> = execution_engine
            .get_function("main")
            .expect("Could not find the `main` function!");

        // 调用 @main 函数
        let result = main_func.call();

        println!("The result of `main` is: {}", result);
    }
}
