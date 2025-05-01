use crate::asm_builder::compile_func;
use crate::mem::{FUNC_TABLE, HOTNESS, get_local_fn_by_rnk, get_local_rnk};
use crate::switch::interpreter_call_asm;
use llvm_ir::{BasicBlock, Constant, Instruction, IntPredicate, Type};
use llvm_ir::{Function, Name, Operand, Terminator};
use std::collections::HashMap;
use std::sync::Arc;

pub const CRITICAL_HOTNESS: i64 = 0;

pub struct InterpreterContext {
    pub virt_regs: HashMap<Name, i64>,
    pub stack: Vec<i64>,
    pub last_bb: Name,
}
impl InterpreterContext {
    pub fn new() -> Self {
        InterpreterContext {
            virt_regs: HashMap::new(),
            stack: Vec::new(),
            last_bb: Name::from("entry"),
        }
    }
    pub fn get_operand(&self, op: &Operand) -> i64 {
        match op {
            Operand::ConstantOperand(c) => match &**c {
                Constant::Int { bits, value } => sign_extend(*bits, *value),
                Constant::GlobalReference { name, .. } => {
                    let global_inner = crate::mem::GLOBAL_PTR.exclusive_access();
                    if let Some(ptr) = global_inner.get(name) {
                        *ptr as i64
                    } else {
                        panic!("Global variable not found");
                    }
                }
                _ => panic!("Unsupported constant type"),
            },
            Operand::LocalOperand { name, .. } => *self
                .virt_regs
                .get(name)
                .unwrap_or_else(|| panic!("Virtual register not found: {}", name)),
            _ => {
                panic!("Unsupported operand type");
            }
        }
    }
    pub fn alloc(&mut self, size: usize) -> i64 {
        let addr = self.stack.len();
        for _ in 0..size {
            self.stack.push(0);
        }
        addr as i64
    }
}

pub fn interpret_func(func: Arc<Function>, ctx: &mut InterpreterContext) -> i64 {
    let mut cur_bb = func.get_bb_by_name(&Name::from("entry")).unwrap();
    loop {
        interpret_bb(cur_bb, ctx);
        ctx.last_bb = cur_bb.name.clone();
        match &cur_bb.term {
            Terminator::Br(br) => {
                if let Some(bb) = func.get_bb_by_name(&br.dest) {
                    cur_bb = bb;
                } else {
                    panic!("BasicBlock not found");
                }
            }
            Terminator::CondBr(cond_br) => {
                let cond = ctx.get_operand(&cond_br.condition);
                if cond != 0 {
                    if let Some(bb) = func.get_bb_by_name(&cond_br.true_dest) {
                        cur_bb = bb;
                    } else {
                        panic!("BasicBlock not found");
                    }
                } else {
                    if let Some(bb) = func.get_bb_by_name(&cond_br.false_dest) {
                        cur_bb = bb;
                    } else {
                        panic!("BasicBlock not found");
                    }
                }
            }
            Terminator::Ret(ret) => {
                return if let Some(ret_op) = &ret.return_operand {
                    ctx.get_operand(ret_op)
                } else {
                    0
                };
            }
            _ => panic!("Unsupported terminator type"),
        }
    }
}
pub fn interpret_bb(bb: &BasicBlock, ctx: &mut InterpreterContext) {
    for inst in bb.instrs.iter() {
        interpret_inst(inst, ctx);
    }
}

pub fn interpret_inst(inst: &Instruction, ctx: &mut InterpreterContext) {
    match inst {
        // Integer binary ops
        Instruction::Add(add) => {
            let lhs = ctx.get_operand(&add.operand0);
            let rhs = ctx.get_operand(&add.operand1);
            let res = lhs + rhs;
            ctx.virt_regs.insert(add.dest.clone(), res);
        }
        Instruction::Sub(sub) => {
            let lhs = ctx.get_operand(&sub.operand0);
            let rhs = ctx.get_operand(&sub.operand1);
            let res = lhs - rhs;
            ctx.virt_regs.insert(sub.dest.clone(), res);
        }
        Instruction::Mul(mul) => {
            let lhs = ctx.get_operand(&mul.operand0);
            let rhs = ctx.get_operand(&mul.operand1);
            let res = lhs * rhs;
            ctx.virt_regs.insert(mul.dest.clone(), res);
        }
        Instruction::SDiv(sdiv) => {
            let lhs = ctx.get_operand(&sdiv.operand0);
            let rhs = ctx.get_operand(&sdiv.operand1);
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs / rhs;
            ctx.virt_regs.insert(sdiv.dest.clone(), res);
        }
        Instruction::UDiv(udiv) => {
            let lhs = ctx.get_operand(&udiv.operand0) as usize;
            let rhs = ctx.get_operand(&udiv.operand1) as usize;
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs / rhs;
            ctx.virt_regs.insert(udiv.dest.clone(), res as i64);
        }
        Instruction::SRem(srem) => {
            let lhs = ctx.get_operand(&srem.operand0);
            let rhs = ctx.get_operand(&srem.operand1);
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs % rhs;
            ctx.virt_regs.insert(srem.dest.clone(), res);
        }
        Instruction::URem(urem) => {
            let lhs = ctx.get_operand(&urem.operand0) as usize;
            let rhs = ctx.get_operand(&urem.operand1) as usize;
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs % rhs;
            ctx.virt_regs.insert(urem.dest.clone(), res as i64);
        }

        // Bitwise binary ops
        Instruction::And(and) => {
            let lhs = ctx.get_operand(&and.operand0);
            let rhs = ctx.get_operand(&and.operand1);
            let res = lhs & rhs;
            ctx.virt_regs.insert(and.dest.clone(), res);
        }
        Instruction::Or(or) => {
            let lhs = ctx.get_operand(&or.operand0);
            let rhs = ctx.get_operand(&or.operand1);
            let res = lhs | rhs;
            ctx.virt_regs.insert(or.dest.clone(), res);
        }
        Instruction::Xor(xor) => {
            let lhs = ctx.get_operand(&xor.operand0);
            let rhs = ctx.get_operand(&xor.operand1);
            let res = lhs ^ rhs;
            ctx.virt_regs.insert(xor.dest.clone(), res);
        }
        Instruction::Shl(shl) => {
            let lhs = ctx.get_operand(&shl.operand0);
            let rhs = ctx.get_operand(&shl.operand1);
            let res = lhs << rhs;
            ctx.virt_regs.insert(shl.dest.clone(), res);
        }
        Instruction::LShr(lshr) => {
            let lhs = ctx.get_operand(&lshr.operand0) as usize;
            let rhs = ctx.get_operand(&lshr.operand1) as usize;
            let res = lhs >> rhs;
            ctx.virt_regs.insert(lshr.dest.clone(), res as i64);
        }
        Instruction::AShr(ashr) => {
            let lhs = ctx.get_operand(&ashr.operand0);
            let rhs = ctx.get_operand(&ashr.operand1);
            let res = lhs >> rhs;
            ctx.virt_regs.insert(ashr.dest.clone(), res);
        }
        // Memory-related ops
        Instruction::Alloca(alloca) => {
            let size = match *alloca.allocated_type {
                Type::IntegerType { .. } => 1,
                Type::ArrayType { num_elements, .. } => num_elements,
                _ => panic!("Unsupported alloca type"),
            };
            let addr = ctx.alloc(size);
            ctx.virt_regs.insert(alloca.dest.clone(), addr);
        }
        Instruction::Load(load) => {
            let addr = ctx.get_operand(&load.address);
            let res = ctx.stack[addr as usize];
            ctx.virt_regs.insert(load.dest.clone(), res);
        }
        Instruction::Store(store) => {
            let addr = ctx.get_operand(&store.address);
            let value = ctx.get_operand(&store.value);
            ctx.stack[addr as usize] = value;
        }
        Instruction::GetElementPtr(get_elem_ptr) => {
            // only 1 dimension GEP is supported
            let addr = ctx.get_operand(&get_elem_ptr.address);
            let indices = get_elem_ptr
                .indices
                .iter()
                .map(|op| ctx.get_operand(op))
                .collect::<Vec<i64>>();
            if indices.len() != 2 {
                panic!("Unsupported GEP indices");
            }
            ctx.virt_regs
                .insert(get_elem_ptr.dest.clone(), addr + indices[1]);
        }

        // LLVM's "other operations" category
        Instruction::ICmp(icmp) => {
            let lhs = ctx.get_operand(&icmp.operand0);
            let rhs = ctx.get_operand(&icmp.operand1);
            let res = match icmp.predicate {
                IntPredicate::EQ => (lhs == rhs) as i64,
                IntPredicate::NE => (lhs != rhs) as i64,
                IntPredicate::UGT => ((lhs as usize) > (rhs as usize)) as i64,
                IntPredicate::UGE => ((lhs as usize) >= (rhs as usize)) as i64,
                IntPredicate::ULT => ((lhs as usize) < (rhs as usize)) as i64,
                IntPredicate::ULE => ((lhs as usize) <= (rhs as usize)) as i64,
                IntPredicate::SGT => (lhs > rhs) as i64,
                IntPredicate::SGE => (lhs >= rhs) as i64,
                IntPredicate::SLT => (lhs < rhs) as i64,
                IntPredicate::SLE => (lhs <= rhs) as i64,
            };
            ctx.virt_regs.insert(icmp.dest.clone(), res);
        }
        Instruction::Phi(phi) => {
            let mut res = 0;
            for (op, bb) in phi.incoming_values.iter() {
                if *bb == ctx.last_bb {
                    res = ctx.get_operand(op);
                    break;
                }
            }
            ctx.virt_regs.insert(phi.dest.clone(), res);
        }
        Instruction::Select(select) => {
            let cond = ctx.get_operand(&select.condition);
            let res = if cond != 0 {
                ctx.get_operand(&select.true_value)
            } else {
                ctx.get_operand(&select.false_value)
            };
            ctx.virt_regs.insert(select.dest.clone(), res);
        }
        Instruction::Call(call) => {
            match call.function.clone().right().unwrap() {
                Operand::ConstantOperand(const_ref) => match &*const_ref {
                    Constant::GlobalReference { name, .. } => {
                        let name = &name.to_string()[1..];
                        let mut args = call
                            .arguments
                            .iter()
                            .map(|(op, _)| ctx.get_operand(op))
                            .collect::<Vec<i64>>();
                        let para_num = args.len() as u64;
                        let mut ret = 0;
                        if let Some(rnk) = get_local_rnk(name) {
                            // this name is some function like define i32 @func
                            let addr = unsafe { FUNC_TABLE[rnk] };
                            if addr != 0 {
                                // this function has been compiled
                                println!("Calling a compiled function: {}", name);
                                if args.len() < 8 {
                                    args.resize(8, 0);
                                }
                                ret = interpreter_call_asm(addr, para_num, args.as_ptr() as u64);
                            } else {
                                // local function
                                let mut hotness = HOTNESS.exclusive_access();
                                if hotness[&(rnk as u64)] >= CRITICAL_HOTNESS {
                                    // compile, then call its asm
                                    println!("Compiling function: {}", name);
                                    let addr = compile_func(get_local_fn_by_rnk(rnk));
                                    unsafe {
                                        FUNC_TABLE[rnk] = addr;
                                    }
                                    drop(hotness);
                                    println!("Calling a compiled function: {}", name);
                                    if args.len() < 8 {
                                        args.resize(8, 0);
                                    }
                                    ret =
                                        interpreter_call_asm(addr, para_num, args.as_ptr() as u64);
                                } else {
                                    if hotness[&(rnk as u64)] >= 0 {
                                        // functions with negative hotness will never be compiled
                                        *hotness.get_mut(&(rnk as u64)).unwrap() += 1;
                                    }
                                    drop(hotness);
                                    let func = get_local_fn_by_rnk(rnk);
                                    let mut new_ctx = InterpreterContext::new();
                                    func.parameters.iter().zip(call.arguments.iter()).for_each(
                                        |(para, (arg, _))| {
                                            let arg_val = ctx.get_operand(&arg);
                                            new_ctx.virt_regs.insert(para.name.clone(), arg_val);
                                        },
                                    );
                                    ret = interpret_func(func, &mut new_ctx);
                                }
                            }
                        } else {
                            // extern function
                            ret = interpret_extern_func(name, args);
                        }
                        if let Some(dest) = &call.dest {
                            ctx.virt_regs.insert(dest.clone(), ret);
                        }
                    }
                    _ => panic!("Unsupported function type"),
                },
                _ => panic!("Unsupported function type"),
            }
        }

        _ => {
            println!("Unsupported instruction: {:?}", inst);
        }
    }
}

pub fn interpret_extern_func(name: &str, paras: Vec<i64>) -> i64 {
    match name {
        "..print" | "..println" => {
            for para in paras.iter() {
                unsafe {
                    let mut p = (*para) as *const i64;
                    loop {
                        if *p == 0 {
                            break;
                        }
                        print!("{}", (*p) as u8 as char);
                        p = p.add(1);
                    }
                }
            }
            if name == "..println" {
                println!();
            }
            0
        }
        "..printInt" | "..printlnInt" => {
            for para in paras.iter() {
                print!("{}", *para);
            }
            if name == "..printlnInt" {
                println!();
            }
            0
        }
        _ => panic!("Undefined function"),
    }
}

pub fn sign_extend(bits: u32, value: u64) -> i64 {
    if bits < 64 {
        let sign_bit = value & (1u64 << (bits - 1));
        if sign_bit != 0 {
            (value | !((1u64 << bits) - 1)) as i64
        } else {
            value as i64
        }
    } else {
        value as i64
    }
}
