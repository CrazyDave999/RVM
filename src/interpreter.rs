use crate::mem::get_local_fn_by_name;
use llvm_ir::{BasicBlock, Constant, Instruction, IntPredicate};
use llvm_ir::{Function, Name, Operand, Terminator};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;
use libc::c_char;

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
                Constant::Int { bits, value } => {
                    sign_extend(*bits, *value)
                }
                Constant::GlobalReference {name, ..} => {
                    let global_inner = crate::mem::GLOBAL_PTR.exclusive_access();
                    if let Some(vec) = global_inner.get(&name.to_string()[1..]) {
                        vec.as_ptr() as i64
                    } else {
                        panic!("Global variable not found");
                    }
                }
                _ => panic!("Unsupported constant type"),
            },
            Operand::LocalOperand { name, .. } => *self.virt_regs.get(name).unwrap(),
            _ => {
                panic!("Unsupported operand type");
            }
        }
    }
    pub fn alloc(&mut self) -> i64 {
        let addr = self.stack.len();
        self.stack.push(0);
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
            let addr = ctx.alloc();
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
                        if let Some(func) = get_local_fn_by_name(&name.to_string()[1..]) {
                            let mut new_ctx = InterpreterContext::new();
                            func.parameters
                                .iter()
                                .zip(call.arguments.iter())
                                .for_each(|(para, (arg, _))| {
                                    let arg_val = ctx.get_operand(&arg);
                                    new_ctx.virt_regs.insert(para.name.clone(), arg_val);
                                });
                            let ret = interpret_func(func, &mut new_ctx);
                            if let Some(dest) = &call.dest {
                                ctx.virt_regs.insert(dest.clone(), ret);
                            }
                        } else {
                            let paras = call
                                .arguments
                                .iter()
                                .map(|(op, _)| ctx.get_operand(op))
                                .collect::<Vec<i64>>();
                            let ret = interpret_extern_func(&name.to_string()[1..], paras);
                            if let Some(dest) = &call.dest {
                                ctx.virt_regs.insert(dest.clone(), ret);
                            }
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
    if name == "print" {
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
        0
    } else {
        panic!("Undefined function");
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