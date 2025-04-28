use crate::alloc::alloc_mem;
use crate::interpreter::sign_extend;
use crate::mem::GLOBAL_PTR;
use llvm_ir::{Constant, Function, Instruction, Name, Operand, Terminator, Type};
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::sync::Arc;

static PHY_REGS: [&'static str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];
struct PhyReg(&'static str);
impl PhyReg {
    pub fn to_binary(&self) -> u32 {
        if self.0.starts_with('x') {
            self.0[1..].parse::<u32>().unwrap()
        } else {
            match self.0 {
                "zero" => 0,
                "ra" => 1,
                "sp" => 2,
                "gp" => 3,
                "tp" => 4,
                "t0" => 5,
                "t1" => 6,
                "t2" => 7,
                "s0" => 8,
                "s1" => 9,
                "a0" => 10,
                "a1" => 11,
                "a2" => 12,
                "a3" => 13,
                "a4" => 14,
                "a5" => 15,
                "a6" => 16,
                "a7" => 17,
                "s2" => 18,
                "s3" => 19,
                "s4" => 20,
                "s5" => 21,
                "s6" => 22,
                "s7" => 23,
                "s8" => 24,
                "s9" => 25,
                "s10" => 26,
                "s11" => 27,
                "t3" => 28,
                "t4" => 29,
                "t5" => 30,
                "t6" => 31,
                _ => panic!("Unsupported register name"),
            }
        }
    }
}
enum InstType {
    R,
    I,
    IStar,
    L, // i.e. I, but different opcode
    S,
    B,
    U,
    J,
}
struct ASMInst {
    name: &'static str,
    rs1: PhyReg,
    rs2: PhyReg,
    rd: PhyReg,
    ty: InstType,
    imm: u32,
    label: Option<String>,
}

impl ASMInst {
    pub fn from_inst(ir_inst: &Instruction, ctx: &ASMContext) -> Vec<Self> {
        let mut res = Vec::new();
        match ir_inst {
            Instruction::Add(add) => {
                res.extend(Self::binary(
                    "add",
                    &add.dest,
                    &add.operand0,
                    &add.operand1,
                    ctx,
                ));
            }
            Instruction::Sub(sub) => {
                res.extend(Self::binary(
                    "sub",
                    &sub.dest,
                    &sub.operand0,
                    &sub.operand1,
                    ctx,
                ));
            }
            Instruction::And(and) => {
                res.extend(Self::binary(
                    "and",
                    &and.dest,
                    &and.operand0,
                    &and.operand1,
                    ctx,
                ));
            }
            Instruction::Or(or) => {
                res.extend(Self::binary(
                    "or",
                    &or.dest,
                    &or.operand0,
                    &or.operand1,
                    ctx,
                ));
            }
            Instruction::Xor(xor) => {
                res.extend(Self::binary(
                    "xor",
                    &xor.dest,
                    &xor.operand0,
                    &xor.operand1,
                    ctx,
                ));
            }
            Instruction::Shl(shl) => {
                res.extend(Self::binary(
                    "sll",
                    &shl.dest,
                    &shl.operand0,
                    &shl.operand1,
                    ctx,
                ));
            }
            Instruction::LShr(lshr) => {
                res.extend(Self::binary(
                    "srl",
                    &lshr.dest,
                    &lshr.operand0,
                    &lshr.operand1,
                    ctx,
                ));
            }
            Instruction::AShr(ashr) => {
                res.extend(Self::binary(
                    "sra",
                    &ashr.dest,
                    &ashr.operand0,
                    &ashr.operand1,
                    ctx,
                ));
            }

            Instruction::Alloca(_) => {
                // do nothing
            }
            Instruction::Load(_) => {
                // do nothing
            }
            Instruction::Store(store) => {
                res.extend(Self::get_operand("t0", &store.value, ctx, false));
                match &store.address {
                    Operand::LocalOperand { name, .. } => {
                        let addr = *ctx.local_ptrs.get(&name).unwrap();
                        res.extend(Self::sd("t0", "sp", addr as i64));
                    }
                    Operand::ConstantOperand(c) => match &**c {
                        Constant::GlobalReference { name, .. } => {
                            let global_inner = GLOBAL_PTR.exclusive_access();
                            let addr = *global_inner.get(name).unwrap();
                            res.extend(Self::addi("t1", "x0", addr as i64));
                            res.extend(Self::sd("t0", "t1", 0));
                        }
                        _ => panic!("Unsupported constant store"),
                    },
                    _ => panic!("Unsupported operand for store"),
                };
            }
            Instruction::GetElementPtr(_) => {}

            Instruction::ICmp(_) => {}

            Instruction::Call(_) => {}
            _ => panic!("Unsupported instruction: {:?}", ir_inst),
        }
        res
    }
    pub fn from_terminator(term: &Terminator, ctx: &ASMContext) -> Vec<Self> {
        let res = Vec::new();

        res
    }
    fn to_binary(&self) -> u32 {
        let opcode = self.get_opcode();
        let (funct3, funct7) = self.get_funct();
        let rs1 = self.rs1.to_binary();
        let rs2 = self.rs2.to_binary();
        let rd = self.rd.to_binary();
        let shifted_imm = self.get_shifted_imm();
        match self.ty {
            InstType::R => {
                opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
            }
            InstType::I | InstType::L => {
                opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | shifted_imm
            }
            InstType::IStar => {
                opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | shifted_imm | (funct7 << 25)
            }
            InstType::S => opcode | shifted_imm | (funct3 << 12) | (rs1 << 15) | (rs2 << 20),
            InstType::B => opcode | shifted_imm | (funct3 << 12) | (rs1 << 15) | (rs2 << 20),
            InstType::U => opcode | (rd << 7) | shifted_imm,
            InstType::J => opcode | (rd << 7) | shifted_imm,
        }
    }
    fn to_string(&self) -> String {
        String::new()
    }
    fn get_funct(&self) -> (u32, u32) {
        match self.name {
            "add" => (0b000, 0b000_0000),
            "sub" => (0b000, 0b010_0000),
            "and" => (0b111, 0b000_0000),
            "or" => (0b110, 0b000_0000),
            "xor" => (0b100, 0b000_0000),
            "sll" => (0b001, 0b000_0000),
            "srl" => (0b101, 0b000_0000),
            "sra" => (0b101, 0b010_0000),
            "slt" => (0b010, 0b000_0000),
            "sltu" => (0b011, 0b000_0000),

            "addi" => (0b000, 0b000_0000),
            "andi" => (0b111, 0b000_0000),
            "ori" => (0b110, 0b000_0000),
            "xori" => (0b100, 0b000_0000),

            "slli" => (0b001, 0b000_0000),
            "srli" => (0b101, 0b000_0000),
            "srai" => (0b101, 0b010_0000),

            "slti" => (0b010, 0b000_0000),
            "sltiu" => (0b011, 0b000_0000),

            "ld" => (0b011, 0b000_0011),

            "sd" => (0b011, 0b010_0011),

            "beq" => (0b000, 0b000_0000),
            "bge" => (0b101, 0b000_0000),
            "bgeu" => (0b111, 0b000_0000),
            "blt" => (0b100, 0b000_0000),
            "bltu" => (0b110, 0b000_0000),
            "bne" => (0b001, 0b000_0000),

            "jal" => (0b000, 0b000_0000),
            "jalr" => (0b000, 0b000_0000),

            "auipc" => (0b000, 0b000_0000),
            "lui" => (0b000, 0b000_0000),

            _ => panic!("Unsupported instruction"),
        }
    }

    fn get_opcode(&self) -> u32 {
        match self.name {
            "jal" => 0b110_1111,
            "jalr" => 0b110_0111,
            "auipc" => 0b001_0111,
            "lui" => 0b011_0111,
            _ => match self.ty {
                InstType::R => 0b011_0011,
                InstType::I | InstType::IStar => 0b001_0011,
                InstType::L => 0b000_0011,
                InstType::S => 0b010_0011,
                InstType::B => 0b110_0011,
                _ => panic!("Unsupported instruction"),
            },
        }
    }
    fn get_shifted_imm(&self) -> u32 {
        match self.ty {
            InstType::I | InstType::L => (self.imm & ((1u32 << 12) - 1)) << 20,
            InstType::IStar => (self.imm & ((1u32 << 5) - 1)) << 20,
            InstType::S => {
                let imm = self.imm & ((1u32 << 12) - 1);
                ((imm & 0b1_1111) << 7) | ((imm >> 5) << 25)
            }
            InstType::B => {
                let imm = self.imm & ((1u32 << 13) - 1);
                ((imm & 0b1_0000_0000_0000) << 19)
                    | ((imm & 0b0_0111_1110_0000) << 20)
                    | ((imm & 0b0_0000_0001_1110) << 6)
                    | ((imm & 0b0_1000_0000_0000) >> 4)
            }
            InstType::U => self.imm & 0xFFFF_F000,
            InstType::J => {
                let imm = self.imm & ((1u32 << 21) - 1);
                ((imm & 0b1_0000_0000_0000_0000_0000) << 11)
                    | ((imm & 0b0_0000_0000_0111_1111_1110) << 19)
                    | ((imm & 0b0_0000_0000_1000_0000_0000) << 11)
                    | (imm & 0b0_1111_1111_0000_0000_0000)
            }
            _ => 0,
        }
    }
    pub fn addi(rd: &'static str, rs1: &'static str, imm: i64) -> Vec<Self> {
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.push(Self {
                name: "addi",
                rs1: PhyReg(rs1),
                rs2: PhyReg("x0"),
                rd: PhyReg(rd),
                ty: InstType::I,
                imm: imm as u32,
                label: None,
            });
        } else {
            todo!()
        }
        res
    }

    pub fn ld(rd: &'static str, rs1: &'static str, imm: i64) -> Vec<Self> {
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.push(Self {
                name: "ld",
                rs1: PhyReg(rs1),
                rs2: PhyReg("x0"),
                rd: PhyReg(rd),
                ty: InstType::L,
                imm: imm as u32,
                label: None,
            });
        } else {
            todo!()
        }
        res
    }

    pub fn sd(rs1: &'static str, rs2: &'static str, imm: i64) -> Vec<Self> {
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.push(Self {
                name: "sd",
                rs1: PhyReg(rs1),
                rs2: PhyReg(rs2),
                rd: PhyReg("x0"),
                ty: InstType::S,
                imm: imm as u32,
                label: None,
            });
        } else {
            todo!()
        }
        res
    }

    pub fn binary(
        inst_name: &'static str,
        dest: &Name,
        op0: &Operand,
        op1: &Operand,
        ctx: &ASMContext,
    ) -> Vec<Self> {
        let mut res = Vec::new();
        res.extend(Self::get_operand("t0", op0, ctx, false));
        res.extend(Self::get_operand("t1", op1, ctx, false));
        res.push(Self {
            name: inst_name,
            rs1: PhyReg("t0"),
            rs2: PhyReg("t1"),
            rd: PhyReg("t2"),
            ty: InstType::R,
            imm: 0,
            label: None,
        });
        res.extend(Self::sd(
            "t2",
            "sp",
            *ctx.local_vars.get(dest).unwrap() as i64,
        ));
        res
    }

    /// load a local var to rd, or load a global ptr to rd. \
    /// local ptrs should never be loaded to phy reg \
    /// if not force, we can consider if op is a para and if we can just pass
    pub fn get_operand(rd: &'static str, op: &Operand, ctx: &ASMContext, force: bool) -> Vec<Self> {
        match op {
            Operand::ConstantOperand(const_ref) => {
                match &**const_ref {
                    Constant::Int { bits, value } => {
                        let imm = sign_extend(*bits, *value);
                        Self::addi(rd, "x0", imm)
                    }
                    Constant::GlobalReference { name, .. } => {
                        // get the value of global ptr
                        let global_inner = GLOBAL_PTR.exclusive_access();
                        let imm = global_inner.get(name).unwrap();
                        Self::addi(rd, "x0", *imm as i64)
                    }
                    _ => panic!("Unsupported constant operand"),
                }
            }
            Operand::LocalOperand { name, .. } => {
                if let Some(reg) = ctx.paras.get(name) {
                    if !force && reg.0 == rd {
                        return vec![];
                    }
                }

                if let Some(offset) = ctx.local_vars.get(name) {
                    Self::ld(rd, "sp", *offset as i64)
                } else if let Some(ptr) = ctx.local_ptr_map.get(name) {
                    let offset = ctx.local_ptrs.get(ptr).unwrap();
                    Self::ld(rd, "sp", *offset as i64)
                } else {
                    panic!("Local var not found! var: {:?}", name);
                }
            }
            _ => panic!("Unsupported operand for get_local_var"),
        }
    }
}

/// Deal with the local val management
struct ASMContext {
    local_vars: HashMap<Name, u64>, // virt reg = Mem[sp + offset]
    local_ptrs: HashMap<Name, u64>, // virt reg will not be valued
    local_ptr_map: HashMap<Name, Name>,
    global_ptr_map: HashMap<Name, Name>,
    paras: HashMap<Name, PhyReg>,
    stack_size: u64,
}

impl ASMContext {
    pub fn from(func: Arc<Function>) -> Self {
        let mut ctx = Self {
            local_vars: HashMap::new(),
            local_ptrs: HashMap::new(),
            local_ptr_map: HashMap::new(),
            global_ptr_map: HashMap::new(),
            paras: HashMap::new(),
            stack_size: 0,
        };
        let para_num = func.parameters.len() as u64;

        // calculate stack size needed
        ctx.stack_size += min(8, para_num) * 8; // for para protection
        let mut max_call_para_num = 0;
        for bb in func.basic_blocks.iter() {
            for inst in bb.instrs.iter() {
                match inst {
                    Instruction::Add(_)
                    | Instruction::Sub(_)
                    | Instruction::And(_)
                    | Instruction::Or(_)
                    | Instruction::Xor(_)
                    | Instruction::Shl(_)
                    | Instruction::LShr(_)
                    | Instruction::AShr(_)
                    | Instruction::GetElementPtr(_)
                    | Instruction::ICmp(_) => {
                        ctx.stack_size += 8;
                    }
                    // only 1 dimension array is supported
                    Instruction::Alloca(alloca) => match *alloca.allocated_type {
                        Type::IntegerType { .. } => {
                            ctx.stack_size += 8;
                        }
                        Type::ArrayType { num_elements, .. } => {
                            ctx.stack_size += num_elements as u64 * 8;
                        }
                        _ => {
                            panic!("Unsupported alloca type");
                        }
                    },

                    Instruction::Load(_) => {}
                    Instruction::Store(_) => {}

                    Instruction::Call(call) => {
                        if let Some(_) = call.dest {
                            ctx.stack_size += 8;
                        }
                        max_call_para_num = max(max_call_para_num, call.arguments.len() as u64);
                    }

                    _ => panic!("Unsupported instruction: {:?}", inst),
                }
            }
        }

        ctx.stack_size += max(max_call_para_num, 8) - 8; // for spill
        ctx.stack_size = (ctx.stack_size + 15) / 16 * 16; // align to 16

        // now get all offset
        // get para offset
        let mut cur_offset = ctx.stack_size - 8;
        for (i, para) in func.parameters.iter().enumerate() {
            if i < 8 {
                ctx.local_vars.insert(para.name.clone(), cur_offset);
                cur_offset -= 8;
                ctx.paras
                    .insert(para.name.clone(), PhyReg(PHY_REGS[i + 10]));
            } else {
                ctx.local_vars
                    .insert(para.name.clone(), ctx.stack_size + i as u64 * 8);
            }
        }

        // get local var offset
        for bb in func.basic_blocks.iter() {
            for inst in bb.instrs.iter() {
                match inst {
                    Instruction::Add(add) => {
                        ctx.local_vars.insert(add.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::Sub(sub) => {
                        ctx.local_vars.insert(sub.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::And(and) => {
                        ctx.local_vars.insert(and.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::Or(or) => {
                        ctx.local_vars.insert(or.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::Xor(xor) => {
                        ctx.local_vars.insert(xor.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::Shl(shl) => {
                        ctx.local_vars.insert(shl.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::LShr(lshr) => {
                        ctx.local_vars.insert(lshr.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }
                    Instruction::AShr(ashr) => {
                        ctx.local_vars.insert(ashr.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }

                    Instruction::Alloca(alloca) => match *alloca.allocated_type {
                        Type::IntegerType { .. } => {
                            ctx.local_ptrs.insert(alloca.dest.clone(), cur_offset);
                            cur_offset -= 8;
                        }
                        Type::ArrayType { num_elements, .. } => {
                            ctx.local_ptrs.insert(
                                alloca.dest.clone(),
                                cur_offset + 8 - 8 * num_elements as u64,
                            );
                            cur_offset -= 8 * num_elements as u64;
                        }
                        _ => {
                            panic!("Unsupported alloca type");
                        }
                    },
                    Instruction::Load(load) => {
                        // create binding between virt reg its ptr
                        match &load.address {
                            Operand::LocalOperand { name, .. } => {
                                ctx.local_ptr_map.insert(load.dest.clone(), name.clone());
                            }
                            Operand::ConstantOperand(const_ref) => match &**const_ref {
                                Constant::GlobalReference { name, .. } => {
                                    ctx.global_ptr_map.insert(load.dest.clone(), name.clone());
                                }
                                _ => panic!("Unsupported load address"),
                            },
                            _ => panic!("Unsupported load address"),
                        }
                    }
                    Instruction::Store(_) => {}
                    Instruction::GetElementPtr(get_elem_ptr) => {
                        ctx.local_vars.insert(get_elem_ptr.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }

                    Instruction::ICmp(icmp) => {
                        ctx.local_vars.insert(icmp.dest.clone(), cur_offset);
                        cur_offset -= 8;
                    }

                    Instruction::Call(call) => {
                        if let Some(dest) = &call.dest {
                            ctx.local_vars.insert(dest.clone(), cur_offset);
                            cur_offset -= 8;
                        }
                    }
                    _ => panic!("Unsupported instruction: {:?}", inst),
                }
            }
        }

        ctx
    }
}

struct ASMBuilder {
    func: Arc<Function>,
    ctx: ASMContext,
    asm: RefCell<Vec<ASMInst>>,
}
impl ASMBuilder {
    pub fn from(func: Arc<Function>) -> Self {
        Self {
            func: func.clone(),
            ctx: ASMContext::from(func),
            asm: RefCell::new(Vec::new()),
        }
    }
    pub fn addi(&self, rd: &'static str, rs: &'static str, imm: i64) {
        self.asm.borrow_mut().extend(ASMInst::addi(rd, rs, imm));
    }

    /// after call, retrieve the para from stack again
    pub fn restore_para(&self) {}

    fn build_inst(&self, ir_inst: &Instruction) {
        self.asm
            .borrow_mut()
            .extend(ASMInst::from_inst(ir_inst, &self.ctx));
    }
    fn build_terminator(&self, term: &Terminator) {
        self.asm
            .borrow_mut()
            .extend(ASMInst::from_terminator(term, &self.ctx));
    }
    pub fn build(&mut self) {
        // prepare stack
        self.addi("sp", "sp", -(self.ctx.stack_size as i64));

        for bb in self.func.basic_blocks.iter() {
            for inst in bb.instrs.iter() {
                self.build_inst(inst);
            }
            self.build_terminator(&bb.term);
        }

        self.addi("sp", "sp", self.ctx.stack_size as i64);
    }

    pub fn to_binary(&self) -> Vec<u32> {
        self.asm
            .borrow()
            .iter()
            .map(|inst| inst.to_binary())
            .collect()
    }
}

/// Build asm for a function， return the start addr of the func
/// functions that can compiled should not have phi
pub fn compile_func(func: Arc<Function>) -> u64 {
    let mut asm_builder = ASMBuilder::from(func.clone());

    asm_builder.build();

    let binary = asm_builder.to_binary();
    let start_ptr = alloc_mem(binary.len() * 4);
    let mut ptr = start_ptr;
    for inst in binary.iter() {
        unsafe {
            *(ptr as *mut u32) = *inst;
        }
        ptr += 4;
    }

    start_ptr
}
