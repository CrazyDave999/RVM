use crate::alloc::alloc_mem;
use crate::interpreter::sign_extend;
use crate::mem::{GLOBAL_PTR, get_local_rnk};
use llvm_ir::{Constant, Function, Instruction, IntPredicate, Name, Operand, Terminator, Type};
use std::arch::global_asm;
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;

global_asm!(include_str!("asm_call.S"));
unsafe extern "C" {
    pub fn __asm_call_fn() -> i64;
}

static PHY_REGS: [&'static str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];
#[derive(Clone, Copy)]
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
                _ => panic!("Unsupported register name, {}", self.0),
            }
        }
    }
}
#[derive(Clone, Copy)]
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

#[derive(Copy, Clone)]
struct ASMInst {
    name: &'static str,
    rs1: PhyReg,
    rs2: PhyReg,
    rd: PhyReg,
    ty: InstType,
    imm: u32,
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
                res.extend(Self::get_operand("t0", &store.value, ctx));
                res.extend(Self::get_operand("t1", &store.address, ctx));
                res.extend(Self::s_type("sd", "t0", "t1", 0));
            }
            Instruction::GetElementPtr(get_elem_ptr) => {
                let dest = &get_elem_ptr.dest;
                let addr = &get_elem_ptr.address;
                let idx = &get_elem_ptr.indices[1];
                res.extend(Self::get_operand("t0", &addr, ctx));
                res.extend(Self::get_operand("t1", &idx, ctx));
                res.extend(Self::i_type("slli", "t1", "t1", 3));
                res.extend(Self::r_type("add", "t2", "t0", "t1"));
                res.extend(Self::s_type(
                    "sd",
                    "t2",
                    "sp",
                    *ctx.local_vars.get(dest).unwrap() as i64,
                ));
            }

            Instruction::ICmp(icmp) => {
                let lhs = &icmp.operand0;
                let rhs = &icmp.operand1;
                let dest = &icmp.dest;
                res.extend(Self::get_operand("t0", &lhs, ctx));
                res.extend(Self::get_operand("t1", &rhs, ctx));

                match icmp.predicate {
                    IntPredicate::EQ => {
                        res.extend(Self::r_type("xor", "t2", "t0", "t1"));
                        res.extend(Self::i_type("sltiu", "t3", "t2", 1));
                    }
                    IntPredicate::NE => {
                        res.extend(Self::r_type("xor", "t2", "t0", "t1"));
                        res.extend(Self::r_type("sltu", "t3", "x0", "t2"));
                    }
                    IntPredicate::UGT => {
                        res.extend(Self::r_type("sltu", "t3", "t1", "t0"));
                    }
                    IntPredicate::UGE => {
                        res.extend(Self::r_type("sltu", "t2", "t0", "t1"));
                        res.extend(Self::i_type("xori", "t3", "t2", 1));
                    }
                    IntPredicate::ULT => res.extend(Self::r_type("sltu", "t3", "t0", "t1")),
                    IntPredicate::ULE => {
                        res.extend(Self::r_type("sltu", "t2", "t1", "t0"));
                        res.extend(Self::i_type("xori", "t3", "t2", 1));
                    }
                    IntPredicate::SGT => {
                        res.extend(Self::r_type("slt", "t3", "t1", "t0"));
                    }
                    IntPredicate::SGE => {
                        res.extend(Self::r_type("slt", "t2", "t0", "t1"));
                        res.extend(Self::i_type("xori", "t3", "t2", 1));
                    }
                    IntPredicate::SLT => {
                        res.extend(Self::r_type("slt", "t3", "t0", "t1"));
                    }
                    IntPredicate::SLE => {
                        res.extend(Self::r_type("slt", "t2", "t1", "t0"));
                        res.extend(Self::i_type("xori", "t3", "t2", 1));
                    }
                }
                res.extend(Self::s_type(
                    "sd",
                    "t3",
                    "sp",
                    *ctx.local_vars.get(dest).unwrap() as i64,
                ));
            }

            Instruction::Call(call) => match &call.function.clone().right().unwrap() {
                Operand::ConstantOperand(const_ref) => match &**const_ref {
                    Constant::GlobalReference { name, .. } => {
                        let fn_index = get_local_rnk(&name.to_string()[1..]).unwrap();

                        // prepare args
                        for (i, (arg, _)) in call.arguments.iter().enumerate() {
                            if i < 8 {
                                res.extend(Self::get_operand(PHY_REGS[i + 10], arg, ctx));
                            } else {
                                res.extend(Self::get_operand("t0", arg, ctx));
                                res.extend(Self::s_type("sd", "t0", "sp", (8 * (i - 8)) as i64));
                            }
                        }

                        // t0 should be the func index
                        res.extend(Self::li("t0", fn_index as i64));

                        // call the asm func: __asm_call_fn
                        res.extend(Self::li("t1", __asm_call_fn as i64));
                        res.extend(Self::i_type("jalr", "ra", "t1", 0));

                        // store the return value
                        if let Some(dest) = &call.dest {
                            res.extend(Self::s_type(
                                "sd",
                                "a0",
                                "sp",
                                *ctx.local_vars.get(dest).unwrap() as i64,
                            ))
                        }

                        // restore para, since they are caller-saved
                        for (name, reg) in ctx.paras.iter() {
                            res.extend(Self::i_type(
                                "ld",
                                reg.0,
                                "sp",
                                *ctx.local_vars.get(name).unwrap() as i64,
                            ))
                        }
                    }
                    _ => panic!("Unsupported function call"),
                },
                _ => panic!("Unsupported function call"),
            },
            _ => panic!("Unsupported instruction: {:?}", ir_inst),
        }
        res
    }
    pub fn from_terminator(term: &Terminator, ctx: &ASMContext) -> Vec<Self> {
        let mut res = Vec::new();
        match term {
            Terminator::Ret(ret) => {
                if let Some(op) = &ret.return_operand {
                    res.extend(Self::get_operand("a0", op, ctx));
                    res.extend(Self::i_type(
                        "ld",
                        "ra",
                        "sp",
                        *ctx.local_vars.get(&Name::from("#ra")).unwrap() as i64,
                    ));
                    res.extend(Self::i_type("addi", "sp", "sp", ctx.stack_size as i64));
                    res.extend(Self::i_type("jalr", "x0", "ra", 0));
                }
            }
            Terminator::Br(br) => {
                // need second build
                let dest = &br.dest;
                res.extend(Self::j_type(
                    "jal",
                    "x0",
                    *ctx.label_rnk.get(dest).unwrap() as i64,
                ));
            }
            Terminator::CondBr(cond_br) => {
                // need second build
                let cond = &cond_br.condition;
                let true_dest = &cond_br.true_dest;
                let false_dest = &cond_br.false_dest;
                res.extend(Self::get_operand("t0", cond, ctx));
                res.extend(Self::b_type("beq", "t0", "x0", 8));
                res.extend(Self::j_type(
                    "jal",
                    "x0",
                    *ctx.label_rnk.get(true_dest).unwrap() as i64,
                ));
                res.extend(Self::j_type(
                    "jal",
                    "x0",
                    *ctx.label_rnk.get(false_dest).unwrap() as i64,
                ));
            }
            _ => panic!("Unsupported terminator: {:?}", term),
        }
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
        match self.ty {
            InstType::R => {
                format!(
                    "{} {}, {}, {}\n",
                    self.name, self.rd.0, self.rs1.0, self.rs2.0
                )
            }
            InstType::I => {
                format!(
                    "{} {}, {}, {}\n",
                    self.name, self.rd.0, self.rs1.0, self.imm as i32
                )
            }
            InstType::IStar => {
                format!(
                    "{} {}, {}, {}\n",
                    self.name, self.rd.0, self.rs1.0, self.imm as i32
                )
            }
            InstType::L => {
                format!(
                    "{} {}, {}({})\n",
                    self.name, self.rd.0, self.imm, self.rs1.0
                )
            }
            InstType::S => {
                format!(
                    "{} {}, {}({})\n",
                    self.name, self.rs2.0, self.imm, self.rs1.0
                )
            }
            InstType::B => {
                format!(
                    "{} {}, {}, {}\n",
                    self.name, self.rs1.0, self.rs2.0, self.imm as i32
                )
            }
            InstType::U => {
                format!("{} {}, {}\n", self.name, self.rd.0, self.imm)
            }
            InstType::J => {
                format!("{} {}, {}\n", self.name, self.rd.0, self.imm as i32)
            }
        }
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

            _ => panic!("Unsupported instruction, {}", self.name),
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
                _ => panic!("Unsupported instruction, {}", self.name),
            },
        }
    }
    fn get_shifted_imm(&self) -> u32 {
        match self.ty {
            InstType::I | InstType::L => (self.imm & ((1u32 << 12) - 1)) << 20,
            InstType::IStar => (self.imm & ((1u32 << 6) - 1)) << 20,
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

    /// will use t5, t6
    pub fn addi(rd: &'static str, rs1: &'static str, imm: i64) -> Vec<Self> {
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.extend(Self::i_type("addi", rd, rs1, imm));
        } else {
            res.extend(Self::li("t5", imm));
            res.extend(Self::r_type("add", rd, rs1, "t5"));
        }
        res
    }

    /// implement li pseudo inst manually \
    /// will use t6 \
    /// 8 asm insts in the worst case
    pub fn li(rd: &'static str, imm: i64) -> Vec<Self> {
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.extend(Self::i_type("addi", rd, "x0", imm));
        } else if imm >= -2147483648 && imm <= 2147483647 {
            let mut p1 = imm;
            let p2 = sign_extend(12, (imm & 0xFFF) as u64);
            if p2 < 0 {
                p1 += 0x1000;
            }
            res.extend(Self::u_type("lui", rd, p1));
            res.extend(Self::addi(rd, rd, p2));
        } else {
            let mut p1 = imm >> 32;
            let p2 = sign_extend(12, (imm & 0x0000_0FFF_0000_0000) as u64 >> 32);
            if p2 < 0 {
                p1 += 0x1000; // if p2 is negative and sign-extended, add more 0x1000 to p1 to cancel the extended value
            }
            let p3 = (imm & 0x0000_0000_FFE0_0000) >> 21;
            let p4 = (imm & 0x0000_0000_001F_FC00) >> 10;
            let p5 = imm & 0x0000_0000_0000_03FF;
            res.extend(Self::u_type("lui", rd, p1));
            res.extend(Self::i_type("addi", rd, rd, p2));
            res.extend(Self::i_type("slli", rd, rd, 11));
            res.extend(Self::i_type("addi", rd, rd, p3));
            res.extend(Self::i_type("slli", rd, rd, 11));
            res.extend(Self::i_type("addi", rd, rd, p4));
            res.extend(Self::i_type("slli", rd, rd, 10));
            res.extend(Self::i_type("addi", rd, rd, p5));
        }
        res
    }

    pub fn r_type(
        inst_name: &'static str,
        rd: &'static str,
        rs1: &'static str,
        rs2: &'static str,
    ) -> Vec<Self> {
        let mut res = Vec::new();
        res.push(Self {
            name: inst_name,
            rs1: PhyReg(rs1),
            rs2: PhyReg(rs2),
            rd: PhyReg(rd),
            ty: InstType::R,
            imm: 0,
        });
        res
    }
    pub fn i_type(
        inst_name: &'static str,
        rd: &'static str,
        rs1: &'static str,
        imm: i64,
    ) -> Vec<Self> {
        let i_r_map = HashMap::from([
            ("addi", "add"),
            ("andi", "and"),
            ("ori", "or"),
            ("xori", "xor"),
            ("slli", "sll"),
            ("srli", "srl"),
            ("srai", "sra"),
            ("slti", "slt"),
            ("sltiu", "sltu"),
        ]);
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.push(Self {
                name: inst_name,
                rs1: PhyReg(rs1),
                rs2: PhyReg("x0"),
                rd: PhyReg(rd),
                ty: match inst_name {
                    "slli" | "srli" | "srai" => {
                        assert!(imm >= 0 && imm <= 63);
                        InstType::IStar
                    }
                    "ld" => InstType::L,
                    _ => InstType::I,
                },
                imm: imm as u32,
            });
        } else {
            match inst_name {
                "jalr" | "ld" => {
                    res.extend(Self::addi("t4", rs1, imm));
                    res.extend(Self::i_type(inst_name, rd, "t4", 0));
                }
                _ => {
                    res.extend(Self::li("t5", imm));
                    res.extend(Self::r_type(i_r_map[inst_name], rd, rs1, "t5"));
                }
            }
        }
        res
    }

    pub fn s_type(
        inst_name: &'static str,
        rs2: &'static str,
        rs1: &'static str,
        imm: i64,
    ) -> Vec<Self> {
        let mut res = Vec::new();
        if imm >= -2048 && imm <= 2047 {
            res.push(Self {
                name: inst_name,
                rs1: PhyReg(rs1),
                rs2: PhyReg(rs2),
                rd: PhyReg("x0"),
                ty: InstType::S,
                imm: imm as u32,
            });
        } else {
            res.extend(Self::addi("t4", rs1, imm));
            res.extend(Self::s_type(inst_name, rs2, "t4", 0));
        }
        res
    }

    pub fn b_type(
        inst_name: &'static str,
        rs1: &'static str,
        rs2: &'static str,
        imm: i64,
    ) -> Vec<Self> {
        assert!(imm >= -2048 && imm <= 2047);
        let mut res = Vec::new();
        res.push(Self {
            name: inst_name,
            rs1: PhyReg(rs1),
            rs2: PhyReg(rs2),
            rd: PhyReg("x0"),
            ty: InstType::B,
            imm: imm as u32,
        });
        res
    }
    pub fn u_type(inst_name: &'static str, rd: &'static str, imm: i64) -> Vec<Self> {
        assert!(imm >= -2147483648 && imm <= 2147483647);
        let mut res = Vec::new();
        res.push(Self {
            name: inst_name,
            rs1: PhyReg("x0"),
            rs2: PhyReg("x0"),
            rd: PhyReg(rd),
            ty: InstType::U,
            imm: imm as u32 & 0xFFFF_F000,
        });
        res
    }

    pub fn j_type(inst_name: &'static str, rd: &'static str, imm: i64) -> Vec<Self> {
        let mut res = Vec::new();
        assert!(imm >= -1048576 && imm <= 1048575);
        res.push(Self {
            name: inst_name,
            rs1: PhyReg("x0"),
            rs2: PhyReg("x0"),
            rd: PhyReg(rd),
            ty: InstType::J,
            imm: imm as u32,
        });
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
        res.extend(Self::get_operand("t0", op0, ctx));
        res.extend(Self::get_operand("t1", op1, ctx));
        res.push(Self {
            name: inst_name,
            rs1: PhyReg("t0"),
            rs2: PhyReg("t1"),
            rd: PhyReg("t2"),
            ty: InstType::R,
            imm: 0,
        });
        res.extend(Self::s_type(
            "sd",
            "t2",
            "sp",
            *ctx.local_vars.get(dest).unwrap() as i64,
        ));
        res
    }

    /// load a local var to rd, or load a global/local ptr to rd. \
    /// if not force, we can consider if op is a para and if we can just pass
    pub fn get_operand(rd: &'static str, op: &Operand, ctx: &ASMContext) -> Vec<Self> {
        match op {
            Operand::ConstantOperand(const_ref) => {
                match &**const_ref {
                    Constant::Int { bits, value } => {
                        let imm = sign_extend(*bits, *value);
                        Self::li(rd, imm)
                    }
                    Constant::GlobalReference { name, .. } => {
                        // get the value of global ptr
                        let global_inner = GLOBAL_PTR.exclusive_access();
                        let imm = global_inner.get(name).unwrap();
                        Self::li(rd, *imm as i64)
                    }
                    _ => panic!("Unsupported constant operand"),
                }
            }
            Operand::LocalOperand { name, .. } => {
                if let Some(offset) = ctx.local_vars.get(name) {
                    Self::i_type("ld", rd, "sp", *offset as i64)
                } else if let Some(ptr) = ctx.local_ptr_map.get(name) {
                    // ptr maybe the dest of alloca or getelementptr
                    if let Some(offset) = ctx.local_ptrs.get(ptr) {
                        // ptr is the dest of alloca
                        Self::i_type("ld", rd, "sp", *offset as i64)
                    } else if let Some(offset) = ctx.local_vars.get(ptr) {
                        // ptr is the dest of getelementptr
                        let mut res = Vec::new();
                        res.extend(Self::i_type("ld", rd, "sp", *offset as i64));
                        res.extend(Self::i_type("ld", rd, rd, 0));
                        res
                    } else {
                        panic!("Local ptr not found! ptr: {:?}, name: {:?}", ptr, name);
                    }
                } else if let Some(offset) = ctx.local_ptrs.get(name) {
                    Self::addi(rd, "sp", *offset as i64)
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
    label_rnk: HashMap<Name, u64>,
    label_offset: HashMap<u64, u64>,
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
            label_rnk: HashMap::new(),
            label_offset: HashMap::new(),
            stack_size: 0,
        };
        let para_num = func.parameters.len() as u64;

        // calculate stack size needed
        ctx.stack_size += 8; // for ra
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
        ctx.stack_size += 8;
        ctx.stack_size = (ctx.stack_size + 15) / 16 * 16; // align to 16

        // now get all offset
        // get ra offset
        let mut cur_offset = ctx.stack_size - 8;
        ctx.local_vars.insert(Name::from("#ra"), cur_offset);
        cur_offset -= 8;

        // get para offset
        for (i, para) in func.parameters.iter().enumerate() {
            if i < 8 {
                ctx.local_vars.insert(para.name.clone(), cur_offset);
                cur_offset -= 8;
                ctx.paras
                    .insert(para.name.clone(), PhyReg(PHY_REGS[i + 10]));
            } else {
                ctx.local_vars
                    .insert(para.name.clone(), ctx.stack_size + (i - 8) as u64 * 8);
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

        // get label rnk
        for (i, bb) in func.basic_blocks.iter().enumerate() {
            ctx.label_rnk.insert(bb.name.clone(), i as u64);
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

    /// the first run, transfer most of the ir to asm, leave placeholder for br and cond br
    pub fn build(&mut self) {
        let mut inner = self.asm.borrow_mut();
        // prepare stack
        inner.extend(ASMInst::i_type(
            "addi",
            "sp",
            "sp",
            -(self.ctx.stack_size as i64),
        ));
        inner.extend(ASMInst::s_type(
            "sd",
            "ra",
            "sp",
            *self.ctx.local_vars.get(&Name::from("#ra")).unwrap() as i64,
        ));

        // store paras
        for (name, reg) in self.ctx.paras.iter() {
            inner.extend(ASMInst::s_type(
                "sd",
                reg.0,
                "sp",
                *self.ctx.local_vars.get(name).unwrap() as i64,
            ));
        }

        for (i, bb) in self.func.basic_blocks.iter().enumerate() {
            // get the actual offset of the label
            self.ctx
                .label_offset
                .insert(i as u64, (inner.len() * 4) as u64);
            for inst in bb.instrs.iter() {
                inner.extend(ASMInst::from_inst(inst, &self.ctx));
            }
            inner.extend(ASMInst::from_terminator(&bb.term, &self.ctx));
        }

        // correct the placeholders
        for (i, inst) in inner.iter_mut().enumerate() {
            match inst.name {
                "jal" => {
                    let label_offset = self.ctx.label_offset[&(inst.imm as u64)];
                    let offset = label_offset - i as u64 * 4;
                    inst.imm = offset as u32;
                }
                _ => {}
            }
        }
    }
    pub fn print(&self) {
        println!("assembly of func: {}", self.func.name);
        let inner = self.asm.borrow();
        for inst in inner.iter() {
            print!("{}", inst.to_string());
        }
        println!()
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
    asm_builder.print();
    // panic!();

    let binary = asm_builder.to_binary();
    let start_ptr = alloc_mem(binary.len() * 4); // will not dealloc this, until the program ends

    println!("Function {} will be placed at: {:#x}", func.name, start_ptr);
    unsafe {
        ptr::copy_nonoverlapping(binary.as_ptr(), start_ptr as *mut u32, binary.len());
    }

    // let mut ptr = start_ptr;
    // for inst in binary.iter() {
    //     unsafe {
    //         *(ptr as *mut u32) = *inst;
    //     }
    //     ptr += 4;
    // }

    start_ptr
}
