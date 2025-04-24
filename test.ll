
define i32 @main() {
entry:
  %1 = alloca i32
  %2 = add i32 1, 3
  %icmp = icmp eq i32 %2, 3
  br i1 %icmp, label %true, label %false
true:
  store i32 1, i32* %1
  br label %exit
false:
  store i32 0, i32* %1
  br label %exit
exit:
  %3 = load i32, ptr %1
  ret i32 %3
}
