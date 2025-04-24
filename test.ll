
define i32 @main() {
entry:
  %1 = add i32 1, 3
  %icmp = icmp eq i32 %1, 3
  br i1 %icmp, label %true, label %false
true:
  ret i32 9999
false:
  ret i32 -9999
}
