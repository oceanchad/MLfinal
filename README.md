# MLfinal

### EDA
1. missing value
   1. children / country / agent / company (with nan) -> 4 / 468 / 13217 / 85917
   2. Meal (‘Undefined’) -> Undefined / SC are same
2. Invalid record (action: should drop)
   1. zero_guests (adults / children / babies) -> (0 / 0 / 0) but still have record
   2. is_canceled 32760/91531 (try to drop / 有取消的32760)
3. is_canceled (58771 未取消)