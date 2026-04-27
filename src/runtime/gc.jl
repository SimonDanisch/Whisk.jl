## Garbage collector — precise mark-sweep with conservative scanning of
## payload bodies. wasm64 target.
##
## Linear-memory layout after wasm-ld:
##
##     0  ┌────────────────────────────────┐
##        │ rodata, data segments          │
##        ├────────────────────────────────┤
##        │ wasm stack (grows down)        │
##  __heap_base ───────────────────────────┤
##        │ allocated heap (header+payload)│
##        │   ↓ bump allocator             │
##  _whisk_heap_next ──────────────────────┤
##        │ free space                     │
##     ↓  └────────────────────────────────┘
##
## Object layout (16-byte header + payload):
##
##     ┌──────────────────────────────────┐
##     │ size : u64  (total bytes incl hdr)│ — multiple of 8
##     │ flags: u64  (bit 0 = mark,        │
##     │              bit 1 = free)        │
##     ├──────────────────────────────────┤
##     │ payload …                         │
##     └──────────────────────────────────┘
##
## We target wasm64 (64-bit pointers and linear-memory addressing).
## This matches Julia's host-time `Int = Int64` and `sizeof(Ptr) == 8`,
## eliminating the cross-compile struct-layout inconsistencies wasm32
## suffers from. V8 and Firefox both support wasm64.
##
## Roots come from Julia's GC frame intrinsics (`julia.{new,push,pop}_
## gc_frame`, `julia.get_gc_frame_slot`, `julia.get_pgcstack`) — we
## lower them into real linked-list operations on a shadow root stack
## kept at the wasm global `_whisk_pgcstack`. Payload bodies are
## scanned conservatively (every aligned 8-byte word treated as a
## potential heap pointer).

# ── Constants ─────────────────────────────────────────────────────────

const HEADER_SIZE      = UInt64(16)
const HEADER_FLAGS_OFF = UInt64(8)
const FLAG_MARK        = UInt64(0x1)
const FLAG_FREE        = UInt64(0x2)
const MIN_PAYLOAD      = UInt64(8)   # at least 8 bytes — fits free-list link

"""
    emit_wasm_runtime!(mod::LLVM.Module)

Inject the entire GC runtime into `mod`:

  * `@__heap_base`           — extern, supplied by wasm-ld.
  * `@_whisk_heap_next`      — i64 mutable global, lazy-init to
                               `&__heap_base`.
  * `@_whisk_pgcstack`       — i64 mutable global, top-of-stack ptr
                               for the GC root linked list.
  * `@_whisk_free_list_head` — i64 mutable global, head of free list.
  * `whisk_gc_alloc(sz)`              — header-aware allocator.
  * `jl_alloc_genericmemory_unchecked` — Julia 1.11+ Memory{T} ctor.
  * `whisk_gc_collect()`              — mark + sweep entry point.
  * `_whisk_gc_mark()`, `_whisk_gc_sweep()`,
    `_whisk_mark_obj_at_addr(addr)`   — internal helpers.
"""
function emit_wasm_runtime!(mod::LLVM.Module)
    T_i8 = LLVM.Int8Type()

    heap_base = if haskey(LLVM.globals(mod), "__heap_base")
        LLVM.globals(mod)["__heap_base"]
    else
        gv = LLVM.GlobalVariable(mod, T_i8, "__heap_base")
        LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
        LLVM.extinit!(gv, true)
        gv
    end

    heap_next = _get_or_make_i64_global!(mod, "_whisk_heap_next", 0)
    pgcstack  = _get_or_make_i64_global!(mod, "_whisk_pgcstack",  0)
    free_list = _get_or_make_i64_global!(mod, "_whisk_free_list_head", 0)

    _emit_gc_alloc!(mod, heap_base, heap_next, free_list)
    _emit_gc_alloc_memory!(mod)
    _emit_mark_obj_at_addr!(mod, heap_base, heap_next)
    _emit_gc_mark!(mod, pgcstack)
    _emit_gc_sweep!(mod, heap_base, heap_next, free_list)
    _emit_gc_collect!(mod)
end

function _get_or_make_i64_global!(mod, name, init::Integer)
    if haskey(LLVM.globals(mod), name)
        return LLVM.globals(mod)[name]
    end
    T = LLVM.Int64Type()
    gv = LLVM.GlobalVariable(mod, T, name)
    LLVM.initializer!(gv, LLVM.ConstantInt(T, init))
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    return gv
end

# ─────────────────────────────────────────────────────────────────────
# whisk_gc_alloc — free-list-first allocator with header.
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_alloc!(mod, heap_base, heap_next, free_list)
    T_w   = LLVM.Int64Type()
    T_wp  = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "whisk_gc_alloc")
        LLVM.functions(mod)["whisk_gc_alloc"]
    else
        LLVM.Function(mod, "whisk_gc_alloc",
                      LLVM.FunctionType(T_w, [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    sz_param = LLVM.parameters(fn)[1]
    entry  = LLVM.BasicBlock(fn, "entry")
    f_loop = LLVM.BasicBlock(fn, "freelist_loop")
    f_check= LLVM.BasicBlock(fn, "freelist_check")
    f_take = LLVM.BasicBlock(fn, "freelist_take")
    f_next = LLVM.BasicBlock(fn, "freelist_next")
    bump   = LLVM.BasicBlock(fn, "bump")
    done   = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        sz_min = _i64_max!(b, sz_param, LLVM.ConstantInt(T_w, MIN_PAYLOAD))
        sz_hdr = LLVM.add!(b, sz_min, LLVM.ConstantInt(T_w, HEADER_SIZE + 7))
        total  = LLVM.and!(b, sz_hdr,
                            LLVM.ConstantInt(T_w, reinterpret(Int64, ~UInt64(7))))
        prev_link0 = LLVM.ptrtoint!(b, free_list, T_w)
        cur0       = LLVM.load!(b, T_w, free_list)
        LLVM.br!(b, f_loop)

        LLVM.position!(b, f_loop)
        cur       = LLVM.phi!(b, T_w)
        prev_link = LLVM.phi!(b, T_w)
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur,
                              LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, is_zero, bump, f_check)

        LLVM.position!(b, f_check)
        cur_size = LLVM.load!(b, T_w, LLVM.inttoptr!(b, cur, T_wp))
        fits = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, cur_size, total)
        LLVM.br!(b, fits, f_take, f_next)

        LLVM.position!(b, f_take)
        cur_payload_addr = LLVM.add!(b, cur, LLVM.ConstantInt(T_w, HEADER_SIZE))
        cur_payload_ptr  = LLVM.inttoptr!(b, cur_payload_addr, T_wp)
        next_free        = LLVM.load!(b, T_w, cur_payload_ptr)
        prev_link_ptr    = LLVM.inttoptr!(b, prev_link, T_wp)
        LLVM.store!(b, next_free, prev_link_ptr)
        flags_addr_take  = LLVM.add!(b, cur, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        flags_ptr_take   = LLVM.inttoptr!(b, flags_addr_take, T_wp)
        LLVM.store!(b, LLVM.ConstantInt(T_w, 0), flags_ptr_take)
        take_value       = cur_payload_addr
        LLVM.br!(b, done)

        LLVM.position!(b, f_next)
        cur_pl_addr2  = LLVM.add!(b, cur, LLVM.ConstantInt(T_w, HEADER_SIZE))
        new_prev_link = cur_pl_addr2
        new_cur       = LLVM.load!(b, T_w,
                                    LLVM.inttoptr!(b, cur_pl_addr2, T_wp))
        LLVM.br!(b, f_loop)

        LLVM.position!(b, bump)
        cur_hn    = LLVM.load!(b, T_w, heap_next)
        is_unset  = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_hn,
                                LLVM.ConstantInt(T_w, 0))
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        actual_hn = LLVM.select!(b, is_unset, base_addr, cur_hn)
        new_hn    = LLVM.add!(b, actual_hn, total)
        LLVM.store!(b, new_hn, heap_next)
        LLVM.store!(b, total,
                       LLVM.inttoptr!(b, actual_hn, T_wp))
        flags_addr_bump = LLVM.add!(b, actual_hn, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        LLVM.store!(b, LLVM.ConstantInt(T_w, 0),
                       LLVM.inttoptr!(b, flags_addr_bump, T_wp))
        bump_payload    = LLVM.add!(b, actual_hn, LLVM.ConstantInt(T_w, HEADER_SIZE))
        LLVM.br!(b, done)

        LLVM.position!(b, done)
        result = LLVM.phi!(b, T_w)
        LLVM.append!(LLVM.incoming(result),
                     [(take_value, f_take), (bump_payload, bump)])
        LLVM.ret!(b, result)

        LLVM.append!(LLVM.incoming(cur),
                     [(cur0, entry), (new_cur, f_next)])
        LLVM.append!(LLVM.incoming(prev_link),
                     [(prev_link0, entry), (new_prev_link, f_next)])
    end
end

@inline function _i64_max!(b::LLVM.IRBuilder, a, c)
    cond = LLVM.icmp!(b, LLVM.API.LLVMIntUGT, a, c)
    return LLVM.select!(b, cond, a, c)
end

# ─────────────────────────────────────────────────────────────────────
# jl_alloc_genericmemory_unchecked — Julia's Memory{T} ctor.
# Memory layout on wasm64 is `{ i64 length, ptr data }` = 16 bytes,
# matching Julia's host-time assumption.
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_alloc_memory!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "jl_alloc_genericmemory_unchecked")
        LLVM.functions(mod)["jl_alloc_genericmemory_unchecked"]
    else
        ft = LLVM.FunctionType(T_w, [T_w, T_w, T_w])
        LLVM.Function(mod, "jl_alloc_genericmemory_unchecked", ft)
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    T_ret = LLVM.return_type(LLVM.function_type(fn))
    len_bytes = LLVM.parameters(fn)[2]
    entry = LLVM.BasicBlock(fn, "entry")

    whisk_alloc    = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # Memory{T} on wasm64: `{ i64 length, ptr data }` = 16 bytes.
        ms_addr   = LLVM.call!(b, whisk_alloc_ft, whisk_alloc,
                                [LLVM.ConstantInt(T_w, 16)])
        data_addr = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [len_bytes])

        ms_byte_ptr = LLVM.inttoptr!(b, ms_addr, T_i8p)
        slot_ptr    = LLVM.gep!(b, LLVM.Int8Type(), ms_byte_ptr,
                                 [LLVM.ConstantInt(T_w, 8)])
        LLVM.store!(b, data_addr, slot_ptr)
        # Length field at offset 0 — Julia code overwrites it.

        retval = if T_ret isa LLVM.IntegerType
            ms_addr
        else
            cast_ptr0 = LLVM.inttoptr!(b, ms_addr, T_i8p)
            LLVM.addrspacecast!(b, cast_ptr0, T_ret)
        end
        LLVM.ret!(b, retval)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Mark phase: walk pgcstack frames, mark reachable, recurse.
# ─────────────────────────────────────────────────────────────────────

function _emit_mark_obj_at_addr!(mod, heap_base, heap_next)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_mark_obj_at_addr")
        LLVM.functions(mod)["_whisk_mark_obj_at_addr"]
    else
        LLVM.Function(mod, "_whisk_mark_obj_at_addr",
                      LLVM.FunctionType(LLVM.VoidType(), [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    addr = LLVM.parameters(fn)[1]
    entry      = LLVM.BasicBlock(fn, "entry")
    in_heap    = LLVM.BasicBlock(fn, "in_heap")
    find_loop  = LLVM.BasicBlock(fn, "find_loop")
    find_check = LLVM.BasicBlock(fn, "find_check")
    find_next  = LLVM.BasicBlock(fn, "find_next")
    do_mark    = LLVM.BasicBlock(fn, "do_mark")
    already_b  = LLVM.BasicBlock(fn, "already_marked")
    set_mark   = LLVM.BasicBlock(fn, "set_mark")
    scan_loop  = LLVM.BasicBlock(fn, "scan_loop")
    scan_body  = LLVM.BasicBlock(fn, "scan_body")
    done       = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        heap_end  = LLVM.load!(b, T_w, heap_next)
        ge_base   = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, addr, base_addr)
        lt_end    = LLVM.icmp!(b, LLVM.API.LLVMIntULT, addr, heap_end)
        ok        = LLVM.and!(b, ge_base, lt_end)
        LLVM.br!(b, ok, in_heap, done)

        LLVM.position!(b, in_heap)
        LLVM.br!(b, find_loop)

        LLVM.position!(b, find_loop)
        obj = LLVM.phi!(b, T_w)
        past_end = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, obj, heap_end)
        LLVM.br!(b, past_end, done, find_check)

        LLVM.position!(b, find_check)
        obj_size = LLVM.load!(b, T_w, LLVM.inttoptr!(b, obj, T_wp))
        obj_end  = LLVM.add!(b, obj, obj_size)
        cont_lo  = LLVM.icmp!(b, LLVM.API.LLVMIntULE, obj, addr)
        cont_hi  = LLVM.icmp!(b, LLVM.API.LLVMIntULT, addr, obj_end)
        contains = LLVM.and!(b, cont_lo, cont_hi)
        LLVM.br!(b, contains, do_mark, find_next)

        LLVM.position!(b, find_next)
        next_obj = LLVM.add!(b, obj, obj_size)
        LLVM.br!(b, find_loop)

        LLVM.position!(b, do_mark)
        flags_addr  = LLVM.add!(b, obj, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        flags_ptr   = LLVM.inttoptr!(b, flags_addr, T_wp)
        flags       = LLVM.load!(b, T_w, flags_ptr)
        marked_bit  = LLVM.and!(b, flags, LLVM.ConstantInt(T_w, FLAG_MARK))
        is_marked   = LLVM.icmp!(b, LLVM.API.LLVMIntNE, marked_bit,
                                   LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, is_marked, already_b, set_mark)

        LLVM.position!(b, already_b)
        LLVM.br!(b, done)

        LLVM.position!(b, set_mark)
        new_flags = LLVM.or!(b, flags, LLVM.ConstantInt(T_w, FLAG_MARK))
        LLVM.store!(b, new_flags, flags_ptr)
        LLVM.br!(b, scan_loop)

        LLVM.position!(b, scan_loop)
        scan_i = LLVM.phi!(b, T_w)
        in_obj = LLVM.icmp!(b, LLVM.API.LLVMIntULT, scan_i, obj_size)
        LLVM.br!(b, in_obj, scan_body, done)

        LLVM.position!(b, scan_body)
        word_addr = LLVM.add!(b, obj, scan_i)
        word_ptr  = LLVM.inttoptr!(b, word_addr, T_wp)
        word_val  = LLVM.load!(b, T_w, word_ptr)
        LLVM.call!(b, LLVM.function_type(fn), fn, [word_val])
        scan_next = LLVM.add!(b, scan_i, LLVM.ConstantInt(T_w, 8))
        LLVM.br!(b, scan_loop)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(obj),
                     [(base_addr, in_heap), (next_obj, find_next)])
        LLVM.append!(LLVM.incoming(scan_i),
                     [(LLVM.ConstantInt(T_w, HEADER_SIZE), set_mark),
                      (scan_next, scan_body)])
    end
end

function _emit_gc_mark!(mod, pgcstack)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_gc_mark")
        LLVM.functions(mod)["_whisk_gc_mark"]
    else
        LLVM.Function(mod, "_whisk_gc_mark",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    mark_obj    = LLVM.functions(mod)["_whisk_mark_obj_at_addr"]
    mark_obj_ft = LLVM.function_type(mark_obj)

    entry       = LLVM.BasicBlock(fn, "entry")
    frame_loop  = LLVM.BasicBlock(fn, "frame_loop")
    frame_check = LLVM.BasicBlock(fn, "frame_check")
    slot_loop   = LLVM.BasicBlock(fn, "slot_loop")
    slot_body   = LLVM.BasicBlock(fn, "slot_body")
    next_frame  = LLVM.BasicBlock(fn, "next_frame")
    done        = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        cur_frame0 = LLVM.load!(b, T_w, pgcstack)
        LLVM.br!(b, frame_loop)

        LLVM.position!(b, frame_loop)
        cur_frame = LLVM.phi!(b, T_w)
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_frame,
                              LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, is_zero, done, frame_check)

        LLVM.position!(b, frame_check)
        n_addr  = LLVM.add!(b, cur_frame, LLVM.ConstantInt(T_w, 8))
        n_slots = LLVM.load!(b, T_w, LLVM.inttoptr!(b, n_addr, T_wp))
        LLVM.br!(b, slot_loop)

        LLVM.position!(b, slot_loop)
        i = LLVM.phi!(b, T_w)
        in_range = LLVM.icmp!(b, LLVM.API.LLVMIntULT, i, n_slots)
        LLVM.br!(b, in_range, slot_body, next_frame)

        LLVM.position!(b, slot_body)
        i_x8        = LLVM.shl!(b, i, LLVM.ConstantInt(T_w, 3))
        slot_offset = LLVM.add!(b, i_x8, LLVM.ConstantInt(T_w, 16))
        slot_addr   = LLVM.add!(b, cur_frame, slot_offset)
        slot_value  = LLVM.load!(b, T_w, LLVM.inttoptr!(b, slot_addr, T_wp))
        LLVM.call!(b, mark_obj_ft, mark_obj, [slot_value])
        i_next = LLVM.add!(b, i, LLVM.ConstantInt(T_w, 1))
        LLVM.br!(b, slot_loop)

        LLVM.position!(b, next_frame)
        prev_frame = LLVM.load!(b, T_w, LLVM.inttoptr!(b, cur_frame, T_wp))
        LLVM.br!(b, frame_loop)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(cur_frame),
                     [(cur_frame0, entry), (prev_frame, next_frame)])
        LLVM.append!(LLVM.incoming(i),
                     [(LLVM.ConstantInt(T_w, 0), frame_check),
                      (i_next, slot_body)])
    end
end

# ─────────────────────────────────────────────────────────────────────
# Sweep phase
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_sweep!(mod, heap_base, heap_next, free_list)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_gc_sweep")
        LLVM.functions(mod)["_whisk_gc_sweep"]
    else
        LLVM.Function(mod, "_whisk_gc_sweep",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    entry      = LLVM.BasicBlock(fn, "entry")
    obj_loop   = LLVM.BasicBlock(fn, "obj_loop")
    obj_check  = LLVM.BasicBlock(fn, "obj_check")
    is_marked  = LLVM.BasicBlock(fn, "is_marked")
    is_free    = LLVM.BasicBlock(fn, "is_free")
    after      = LLVM.BasicBlock(fn, "after")
    done       = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        LLVM.store!(b, LLVM.ConstantInt(T_w, 0), free_list)
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        heap_end  = LLVM.load!(b, T_w, heap_next)
        LLVM.br!(b, obj_loop)

        LLVM.position!(b, obj_loop)
        obj = LLVM.phi!(b, T_w)
        past = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, obj, heap_end)
        LLVM.br!(b, past, done, obj_check)

        LLVM.position!(b, obj_check)
        size_v    = LLVM.load!(b, T_w, LLVM.inttoptr!(b, obj, T_wp))
        flags_addr= LLVM.add!(b, obj, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        flags_ptr = LLVM.inttoptr!(b, flags_addr, T_wp)
        flags     = LLVM.load!(b, T_w, flags_ptr)
        marked    = LLVM.and!(b, flags, LLVM.ConstantInt(T_w, FLAG_MARK))
        was_marked = LLVM.icmp!(b, LLVM.API.LLVMIntNE, marked,
                                   LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, was_marked, is_marked, is_free)

        LLVM.position!(b, is_marked)
        cleared = LLVM.and!(b, flags,
                            LLVM.ConstantInt(T_w, reinterpret(Int64, ~FLAG_MARK)))
        LLVM.store!(b, cleared, flags_ptr)
        LLVM.br!(b, after)

        LLVM.position!(b, is_free)
        old_head     = LLVM.load!(b, T_w, free_list)
        payload_addr = LLVM.add!(b, obj, LLVM.ConstantInt(T_w, HEADER_SIZE))
        payload_ptr  = LLVM.inttoptr!(b, payload_addr, T_wp)
        LLVM.store!(b, old_head, payload_ptr)
        LLVM.store!(b, obj, free_list)
        free_flags   = LLVM.or!(b, flags, LLVM.ConstantInt(T_w, FLAG_FREE))
        LLVM.store!(b, free_flags, flags_ptr)
        LLVM.br!(b, after)

        LLVM.position!(b, after)
        next_obj = LLVM.add!(b, obj, size_v)
        LLVM.br!(b, obj_loop)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(obj),
                     [(base_addr, entry), (next_obj, after)])
    end
end

function _emit_gc_collect!(mod::LLVM.Module)
    fn = if haskey(LLVM.functions(mod), "whisk_gc_collect")
        LLVM.functions(mod)["whisk_gc_collect"]
    else
        LLVM.Function(mod, "whisk_gc_collect",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    # External linkage + wasm-export-name attribute exports the symbol
    # at link time. Append to `@llvm.used` so the optimizer doesn't DCE
    # it before the linker sees it.
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_gc_collect"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    mark_fn  = LLVM.functions(mod)["_whisk_gc_mark"]
    sweep_fn = LLVM.functions(mod)["_whisk_gc_sweep"]
    entry    = LLVM.BasicBlock(fn, "entry")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        LLVM.call!(b, LLVM.function_type(mark_fn),  mark_fn,  LLVM.Value[])
        LLVM.call!(b, LLVM.function_type(sweep_fn), sweep_fn, LLVM.Value[])
        LLVM.ret!(b)
    end

    mod_obj = LLVM.parent(fn)
    LLVM.API.LLVMAppendToUsed(mod_obj, [fn], 1)
end

# ─────────────────────────────────────────────────────────────────────
# Lowering passes invoked from finish_module!
# ─────────────────────────────────────────────────────────────────────

"""
    lower_gc_alloc_wasm!(mod::LLVM.Module)

Replace every `julia.gc_alloc_obj(_, sz_i64, _)` with
`whisk_gc_alloc(sz_i64)` plus an addrspacecast back to the original
return type.
"""
function lower_gc_alloc_wasm!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "julia.gc_alloc_obj") || return
    alloc_obj = LLVM.functions(mod)["julia.gc_alloc_obj"]
    T_ret = LLVM.return_type(LLVM.function_type(alloc_obj))

    whisk_alloc    = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)
    T_ptr0 = LLVM.PointerType(LLVM.Int8Type())

    for use in collect(LLVM.uses(alloc_obj))
        call = LLVM.user(use)::LLVM.CallInst
        ops  = LLVM.arguments(call)
        sz   = ops[2]
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call)
            i64_addr = LLVM.call!(builder, whisk_alloc_ft, whisk_alloc, [sz])
            as_ptr0  = LLVM.inttoptr!(builder, i64_addr, T_ptr0)
            as_ptr   = LLVM.addrspacecast!(builder, as_ptr0, T_ret)
            LLVM.replace_uses!(call, as_ptr)
        end
        LLVM.erase!(call)
    end
    isempty(LLVM.uses(alloc_obj)) && LLVM.erase!(alloc_obj)
end

"""
    lower_gc_frame_intrinsics!(mod::LLVM.Module)

Lower Julia 1.11+ GC-frame intrinsics into linked-list operations on
`_whisk_pgcstack`. Frame layout in linear memory:

    ┌────────────────────┐
    │ prev_frame : i64   │  offset 0
    │ n_slots    : i64   │  offset 8
    │ slot[0..n-1]: i64  │  offset 16, 24, 32, ...
    └────────────────────┘
"""
function lower_gc_frame_intrinsics!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_wp  = LLVM.PointerType(T_w)
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    pgcstack = LLVM.globals(mod)["_whisk_pgcstack"]
    whisk_alloc = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    # ── julia.new_gc_frame(n) -> ptr addrspace(_)
    if haskey(LLVM.functions(mod), "julia.new_gc_frame")
        intr = LLVM.functions(mod)["julia.new_gc_frame"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            n = LLVM.arguments(call)[1]
            n64 = if LLVM.value_type(n) == T_w
                n
            else
                @dispose b=LLVM.IRBuilder() begin
                    LLVM.position!(b, call)
                    LLVM.zext!(b, n, T_w)
                end
            end
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                # bytes = 16 + 8*n  (header is 2 i64 fields)
                n_x8   = LLVM.shl!(b, n64, LLVM.ConstantInt(T_w, 3))
                bytes  = LLVM.add!(b, n_x8, LLVM.ConstantInt(T_w, 16))
                frame  = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [bytes])
                # prev_frame = 0 (set on push)
                prev_p = LLVM.inttoptr!(b, frame, T_wp)
                LLVM.store!(b, LLVM.ConstantInt(T_w, 0), prev_p)
                # n_slots = n
                n_addr = LLVM.add!(b, frame, LLVM.ConstantInt(T_w, 8))
                LLVM.store!(b, n64,
                              LLVM.inttoptr!(b, n_addr, T_wp))
                fr_ptr0 = LLVM.inttoptr!(b, frame, T_i8p)
                cast    = ret_ty isa LLVM.PointerType ?
                          LLVM.addrspacecast!(b, fr_ptr0, ret_ty) : frame
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.push_gc_frame(frame, _n)
    if haskey(LLVM.functions(mod), "julia.push_gc_frame")
        intr = LLVM.functions(mod)["julia.push_gc_frame"]
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f_ptr = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int = LLVM.ptrtoint!(b, f_ptr, T_w)
                top   = LLVM.load!(b, T_w, pgcstack)
                LLVM.store!(b, top, LLVM.inttoptr!(b, f_int, T_wp))
                LLVM.store!(b, f_int, pgcstack)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.pop_gc_frame(frame)
    if haskey(LLVM.functions(mod), "julia.pop_gc_frame")
        intr = LLVM.functions(mod)["julia.pop_gc_frame"]
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f_ptr = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int = LLVM.ptrtoint!(b, f_ptr, T_w)
                prev  = LLVM.load!(b, T_w, LLVM.inttoptr!(b, f_int, T_wp))
                LLVM.store!(b, prev, pgcstack)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.get_pgcstack() -> ptr
    if haskey(LLVM.functions(mod), "julia.get_pgcstack")
        intr = LLVM.functions(mod)["julia.get_pgcstack"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                cast = ret_ty isa LLVM.PointerType ?
                       LLVM.addrspacecast!(b, pgcstack, ret_ty) :
                       LLVM.ptrtoint!(b, pgcstack, ret_ty)
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.get_gc_frame_slot(frame, i) -> ptr
    if haskey(LLVM.functions(mod), "julia.get_gc_frame_slot")
        intr = LLVM.functions(mod)["julia.get_gc_frame_slot"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f, i = LLVM.arguments(call)
            i64 = if LLVM.value_type(i) == T_w
                i
            else
                @dispose b=LLVM.IRBuilder() begin
                    LLVM.position!(b, call)
                    LLVM.zext!(b, i, T_w)
                end
            end
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int    = LLVM.ptrtoint!(b, f, T_w)
                i_x8     = LLVM.shl!(b, i64, LLVM.ConstantInt(T_w, 3))
                offset   = LLVM.add!(b, i_x8, LLVM.ConstantInt(T_w, 16))
                slot_int = LLVM.add!(b, f_int, offset)
                slot_p0  = LLVM.inttoptr!(b, slot_int, T_i8p)
                cast     = ret_ty isa LLVM.PointerType ?
                           LLVM.addrspacecast!(b, slot_p0, ret_ty) : slot_p0
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.write_barrier — erase
    if haskey(LLVM.functions(mod), "julia.write_barrier")
        intr = LLVM.functions(mod)["julia.write_barrier"]
        for use in collect(LLVM.uses(intr))
            LLVM.erase!(LLVM.user(use)::LLVM.CallInst)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end
end

"""
    rewrite_host_pointer_constants!(mod::LLVM.Module)

In Julia-emitted IR, `inttoptr (ConstantInt to ptr)` only ever comes
from `pointer_from_objref` getting baked in by const-prop — always a
host pointer, regardless of value. wasm-internal addresses come back
as register values from `whisk_gc_alloc` calls, never as literal
constants. Our own emitted runtime code uses `inttoptr` only on
register values too.

Sweep all such constants and rewrite them to point at
`@_whisk_empty_memory`, a single 16-byte zero buffer. Empty
Memory{T} instances are observably identical (length=0, ptr=null);
the same buffer covers them all. Non-Memory host objects (Symbols,
type tags) typically aren't dereferenced — they're passed as ignored
ccall args.
"""
function rewrite_host_pointer_constants!(mod::LLVM.Module)
    empty_mem_name = "_whisk_empty_memory"
    empty_buf = if haskey(LLVM.globals(mod), empty_mem_name)
        LLVM.globals(mod)[empty_mem_name]
    else
        ty = LLVM.ArrayType(LLVM.Int8Type(), 16)
        gv = LLVM.GlobalVariable(mod, ty, empty_mem_name)
        LLVM.initializer!(gv, LLVM.null(ty))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        gv
    end

    seen = Set{LLVM.ConstantExpr}()
    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            for op in LLVM.operands(inst)
                op isa LLVM.ConstantExpr || continue
                LLVM.opcode(op) == LLVM.API.LLVMIntToPtr || continue
                src = LLVM.operands(op)[1]
                src isa LLVM.ConstantInt || continue
                push!(seen, op)
            end
        end
    end

    for ce in seen
        LLVM.replace_uses!(ce, empty_buf)
    end
    return length(seen)
end

"""
    bypass_argument_error_checks!(mod::LLVM.Module)

Bypass Julia's mistruncated wasm bounds checks. Find every
conditional branch whose fail target ends with
`call jl_argument_error; unreachable` and rewrite to unconditionally
take the success target.
"""
function bypass_argument_error_checks!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "jl_argument_error") || return
    err_fn = LLVM.functions(mod)["jl_argument_error"]
    fail_blocks = LLVM.BasicBlock[]
    for use in collect(LLVM.uses(err_fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        push!(fail_blocks, LLVM.parent(call))
    end

    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        for bb in LLVM.blocks(fn)
            term = LLVM.terminator(bb)
            term isa LLVM.BrInst || continue
            ops = LLVM.operands(term)
            length(ops) == 3 || continue
            true_tgt, false_tgt = ops[3], ops[2]
            true_is_fail  = true_tgt  in fail_blocks
            false_is_fail = false_tgt in fail_blocks
            true_is_fail == false_is_fail && continue
            keep = true_is_fail ? false_tgt : true_tgt
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, term)
                LLVM.br!(b, keep)
            end
            LLVM.erase!(term)
        end
    end
end
