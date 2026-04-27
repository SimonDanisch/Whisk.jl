## Garbage collector — precise mark-sweep with conservative scanning of
## payload bodies.
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
## Object layout (8-byte header + payload):
##
##     ┌──────────────────────────────────┐
##     │ size : u32  (total bytes incl hdr)│ — multiple of 8
##     │ flags: u32  (bit 0 = mark,        │
##     │              bit 1 = free)        │
##     ├──────────────────────────────────┤
##     │ payload …                         │
##     └──────────────────────────────────┘
##
## Roots come from Julia's GC frame intrinsics (`julia.{new,push,pop}_
## gc_frame`, `julia.get_gc_frame_slot`, `julia.get_pgcstack`). We
## lower them to real linked-list operations on a shadow root stack
## kept at the wasm global `_whisk_pgcstack` — that gives PRECISE
## roots (the Julia compiler tells us which locals are GC pointers).
## Payload bodies are scanned conservatively (every aligned 4-byte word
## treated as a potential heap pointer).

# ── Constants ─────────────────────────────────────────────────────────

const HEADER_SIZE      = UInt32(8)
const HEADER_FLAGS_OFF = UInt32(4)
const FLAG_MARK        = UInt32(0x1)
const FLAG_FREE        = UInt32(0x2)
const MIN_PAYLOAD      = UInt32(8)   # at least 8 bytes — fits free-list link

"""
    emit_wasm_runtime!(mod::LLVM.Module)

Inject the entire GC runtime into `mod`:

  * `@__heap_base`           — extern, supplied by wasm-ld.
  * `@_whisk_heap_next`      — i32 mutable global, lazy-init to
                               `&__heap_base`.
  * `@_whisk_pgcstack`       — i32 mutable global, top-of-stack ptr
                               for the GC root linked list.
  * `@_whisk_free_list_head` — i32 mutable global, head of free list.
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

    heap_next = _get_or_make_i32_global!(mod, "_whisk_heap_next", 0)
    pgcstack  = _get_or_make_i32_global!(mod, "_whisk_pgcstack",  0)
    free_list = _get_or_make_i32_global!(mod, "_whisk_free_list_head", 0)

    _emit_gc_alloc!(mod, heap_base, heap_next, free_list)
    _emit_gc_alloc_memory!(mod)
    _emit_mark_obj_at_addr!(mod, heap_base, heap_next)
    _emit_gc_mark!(mod, pgcstack)
    _emit_gc_sweep!(mod, heap_base, heap_next, free_list)
    _emit_gc_collect!(mod)
end

function _get_or_make_i32_global!(mod, name, init::Integer)
    if haskey(LLVM.globals(mod), name)
        return LLVM.globals(mod)[name]
    end
    T = LLVM.Int32Type()
    gv = LLVM.GlobalVariable(mod, T, name)
    LLVM.initializer!(gv, LLVM.ConstantInt(T, init))
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    return gv
end

# ─────────────────────────────────────────────────────────────────────
# whisk_gc_alloc — free-list-first allocator with header.
#
# def whisk_gc_alloc(sz: i32) -> i32:
#     payload_sz = max(sz, MIN_PAYLOAD)
#     total      = (payload_sz + HEADER_SIZE + 7) & ~7
#
#     # Free-list first-fit.
#     prev_link = &_whisk_free_list_head
#     cur       = *prev_link
#     while cur != 0:
#         if size(cur) >= total:
#             *prev_link = next_free(cur)
#             flags(cur) = 0
#             return cur + HEADER_SIZE
#         prev_link = next_free_addr(cur)
#         cur       = next_free(cur)
#
#     # Bump.
#     hn  = _whisk_heap_next
#     if hn == 0: hn = &__heap_base
#     hdr = hn
#     hn  = hn + total
#     _whisk_heap_next = hn
#     *(hdr     ) = total
#     *(hdr + 4 ) = 0
#     return hdr + HEADER_SIZE
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_alloc!(mod, heap_base, heap_next, free_list)
    T_i32  = LLVM.Int32Type()
    T_i32p = LLVM.PointerType(T_i32)

    fn = if haskey(LLVM.functions(mod), "whisk_gc_alloc")
        LLVM.functions(mod)["whisk_gc_alloc"]
    else
        LLVM.Function(mod, "whisk_gc_alloc",
                      LLVM.FunctionType(T_i32, [T_i32]))
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
        # ── entry: compute aligned total, prime free-list traversal
        LLVM.position!(b, entry)
        sz_min = _i32_max!(b, sz_param, LLVM.ConstantInt(T_i32, MIN_PAYLOAD))
        sz_hdr = LLVM.add!(b, sz_min, LLVM.ConstantInt(T_i32, HEADER_SIZE + 7))
        total  = LLVM.and!(b, sz_hdr,
                           LLVM.ConstantInt(T_i32, reinterpret(Int32, ~UInt32(7))))
        prev_link0 = LLVM.ptrtoint!(b, free_list, T_i32)
        cur0       = LLVM.load!(b, T_i32, free_list)
        LLVM.br!(b, f_loop)

        # ── freelist_loop: phi cur, prev_link
        LLVM.position!(b, f_loop)
        cur       = LLVM.phi!(b, T_i32)
        prev_link = LLVM.phi!(b, T_i32)
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur,
                              LLVM.ConstantInt(T_i32, 0))
        LLVM.br!(b, is_zero, bump, f_check)

        # ── freelist_check: peek size
        LLVM.position!(b, f_check)
        cur_size = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, cur, T_i32p))
        fits = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, cur_size, total)
        LLVM.br!(b, fits, f_take, f_next)

        # ── freelist_take: unlink, clear flags, return payload
        LLVM.position!(b, f_take)
        cur_payload_addr = LLVM.add!(b, cur, LLVM.ConstantInt(T_i32, HEADER_SIZE))
        cur_payload_ptr  = LLVM.inttoptr!(b, cur_payload_addr, T_i32p)
        next_free        = LLVM.load!(b, T_i32, cur_payload_ptr)
        prev_link_ptr    = LLVM.inttoptr!(b, prev_link, T_i32p)
        LLVM.store!(b, next_free, prev_link_ptr)
        flags_addr_take  = LLVM.add!(b, cur, LLVM.ConstantInt(T_i32, HEADER_FLAGS_OFF))
        flags_ptr_take   = LLVM.inttoptr!(b, flags_addr_take, T_i32p)
        LLVM.store!(b, LLVM.ConstantInt(T_i32, 0), flags_ptr_take)
        take_value       = cur_payload_addr
        LLVM.br!(b, done)

        # ── freelist_next: advance prev_link, cur
        LLVM.position!(b, f_next)
        cur_pl_addr2 = LLVM.add!(b, cur, LLVM.ConstantInt(T_i32, HEADER_SIZE))
        new_prev_link = cur_pl_addr2
        new_cur       = LLVM.load!(b,  T_i32,
                                   LLVM.inttoptr!(b, cur_pl_addr2, T_i32p))
        LLVM.br!(b, f_loop)

        # ── bump
        LLVM.position!(b, bump)
        cur_hn    = LLVM.load!(b, T_i32, heap_next)
        is_unset  = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_hn,
                                LLVM.ConstantInt(T_i32, 0))
        base_addr = LLVM.ptrtoint!(b, heap_base, T_i32)
        actual_hn = LLVM.select!(b, is_unset, base_addr, cur_hn)
        new_hn    = LLVM.add!(b, actual_hn, total)
        LLVM.store!(b, new_hn, heap_next)
        LLVM.store!(b, total,
                       LLVM.inttoptr!(b, actual_hn, T_i32p))
        flags_addr_bump = LLVM.add!(b, actual_hn, LLVM.ConstantInt(T_i32, HEADER_FLAGS_OFF))
        LLVM.store!(b, LLVM.ConstantInt(T_i32, 0),
                       LLVM.inttoptr!(b, flags_addr_bump, T_i32p))
        bump_payload    = LLVM.add!(b, actual_hn, LLVM.ConstantInt(T_i32, HEADER_SIZE))
        LLVM.br!(b, done)

        # ── done: phi take_value vs bump_payload
        LLVM.position!(b, done)
        result = LLVM.phi!(b, T_i32)
        LLVM.append!(LLVM.incoming(result),
                     [(take_value, f_take), (bump_payload, bump)])
        LLVM.ret!(b, result)

        # ── phi fixups for f_loop
        LLVM.append!(LLVM.incoming(cur),
                     [(cur0, entry), (new_cur, f_next)])
        LLVM.append!(LLVM.incoming(prev_link),
                     [(prev_link0, entry), (new_prev_link, f_next)])
    end
end

@inline function _i32_max!(b::LLVM.IRBuilder, a, c)
    cond = LLVM.icmp!(b, LLVM.API.LLVMIntUGT, a, c)
    return LLVM.select!(b, cond, a, c)
end

# ─────────────────────────────────────────────────────────────────────
# jl_alloc_genericmemory_unchecked — Julia's Memory{T} ctor.
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_alloc_memory!(mod::LLVM.Module)
    T_i32 = LLVM.Int32Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "jl_alloc_genericmemory_unchecked")
        LLVM.functions(mod)["jl_alloc_genericmemory_unchecked"]
    else
        ft = LLVM.FunctionType(T_i32, [T_i32, T_i32, T_i32])
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
        ms_addr   = LLVM.call!(b, whisk_alloc_ft, whisk_alloc,
                                [LLVM.ConstantInt(T_i32, 8)])
        data_addr = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [len_bytes])

        ms_ptr   = LLVM.inttoptr!(b, ms_addr, T_i8p)
        slot_ptr = LLVM.gep!(b, LLVM.Int8Type(), ms_ptr,
                              [LLVM.ConstantInt(T_i32, 4)])
        LLVM.store!(b, data_addr, slot_ptr)

        retval = if T_ret isa LLVM.IntegerType
            ms_addr
        else
            ms_ptr0 = LLVM.inttoptr!(b, ms_addr, T_i8p)
            LLVM.addrspacecast!(b, ms_ptr0, T_ret)
        end
        LLVM.ret!(b, retval)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Mark phase: walk pgcstack frames, mark reachable, recurse.
# ─────────────────────────────────────────────────────────────────────

function _emit_mark_obj_at_addr!(mod, heap_base, heap_next)
    T_i32  = LLVM.Int32Type()
    T_i32p = LLVM.PointerType(T_i32)

    fn = if haskey(LLVM.functions(mod), "_whisk_mark_obj_at_addr")
        LLVM.functions(mod)["_whisk_mark_obj_at_addr"]
    else
        LLVM.Function(mod, "_whisk_mark_obj_at_addr",
                      LLVM.FunctionType(LLVM.VoidType(), [T_i32]))
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
        base_addr = LLVM.ptrtoint!(b, heap_base, T_i32)
        heap_end  = LLVM.load!(b, T_i32, heap_next)
        ge_base   = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, addr, base_addr)
        lt_end    = LLVM.icmp!(b, LLVM.API.LLVMIntULT, addr, heap_end)
        ok        = LLVM.and!(b, ge_base, lt_end)
        LLVM.br!(b, ok, in_heap, done)

        LLVM.position!(b, in_heap)
        LLVM.br!(b, find_loop)

        LLVM.position!(b, find_loop)
        obj = LLVM.phi!(b, T_i32)
        past_end = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, obj, heap_end)
        LLVM.br!(b, past_end, done, find_check)

        LLVM.position!(b, find_check)
        obj_size = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, obj, T_i32p))
        obj_end  = LLVM.add!(b, obj, obj_size)
        cont_lo  = LLVM.icmp!(b, LLVM.API.LLVMIntULE, obj, addr)
        cont_hi  = LLVM.icmp!(b, LLVM.API.LLVMIntULT, addr, obj_end)
        contains = LLVM.and!(b, cont_lo, cont_hi)
        LLVM.br!(b, contains, do_mark, find_next)

        LLVM.position!(b, find_next)
        next_obj = LLVM.add!(b, obj, obj_size)
        LLVM.br!(b, find_loop)

        # ── do_mark: skip if already marked, else set + scan
        LLVM.position!(b, do_mark)
        flags_addr  = LLVM.add!(b, obj, LLVM.ConstantInt(T_i32, HEADER_FLAGS_OFF))
        flags_ptr   = LLVM.inttoptr!(b, flags_addr, T_i32p)
        flags       = LLVM.load!(b, T_i32, flags_ptr)
        marked_bit  = LLVM.and!(b, flags, LLVM.ConstantInt(T_i32, FLAG_MARK))
        is_marked   = LLVM.icmp!(b, LLVM.API.LLVMIntNE, marked_bit,
                                   LLVM.ConstantInt(T_i32, 0))
        LLVM.br!(b, is_marked, already_b, set_mark)

        LLVM.position!(b, already_b)
        LLVM.br!(b, done)

        LLVM.position!(b, set_mark)
        new_flags = LLVM.or!(b, flags, LLVM.ConstantInt(T_i32, FLAG_MARK))
        LLVM.store!(b, new_flags, flags_ptr)
        LLVM.br!(b, scan_loop)

        LLVM.position!(b, scan_loop)
        scan_i = LLVM.phi!(b, T_i32)
        in_obj = LLVM.icmp!(b, LLVM.API.LLVMIntULT, scan_i, obj_size)
        LLVM.br!(b, in_obj, scan_body, done)

        LLVM.position!(b, scan_body)
        word_addr = LLVM.add!(b, obj, scan_i)
        word_ptr  = LLVM.inttoptr!(b, word_addr, T_i32p)
        word_val  = LLVM.load!(b, T_i32, word_ptr)
        LLVM.call!(b, LLVM.function_type(fn), fn, [word_val])
        scan_next = LLVM.add!(b, scan_i, LLVM.ConstantInt(T_i32, 4))
        LLVM.br!(b, scan_loop)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(obj),
                     [(base_addr, in_heap), (next_obj, find_next)])
        LLVM.append!(LLVM.incoming(scan_i),
                     [(LLVM.ConstantInt(T_i32, HEADER_SIZE), set_mark),
                      (scan_next, scan_body)])
    end
end

function _emit_gc_mark!(mod, pgcstack)
    T_i32  = LLVM.Int32Type()
    T_i32p = LLVM.PointerType(T_i32)

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
        cur_frame0 = LLVM.load!(b, T_i32, pgcstack)
        LLVM.br!(b, frame_loop)

        LLVM.position!(b, frame_loop)
        cur_frame = LLVM.phi!(b, T_i32)
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_frame,
                              LLVM.ConstantInt(T_i32, 0))
        LLVM.br!(b, is_zero, done, frame_check)

        LLVM.position!(b, frame_check)
        n_addr  = LLVM.add!(b, cur_frame, LLVM.ConstantInt(T_i32, 4))
        n_slots = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, n_addr, T_i32p))
        LLVM.br!(b, slot_loop)

        LLVM.position!(b, slot_loop)
        i = LLVM.phi!(b, T_i32)
        in_range = LLVM.icmp!(b, LLVM.API.LLVMIntULT, i, n_slots)
        LLVM.br!(b, in_range, slot_body, next_frame)

        LLVM.position!(b, slot_body)
        i_x4        = LLVM.shl!(b, i, LLVM.ConstantInt(T_i32, 2))
        slot_offset = LLVM.add!(b, i_x4, LLVM.ConstantInt(T_i32, 8))
        slot_addr   = LLVM.add!(b, cur_frame, slot_offset)
        slot_value  = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, slot_addr, T_i32p))
        LLVM.call!(b, mark_obj_ft, mark_obj, [slot_value])
        i_next = LLVM.add!(b, i, LLVM.ConstantInt(T_i32, 1))
        LLVM.br!(b, slot_loop)

        LLVM.position!(b, next_frame)
        prev_frame = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, cur_frame, T_i32p))
        LLVM.br!(b, frame_loop)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(cur_frame),
                     [(cur_frame0, entry), (prev_frame, next_frame)])
        LLVM.append!(LLVM.incoming(i),
                     [(LLVM.ConstantInt(T_i32, 0), frame_check),
                      (i_next, slot_body)])
    end
end

# ─────────────────────────────────────────────────────────────────────
# Sweep phase: walk heap, marked → unmark, unmarked → free list.
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_sweep!(mod, heap_base, heap_next, free_list)
    T_i32  = LLVM.Int32Type()
    T_i32p = LLVM.PointerType(T_i32)

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
        # Reset free list — sweep rebuilds it.
        LLVM.store!(b, LLVM.ConstantInt(T_i32, 0), free_list)
        base_addr = LLVM.ptrtoint!(b, heap_base, T_i32)
        heap_end  = LLVM.load!(b, T_i32, heap_next)
        LLVM.br!(b, obj_loop)

        LLVM.position!(b, obj_loop)
        obj = LLVM.phi!(b, T_i32)
        past = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, obj, heap_end)
        LLVM.br!(b, past, done, obj_check)

        LLVM.position!(b, obj_check)
        size_v    = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, obj, T_i32p))
        flags_addr= LLVM.add!(b, obj, LLVM.ConstantInt(T_i32, HEADER_FLAGS_OFF))
        flags_ptr = LLVM.inttoptr!(b, flags_addr, T_i32p)
        flags     = LLVM.load!(b, T_i32, flags_ptr)
        marked    = LLVM.and!(b, flags, LLVM.ConstantInt(T_i32, FLAG_MARK))
        was_marked = LLVM.icmp!(b, LLVM.API.LLVMIntNE, marked,
                                   LLVM.ConstantInt(T_i32, 0))
        LLVM.br!(b, was_marked, is_marked, is_free)

        LLVM.position!(b, is_marked)
        cleared = LLVM.and!(b, flags,
                            LLVM.ConstantInt(T_i32, reinterpret(Int32, ~FLAG_MARK)))
        LLVM.store!(b, cleared, flags_ptr)
        LLVM.br!(b, after)

        LLVM.position!(b, is_free)
        old_head     = LLVM.load!(b, T_i32, free_list)
        payload_addr = LLVM.add!(b, obj, LLVM.ConstantInt(T_i32, HEADER_SIZE))
        payload_ptr  = LLVM.inttoptr!(b, payload_addr, T_i32p)
        LLVM.store!(b, old_head, payload_ptr)
        LLVM.store!(b, obj, free_list)
        free_flags   = LLVM.or!(b, flags, LLVM.ConstantInt(T_i32, FLAG_FREE))
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
    # at link time. We also append it to `@llvm.used` so the LLVM
    # optimizer doesn't DCE it before we get to wasm-ld (it has no
    # callers inside the module — tests and user code drive it from JS).
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_gc_collect"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    mod = LLVM.parent(fn)
    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)

    mark_fn  = LLVM.functions(mod)["_whisk_gc_mark"]
    sweep_fn = LLVM.functions(mod)["_whisk_gc_sweep"]
    entry    = LLVM.BasicBlock(fn, "entry")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        LLVM.call!(b, LLVM.function_type(mark_fn),  mark_fn,  LLVM.Value[])
        LLVM.call!(b, LLVM.function_type(sweep_fn), sweep_fn, LLVM.Value[])
        LLVM.ret!(b)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Lowering passes — invoked from finish_module!
# ─────────────────────────────────────────────────────────────────────

"""
    lower_gc_alloc_wasm32!(mod::LLVM.Module)

Replace every `julia.gc_alloc_obj(_, sz_i32, _)` with `whisk_gc_alloc(sz_i32)`
plus an addrspacecast back to the original return type.
"""
function lower_gc_alloc_wasm32!(mod::LLVM.Module)
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
            i32_addr = LLVM.call!(builder, whisk_alloc_ft, whisk_alloc, [sz])
            as_ptr0  = LLVM.inttoptr!(builder, i32_addr, T_ptr0)
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
`_whisk_pgcstack`:

  * `julia.new_gc_frame(N)`         — alloc `8 + 4N` bytes via
                                       `whisk_gc_alloc`, init slots to 0.
  * `julia.push_gc_frame(frame, _)` — frame.prev = pgcstack;
                                       pgcstack = frame.
  * `julia.pop_gc_frame(frame)`     — pgcstack = frame.prev.
  * `julia.get_pgcstack()`          — return &pgcstack as ptr.
  * `julia.get_gc_frame_slot(f, i)` — return &(f + 8 + 4*i).
  * `julia.write_barrier(...)`      — erase (no generations).
"""
function lower_gc_frame_intrinsics!(mod::LLVM.Module)
    T_i32  = LLVM.Int32Type()
    T_i32p = LLVM.PointerType(T_i32)
    T_i8p  = LLVM.PointerType(LLVM.Int8Type())

    pgcstack = LLVM.globals(mod)["_whisk_pgcstack"]
    whisk_alloc = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    # ── julia.new_gc_frame(n) -> ptr addrspace(_) ────────────────────
    if haskey(LLVM.functions(mod), "julia.new_gc_frame")
        intr = LLVM.functions(mod)["julia.new_gc_frame"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            n = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                n_x4   = LLVM.shl!(b, n, LLVM.ConstantInt(T_i32, 2))
                bytes  = LLVM.add!(b, n_x4, LLVM.ConstantInt(T_i32, 8))
                frame  = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [bytes])
                prev_p = LLVM.inttoptr!(b, frame, T_i32p)
                LLVM.store!(b, LLVM.ConstantInt(T_i32, 0), prev_p)
                n_addr = LLVM.add!(b, frame, LLVM.ConstantInt(T_i32, 4))
                LLVM.store!(b, n,
                              LLVM.inttoptr!(b, n_addr, T_i32p))
                fr_ptr0 = LLVM.inttoptr!(b, frame, T_i8p)
                cast    = ret_ty isa LLVM.PointerType ?
                          LLVM.addrspacecast!(b, fr_ptr0, ret_ty) : frame
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.push_gc_frame(frame, _n) ──────────────────────────────
    if haskey(LLVM.functions(mod), "julia.push_gc_frame")
        intr = LLVM.functions(mod)["julia.push_gc_frame"]
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f_ptr = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int = LLVM.ptrtoint!(b, f_ptr, T_i32)
                top   = LLVM.load!(b, T_i32, pgcstack)
                LLVM.store!(b, top, LLVM.inttoptr!(b, f_int, T_i32p))
                LLVM.store!(b, f_int, pgcstack)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.pop_gc_frame(frame) ───────────────────────────────────
    if haskey(LLVM.functions(mod), "julia.pop_gc_frame")
        intr = LLVM.functions(mod)["julia.pop_gc_frame"]
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f_ptr = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int = LLVM.ptrtoint!(b, f_ptr, T_i32)
                prev  = LLVM.load!(b, T_i32, LLVM.inttoptr!(b, f_int, T_i32p))
                LLVM.store!(b, prev, pgcstack)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.get_pgcstack() -> ptr ─────────────────────────────────
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

    # ── julia.get_gc_frame_slot(frame, i) -> ptr ────────────────────
    if haskey(LLVM.functions(mod), "julia.get_gc_frame_slot")
        intr = LLVM.functions(mod)["julia.get_gc_frame_slot"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f, i = LLVM.arguments(call)
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int    = LLVM.ptrtoint!(b, f, T_i32)
                i_x4     = LLVM.shl!(b, i, LLVM.ConstantInt(T_i32, 2))
                offset   = LLVM.add!(b, i_x4, LLVM.ConstantInt(T_i32, 8))
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

    # ── julia.write_barrier — erase (we have no generations).
    if haskey(LLVM.functions(mod), "julia.write_barrier")
        intr = LLVM.functions(mod)["julia.write_barrier"]
        for use in collect(LLVM.uses(intr))
            LLVM.erase!(LLVM.user(use)::LLVM.CallInst)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end
end

"""
    bypass_argument_error_checks!(mod::LLVM.Module)

Julia's `emit_memorynew` LLVM codegen emits a bounds check using
host-i64 constants truncated to wasm32 i32, producing `slt n*4, -1` —
which always falls into the throw path. Until Julia codegen learns
wasm32's word size we rewrite each conditional branch whose fail target
ends with `call jl_argument_error; unreachable` to be unconditional
toward the success target.
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
