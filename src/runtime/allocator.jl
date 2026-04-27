## Allocator — M2 territory.
##
## Strategy B from the brainstorm: linear memory + bump allocator (M2)
## upgraded to a precise mark-sweep GC (M3). Both live here. Empty until
## M2 lands; M1 doesn't allocate.
