# Heuristic scoring

The formula backend always uses the screenshot heuristic with 10% random mixing. No version, formula variant or mixing parameter is required. Qwen remains a separate backend.

For real visual tokens:
H = exp(-max(t_ref-t_i,0)/8) + exp(-d_i/8)
    + 1[observed] 1[rho>=0.9] exp(-max(L_class-t_i,0)/8).

For tactile tokens:
H = exp(-max(t_ref-t_i,0)/8)
    + 1[observed] 1[rho>=0.9] exp(-max(L_class-t_i,0)/8).

All times and durations use world-time steps, never seconds. Missing/zero features,
unmatched content and predicted frames receive no class bonus. Video classes use
DINO features. Tactile classes use the supplied tactile semantic feature field
`neoforce`; when absent, its class bonus is zero. Tactile and video prototypes
are independent. Only real observations advance class history; action remains FIFO.

Normalize by the fixed theoretical maximum (3 for visual, 2 for tactile), then:
score = 0.9 * H_normalized + 0.1 * U.

U is a uniform-marginal pseudorandom value in (0,1), derived by an integer avalanche
hash of token UID, decision world time and configured seed. Repeated scoring,
candidate permutations, ranks and checkpoint recomputation agree without consuming
global RNG. This is score mixing, not a guarantee that exactly 10% of retained
tokens are randomly selected.

The class threshold is fixed at 0.9 and the class decay scale at 8 for this variant.
The variant and random weight propagate through the task pipeline into resolved
training configs and serving overrides.
