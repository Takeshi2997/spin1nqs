using SparseArrays
using LinearAlgebra
using KrylovKit
using Printf

# ============================================================
# パラメータ
# ============================================================
const K_MAX      = 5        # モード数 = 2*K_MAX + 1
const N_PART     = 6        # 粒子数
const G0         = 0.0      # c0 (spin-independent)
const G1         = 0.2      # c1 (spin interaction)
const HBAR2_2M   = 1.0
const TARGET_SZ  = 0
const CONSTRAIN_P = true   # true: P=0 に制限, false: 全 P
const HALF_FACTOR = true    # true: 全チャネルに 1/2 (physics.jl と同じ)

const N_MODES = 2 * K_MAX + 1
const TWO_PI  = 2π

# ============================================================
# 1. Fock 基底の列挙
#    状態は長さ N_MODES*3 の Vector{Int8} (column-major: [m, s] -> (s-1)*N_MODES + m)
# ============================================================
"""
占有数ベクトル occ の (mode m, spin s) 成分へのインデックス
"""
@inline cell_index(m, s) = (s - 1) * N_MODES + m

"""
Sz (と必要なら P) の制約を満たす Fock 基底をすべて列挙する。
枝刈り付き DFS。
"""
function enumerate_basis(n_part::Int, target_sz::Int, constrain_p::Bool)
    n_cells = N_MODES * 3
    # cells[i] = (m, s)
    cells = [(m, s) for s in 1:3 for m in 1:N_MODES]

    basis = Vector{Vector{Int8}}()
    occ   = zeros(Int8, n_cells)

    function rec(i::Int, rest::Int, sz::Int, p::Int)
        if i > n_cells
            if rest == 0 && sz == target_sz && (!constrain_p || p == 0)
                push!(basis, copy(occ))
            end
            return
        end
        if rest == 0
            # 残りは全部 0
            for j in i:n_cells; occ[j] = 0; end
            if sz == target_sz && (!constrain_p || p == 0)
                push!(basis, copy(occ))
            end
            return
        end

        m, s = cells[i]
        sz_unit = s - 2          # spin index 1,2,3 -> m = -1, 0, +1
        p_unit  = m - K_MAX - 1  # mode index -> l

        for n in 0:rest
            occ[i] = Int8(n)
            rec(i + 1, rest - n, sz + n * sz_unit, p + n * p_unit)
        end
        occ[i] = 0
    end

    rec(1, n_part, 0, 0)
    return basis
end

# ============================================================
# 2. スピン1行列と F·F テンソル
#    T[a,b,c,d] = Σ_μ (F_μ)_{ac} (F_μ)_{bd}
#    -> a†_c a†_d a_b a_a の係数 (physics.jl のチャネル分岐と等価)
# ============================================================
function build_FF_tensor()
    s = 1 / sqrt(2)
    Fx = [0.0  s    0.0;
          s    0.0  s;
          0.0  s    0.0]
    Fy = ComplexF64[0.0    -im*s   0.0;
                    im*s    0.0   -im*s;
                    0.0     im*s   0.0]
    Fz = [-1.0 0.0 0.0;
           0.0 0.0 0.0;
           0.0 0.0 1.0]

    T = zeros(ComplexF64, 3, 3, 3, 3)
    for F in (ComplexF64.(Fx), Fy, ComplexF64.(Fz))
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
            T[a,b,c,d] += F[a,c] * F[b,d]
        end
    end
    @assert maximum(abs.(imag.(T))) < 1e-12 "F·F は実であるべき"
    return real.(T)
end

"""密度チャネル δ_ac δ_bd"""
function build_density_tensor()
    D = zeros(Float64, 3, 3, 3, 3)
    for a in 1:3, b in 1:3
        D[a,b,a,b] = 1.0
    end
    return D
end

# ============================================================
# 3. ハミルトニアンの疎行列構築
# ============================================================
"""
a†_{l1n,c} a†_{l2n,d} a_{l2,b} a_{l1,a} を occ に作用させる。
戻り値: (新しい occ, 振幅) または nothing
"""
@inline function apply_pair!(new_occ::Vector{Int8}, occ::Vector{Int8},
                             l1::Int, a::Int, l2::Int, b::Int,
                             l1n::Int, c::Int, l2n::Int, d::Int)
    copyto!(new_occ, occ)

    # 消滅: a_{l1,a} を先に作用 (a_{l2,b} a_{l1,a} の順序)
    i1 = cell_index(l1, a)
    new_occ[i1] == 0 && return nothing
    amp = sqrt(Float64(new_occ[i1])); new_occ[i1] -= 1

    i2 = cell_index(l2, b)
    new_occ[i2] == 0 && return nothing
    amp *= sqrt(Float64(new_occ[i2])); new_occ[i2] -= 1

    # 生成: a†_{l2n,d} を先に
    j2 = cell_index(l2n, d)
    new_occ[j2] += 1; amp *= sqrt(Float64(new_occ[j2]))

    j1 = cell_index(l1n, c)
    new_occ[j1] += 1; amp *= sqrt(Float64(new_occ[j1]))

    return amp
end

"""運動エネルギー Σ_l l² n_l"""
function kinetic_energy(occ::Vector{Int8})
    E = 0.0
    for s in 1:3, m in 1:N_MODES
        n = occ[cell_index(m, s)]
        if n > 0
            l = m - K_MAX - 1
            E += HBAR2_2M * Float64(l^2) * Float64(n)
        end
    end
    return E
end

function build_hamiltonian(basis::Vector{Vector{Int8}},
                           index::Dict{Vector{Int8}, Int})
    dim = length(basis)
    T = build_FF_tensor()
    D = build_density_tensor()

    v0 = G0 / TWO_PI
    v1 = G1 / TWO_PI
    half = HALF_FACTOR ? 0.5 : 1.0

    # 非ゼロチャネル (a,b,c,d, coef) を事前に集める
    channels = Tuple{Int,Int,Int,Int,Float64}[]
    for a in 1:3, b in 1:3, c in 1:3, d in 1:3
        coef = (v0 * D[a,b,c,d] + v1 * T[a,b,c,d]) * half
        if abs(coef) > 1e-15
            push!(channels, (a, b, c, d, coef))
        end
    end
    @printf("非ゼロチャネル数: %d\n", length(channels))

    rows = Int[]; cols = Int[]; vals = Float64[]
    sizehint!(rows, dim * 50); sizehint!(cols, dim * 50); sizehint!(vals, dim * 50)

    new_occ = zeros(Int8, N_MODES * 3)

    for (i, occ) in enumerate(basis)
        # 対角: 運動エネルギー
        push!(rows, i); push!(cols, i); push!(vals, kinetic_energy(occ))

        # 相互作用
        for l1 in 1:N_MODES, l2 in 1:N_MODES
            for q in -2*K_MAX:2*K_MAX
                l1n = l1 + q
                l2n = l2 - q
                (1 <= l1n <= N_MODES && 1 <= l2n <= N_MODES) || continue

                for (a, b, c, d, coef) in channels
                    amp = apply_pair!(new_occ, occ, l1, a, l2, b, l1n, c, l2n, d)
                    amp === nothing && continue
                    j = get(index, new_occ, 0)
                    j == 0 && continue          # セクター外 (保存則より本来出ない)
                    push!(rows, j); push!(cols, i); push!(vals, coef * amp)
                end
            end
        end

        if i % 20000 == 0
            @printf("  構築中... %d / %d (%.1f%%)\n", i, dim, 100i/dim)
            flush(stdout)
        end
    end

    H = sparse(rows, cols, vals, dim, dim)
    return H
end

# ============================================================
# 4. メイン
# ============================================================
function main()
    @printf("=== スピン1ボソン 厳密対角化 ===\n")
    @printf("k_max = %d (モード数 %d), N = %d\n", K_MAX, N_MODES, N_PART)
    @printf("g0 = %.3e, g1 = %.3e, hbar2/2m = %.3f\n", G0, G1, HBAR2_2M)
    @printf("Sz = %d, P制限 = %s, 1/2因子 = %s\n\n",
            TARGET_SZ, CONSTRAIN_P ? "あり (P=0)" : "なし (全P)",
            HALF_FACTOR ? "あり" : "なし")

    print("基底を列挙中... "); flush(stdout)
    t0 = time()
    basis = enumerate_basis(N_PART, TARGET_SZ, CONSTRAIN_P)
    dim = length(basis)
    @printf("完了 (%.1f 秒)\n", time() - t0)
    @printf("セクター次元: %d\n\n", dim)

    print("インデックス辞書を構築中... "); flush(stdout)
    t0 = time()
    index = Dict{Vector{Int8}, Int}()
    sizehint!(index, dim)
    for (i, occ) in enumerate(basis)
        index[occ] = i
    end
    @printf("完了 (%.1f 秒)\n\n", time() - t0)

    print("ハミルトニアンを構築中...\n"); flush(stdout)
    t0 = time()
    H = build_hamiltonian(basis, index)
    @printf("完了 (%.1f 秒)\n", time() - t0)
    @printf("非ゼロ要素数: %d (密度 %.2e)\n", nnz(H), nnz(H) / dim^2)

    # エルミート性チェック
    herm = maximum(abs.(H - transpose(H)))
    @printf("エルミート性: max|H - Hᵀ| = %.3e %s\n\n",
            herm, herm < 1e-10 ? "(OK)" : "(NG!)")

    print("Lanczos で基底状態を計算中...\n"); flush(stdout)
    t0 = time()
    # 対称行列なので eigsolve に issymmetric=true を渡す
    vals, vecs, info = eigsolve(H, dim, 2, :SR;
                                issymmetric = true,
                                krylovdim = 60,
                                maxiter = 500,
                                tol = 1e-12)
    @printf("完了 (%.1f 秒, 収束: %s)\n\n", time() - t0,
            info.converged >= 2 ? "OK" : "不十分")

    @printf("基底エネルギー   E0 = %.10f\n", vals[1])
    if length(vals) >= 2
        @printf("第1励起          E1 = %.10f  (gap = %.6f)\n", vals[2], vals[2] - vals[1])
    end

    # ----- 基底状態の解析 -----
    gs = vecs[1]
    gs ./= norm(gs)

    # 全運動量 P の分布 (P制限なしのとき、どの P に基底状態があるか)
    if !CONSTRAIN_P
        p_weight = Dict{Int, Float64}()
        for (i, occ) in enumerate(basis)
            p = 0
            for s in 1:3, m in 1:N_MODES
                p += Int(occ[cell_index(m, s)]) * (m - K_MAX - 1)
            end
            p_weight[p] = get(p_weight, p, 0.0) + abs2(gs[i])
        end
        @printf("\n基底状態の全運動量 P の分布:\n")
        for p in sort(collect(keys(p_weight)))
            w = p_weight[p]
            if w > 1e-8
                @printf("  P = %+d : %.6f\n", p, w)
            end
        end
    end

    # スピンセクター分布
    sec = Dict{NTuple{3,Int}, Float64}()
    for (i, occ) in enumerate(basis)
        ns = ntuple(s -> sum(Int(occ[cell_index(m, s)]) for m in 1:N_MODES), 3)
        sec[ns] = get(sec, ns, 0.0) + abs2(gs[i])
    end
    @printf("\n基底状態の (N₋₁, N₀, N₊₁) 分布 (上位10件):\n")
    for (ns, w) in first(sort(collect(sec), by = x -> -x[2]), 10)
        w > 1e-6 && @printf("  %s : %.6f\n", ns, w)
    end

    # <N_s>, <N_s(N_s-1)>  (rho2(q=0) の検証用)
    Ns_mean = zeros(3); Ns_diag = zeros(3)
    for (i, occ) in enumerate(basis)
        w = abs2(gs[i])
        for s in 1:3
            Ns = sum(Int(occ[cell_index(m, s)]) for m in 1:N_MODES)
            Ns_mean[s] += w * Ns
            Ns_diag[s] += w * Ns * (Ns - 1)
        end
    end
    @printf("\n<N_s>        = [%.6f, %.6f, %.6f]\n", Ns_mean...)
    @printf("<N_s(N_s-1)> = [%.6f, %.6f, %.6f]  (ρ₂(q=0) の理論値)\n", Ns_diag...)

    return vals[1]
end

main()