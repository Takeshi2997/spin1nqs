module Exact

using ..Physics

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
function enumerate_basis(n_part::Int, n_modes::Int, k_max::Int, target_sz::Int, constrain_p::Bool)
    n_cells = n_modes * 3
    # cells[i] = (m, s)
    cells = [(m, s) for s in 1:3 for m in 1:n_modes]

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
        p_unit  = m - k_max - 1  # mode index -> l

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
function kinetic_energy(occ::Vector{Int8}, params::SystemParams)
    k_max = params.k_max
    n_modes = params.n_modes
    hbar2_over_2m = params.hbar2_over_2m
    E = 0.0
    for s in 1:3, m in 1:n_modes
        n = occ[cell_index(m, s)]
        if n > 0
            l = m - k_max - 1
            E += hbar2_over_2m * Float64(l^2) * Float64(n)
        end
    end
    return E
end

function build_hamiltonian(basis::Vector{Vector{Int8}},
                           index::Dict{Vector{Int8}, Int}, 
                           params::SystemParams)
    dim = length(basis)
    T = build_FF_tensor()
    D = build_density_tensor()

    v0 = params.c0 / (2 * π)
    v1 = params.c1 / (2 * π)
    k_max = params.k_max
    n_modes = params.n_modes
    hbar2_over_2m = params.hbar2_over_2m

    # 非ゼロチャネル (a,b,c,d, coef) を事前に集める
    channels = Tuple{Int,Int,Int,Int,Float64}[]
    for a in 1:3, b in 1:3, c in 1:3, d in 1:3
        coef = (v0 * D[a,b,c,d] + v1 * T[a,b,c,d]) / 2
        if abs(coef) > 1e-15
            push!(channels, (a, b, c, d, coef))
        end
    end
    @printf("非ゼロチャネル数: %d\n", length(channels))

    rows = Int[]; cols = Int[]; vals = Float64[]
    sizehint!(rows, dim * 50); sizehint!(cols, dim * 50); sizehint!(vals, dim * 50)

    new_occ = zeros(Int8, n_modes * 3)

    for (i, occ) in enumerate(basis)
        # 対角: 運動エネルギー
        push!(rows, i); push!(cols, i); push!(vals, kinetic_energy(occ, params))

        # 相互作用
        for l1 in 1:n_modes, l2 in 1:n_modes
            for q in -2*k_max:2*k_max
                l1n = l1 + q
                l2n = l2 - q
                (1 <= l1n <= n_modes && 1 <= l2n <= n_modes) || continue

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



end # module Exact