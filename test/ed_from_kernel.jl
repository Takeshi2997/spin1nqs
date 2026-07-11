#!/usr/bin/env julia
"""
ed_from_kernel.jl — physics.jl の散乱カーネルを再利用した厳密対角化

【目的と位置づけ】
physics.jl の _scattering_kernel! をそのまま使ってハミルトニアン行列を構築し、
独立実装の ED (ed_spin1.jl) と比較する「回帰テスト」。

  - 独立 ED (ed_spin1.jl) : F·F テンソルから第一原理で構築 = 正解の基準
  - 本ファイル             : 実際の GPU カーネルの出力から構築 = physics.jl の検証

両者が一致すれば physics.jl の行列要素は正しい。physics.jl を変更するたびに
実行することで、実装の破壊を自動検出できる。

【注意】
本ファイルは physics.jl のバグをそのまま継承する。「正解」としては使わないこと。
独立 ED との一致確認が本質であり、片方だけでは検証にならない。

【使い方】
1. ★印の箇所を physics.jl の実際のシグネチャに合わせて調整
2. julia --project ed_from_kernel.jl
3. k_max=1 (次元79) で既知値 E0 = -0.22320182 と、独立EDとの一致を確認

【必要パッケージ】 CUDA, SparseArrays, KrylovKit
"""

using CUDA
using SparseArrays
using LinearAlgebra
using KrylovKit
using Printf

include("../src/model.jl")
# physics.jl のカーネルを読み込む (プロジェクト構成に合わせてパスを調整)
include("../src/physics.jl")   # ★ _scattering_kernel! が定義されているファイル

# ============================================================
# パラメータ (config と合わせる)
# ============================================================
const K_MAX  = 5        # まず k_max=1 (次元79) で検証してから大きくする
const N_PART = 6
const G0     = 0.0
const G1     = 0.2
const HBAR2_2M = 1.0

const N_MODES = 2 * K_MAX + 1
const MAX_T   = 3 * (4 * K_MAX + 1) * min(N_PART, 3 * N_MODES)^2   # physics.jl と同じ上限

# ============================================================
# 1. セクター (Sz=0, P=0) の基底列挙
# ============================================================
function enumerate_basis(n_part::Int; target_sz::Int = 0, constrain_p::Bool = true)
    n_cells = N_MODES * 3
    cells = [(m, s) for s in 1:3 for m in 1:N_MODES]
    basis = Vector{Vector{Int8}}()
    occ = zeros(Int8, n_cells)

    function rec(i::Int, rest::Int, sz::Int, p::Int)
        if i > n_cells
            if rest == 0 && sz == target_sz && (!constrain_p || p == 0)
                push!(basis, copy(occ))
            end
            return
        end
        if rest == 0
            for j in i:n_cells; occ[j] = 0; end
            if sz == target_sz && (!constrain_p || p == 0)
                push!(basis, copy(occ))
            end
            return
        end
        m, s = cells[i]
        for n in 0:rest
            occ[i] = Int8(n)
            rec(i + 1, rest - n, sz + n * (s - 2), p + n * (m - K_MAX - 1))
        end
        occ[i] = 0
    end

    rec(1, n_part, 0, 0)
    return basis
end

# 占有数ベクトル (長さ n_modes*3) のフラット化順序:
#   index = (s-1)*N_MODES + m   (states[m, s] の column-major と一致)
@inline cell_index(m, s) = (s - 1) * N_MODES + m

# ============================================================
# 2. カーネル出力からハミルトニアンを構築
# ============================================================
"""
セクターの全基底状態を「ウォーカー」として _scattering_kernel! に流し、
出力 (proposed_states, matrix_elements) から疎行列 H を組み立てる。

H[j, i] = Σ V_{x_i → x_j}  (相互作用; 同じ (i,j) への複数寄与は sparse() が加算)
H[i, i] += E_kin(x_i)      (運動エネルギー; physics.jl の _compute_kinetic と同じ式)
"""
function build_H_from_kernel(basis::Vector{Vector{Int8}}; batch::Int = 4096)
    dim = length(basis)

    # 占有数ベクトル → 基底インデックスの辞書
    index = Dict{Vector{Int8}, Int}()
    sizehint!(index, dim)
    for (i, occ) in enumerate(basis)
        index[occ] = i
    end

    rows = Int[]; cols = Int[]; vals = Float64[]

    n_chunks = cld(dim, batch)
    for (ci, chunk) in enumerate(Iterators.partition(1:dim, batch))
        nb = length(chunk)

        # 基底状態をウォーカー配列に詰める [n_modes, 3, nb]
        states_h = Array{Int32}(undef, N_MODES, 3, nb)
        for (b, i) in enumerate(chunk)
            states_h[:, :, b] .= reshape(Int32.(basis[i]), N_MODES, 3)
        end
        states   = CuArray(states_h)
        proposed = CUDA.zeros(Int32, N_MODES, 3, MAX_T, nb)
        melems   = CUDA.zeros(Float32, MAX_T, nb)

        # ★★★ physics.jl の実際のシグネチャに合わせて調整する ★★★
        # 例: 結合定数を直接渡す形の場合
        threads = 256
        blocks = cld(nb, threads)
        @cuda threads=threads blocks=blocks Physics._scattering_kernel!(
            states, proposed, melems,
            Int64(K_MAX), Float32(G0), Float32(G1),
        )
        # params 構造体を渡す形なら、それを構築して渡すこと。
        # 引数の順序・型はカーネル定義と厳密に一致させる。
        # ★★★ ここまで ★★★

        CUDA.synchronize()
        P = Array(proposed)
        M = Array(melems)

        # 疎行列エントリの収集
        occ_buf = Vector{Int8}(undef, N_MODES * 3)
        for b in 1:nb, t in 1:MAX_T
            v = M[t, b]
            v == 0.0f0 && continue    # パディング & 係数ゼロのチャネル (g0=0 の m1*m2=0 など)
            for k in 1:N_MODES*3
                occ_buf[k] = Int8(P[mod1(k, N_MODES), div(k - 1, N_MODES) + 1, t, b])
            end
            # ↑ P[m, s, t, b] を index = (s-1)*N_MODES + m の順で平坦化
            j = get(index, occ_buf, 0)
            if j == 0
                error("カーネルがセクター外の状態を生成 (chunk=$ci, walker=$b, slot=$t)。" *
                      "保存則の破れ = physics.jl のバグの可能性")
            end
            push!(rows, j)
            push!(cols, chunk[b])
            push!(vals, Float64(v))
            occ_buf = copy(occ_buf)   # Dict のキーに使ったので新しいバッファに
        end

        @printf("  チャンク %d / %d 完了\n", ci, n_chunks)
        flush(stdout)
    end

    H = sparse(rows, cols, vals, dim, dim)

    # 運動エネルギー (対角): E_kin = hbar2/2m * Σ l² n_l   (k = l, 論文規約)
    kin = zeros(Float64, dim)
    for (i, occ) in enumerate(basis)
        for s in 1:3, m in 1:N_MODES
            n = occ[cell_index(m, s)]
            n > 0 && (kin[i] += HBAR2_2M * (m - K_MAX - 1)^2 * n)
        end
    end

    return H + spdiagm(kin)
end

# ============================================================
# 3. 独立実装 (F·F テンソル) — ed_spin1.jl から流用
#    比較の基準。physics.jl とは無関係に構築される。
# ============================================================
function build_FF_tensor()
    s = 1 / sqrt(2)
    Fx = ComplexF64[0 s 0; s 0 s; 0 s 0]
    Fy = ComplexF64[0 -im*s 0; im*s 0 -im*s; 0 im*s 0]
    Fz = ComplexF64[-1 0 0; 0 0 0; 0 0 1]
    T = zeros(ComplexF64, 3, 3, 3, 3)
    for F in (Fx, Fy, Fz), a in 1:3, b in 1:3, c in 1:3, d in 1:3
        T[a, b, c, d] += F[a, c] * F[b, d]
    end
    return real.(T)
end

function build_H_independent(basis::Vector{Vector{Int8}})
    dim = length(basis)
    index = Dict{Vector{Int8}, Int}()
    for (i, occ) in enumerate(basis); index[occ] = i; end

    T = build_FF_tensor()
    D = zeros(3, 3, 3, 3)
    for a in 1:3, b in 1:3; D[a, b, a, b] = 1.0; end

    v0 = G0 / (2π); v1 = G1 / (2π)
    half = 0.5                            # 全チャネルの 1/2 (physics.jl / 共同研究者の規約)

    channels = Tuple{Int,Int,Int,Int,Float64}[]
    for a in 1:3, b in 1:3, c in 1:3, d in 1:3
        coef = (v0 * D[a,b,c,d] + v1 * T[a,b,c,d]) * half
        abs(coef) > 1e-15 && push!(channels, (a, b, c, d, coef))
    end

    rows = Int[]; cols = Int[]; vals = Float64[]
    new_occ = zeros(Int8, N_MODES * 3)

    for (i, occ) in enumerate(basis)
        # 運動エネルギー
        Ek = 0.0
        for s in 1:3, m in 1:N_MODES
            n = occ[cell_index(m, s)]
            n > 0 && (Ek += HBAR2_2M * (m - K_MAX - 1)^2 * n)
        end
        push!(rows, i); push!(cols, i); push!(vals, Ek)

        # 相互作用: a†_{l1n,c} a†_{l2n,d} a_{l2,b} a_{l1,a}
        for l1 in 1:N_MODES, l2 in 1:N_MODES, q in -2*K_MAX:2*K_MAX
            l1n = l1 + q; l2n = l2 - q
            (1 <= l1n <= N_MODES && 1 <= l2n <= N_MODES) || continue
            for (a, b, c, d, coef) in channels
                copyto!(new_occ, occ)
                i1 = cell_index(l1, a)
                new_occ[i1] == 0 && continue
                amp = sqrt(Float64(new_occ[i1])); new_occ[i1] -= 1
                i2 = cell_index(l2, b)
                new_occ[i2] == 0 && continue
                amp *= sqrt(Float64(new_occ[i2])); new_occ[i2] -= 1
                j2 = cell_index(l2n, d); new_occ[j2] += 1; amp *= sqrt(Float64(new_occ[j2]))
                j1 = cell_index(l1n, c); new_occ[j1] += 1; amp *= sqrt(Float64(new_occ[j1]))
                j = get(index, new_occ, 0)
                j == 0 && continue
                push!(rows, j); push!(cols, i); push!(vals, coef * amp)
            end
        end
    end
    return sparse(rows, cols, vals, dim, dim)
end

# ============================================================
# 4. メイン: 構築 → 比較 → 対角化
# ============================================================
function main()
    @printf("=== physics.jl カーネル再利用 ED (回帰テスト) ===\n")
    @printf("k_max = %d, N = %d, g0 = %g, g1 = %g\n\n", K_MAX, N_PART, G0, G1)

    basis = enumerate_basis(N_PART)
    dim = length(basis)
    @printf("セクター次元 (Sz=0, P=0): %d\n\n", dim)

    println("カーネル出力から H を構築中...")
    H_kernel = build_H_from_kernel(basis)

    println("独立実装 (F·F) から H を構築中...")
    H_indep = build_H_independent(basis)

    # ---- 検証1: エルミート性 ----
    herm = maximum(abs.(H_kernel - transpose(H_kernel)))
    @printf("\n[検証1] エルミート性: max|H - Hᵀ| = %.3e  %s\n",
            herm, herm < 1e-5 ? "OK" : "★ NG: カーネルの行列要素が非対称")

    # ---- 検証2: 独立実装との一致 ----
    diff = maximum(abs.(H_kernel - H_indep))
    @printf("[検証2] 独立EDとの一致: max|ΔH| = %.3e  %s\n",
            diff, diff < 1e-5 ? "OK" : "★ NG: physics.jl の行列要素に不一致")
    if diff >= 1e-5
        # 最大不一致の要素を特定して表示 (デバッグ用)
        A = abs.(H_kernel - H_indep)
        I_, J_, V_ = findnz(A)
        k = argmax(V_)
        i, j = I_[k], J_[k]
        @printf("  最大不一致: H[%d, %d] kernel=%.6f indep=%.6f\n",
                i, j, H_kernel[i, j], H_indep[i, j])
        println("  遷移: ", basis[j], " → ", basis[i])
    end

    # ---- 検証3: 基底エネルギー ----
    vals_k, _, _ = eigsolve(H_kernel, dim, 1, :SR; issymmetric = true, tol = 1e-12)
    vals_i, _, _ = eigsolve(H_indep,  dim, 1, :SR; issymmetric = true, tol = 1e-12)
    @printf("\n[検証3] 基底エネルギー\n")
    @printf("  カーネル由来: E0 = %.10f\n", vals_k[1])
    @printf("  独立実装    : E0 = %.10f\n", vals_i[1])
    if K_MAX == 1 && N_PART == 6 && G0 == 0.0 && G1 == 0.2
        @printf("  既知の参照値: E0 = −0.2238282\n")
    elseif K_MAX == 5 && N_PART == 6 && G0 == 0.0 && G1 == 0.2
        @printf("  既知の参照値: E0 = -0.241916\n")
    end

    ok = herm < 1e-5 && diff < 1e-5
    @printf("\n%s\n", ok ? "✅ 全検証パス: physics.jl の行列要素は独立実装と一致" :
                          "❌ 検証失敗: physics.jl を確認せよ")
    return ok
end

main()
