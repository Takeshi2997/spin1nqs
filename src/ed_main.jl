using SparseArrays
using LinearAlgebra
using KrylovKit
using Printf
using TOML
using LinearAlgebra
using CUDA
using Random
using ComponentArrays
using Dates

include("exact_hamiltonian.jl")
using .Exact

function main()
    @printf("=== スピン1ボソン 厳密対角化 ===\n")

    k_max = 1
    n_modes = 2 * k_max + 1
    n_particles = 6
    hbar2_over_2m = 1.0
    c0 = 0.0
    c1 = 0.2
    target_Mz = 0
    constrain_P = true
    
    # ハミルトニアン係数（接触相互作用）
    params = SystemParams(
        k_max,
        n_modes,
        hbar2_over_2m,
        c0,  # c0 (密度相互作用)
        c1   # c1 (スピン交換相互作用)
    )

    @printf("k_max = %d (モード数 %d), N = %d\n", k_max, n_modes, n_particles)
    @printf("c0 = %.3e, c1 = %.3e, hbar2/2m = %.3f\n", c0, c1, hbar2_over_2m)

    print("基底を列挙中... "); flush(stdout)
    t0 = time()
    basis = enumerate_basis(n_particles, n_modes, k_max, target_Mz, constrain_P)
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
    H = build_hamiltonian(basis, index, params)
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
    if !constrain_P
        p_weight = Dict{Int, Float64}()
        for (i, occ) in enumerate(basis)
            p = 0
            for s in 1:3, m in 1:n_modes
                p += Int(occ[cell_index(m, s)]) * (m - k_max - 1)
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
    @inline cell_index(m, s) = (s - 1) * n_modes + m
    sec = Dict{NTuple{3,Int}, Float64}()
    for (i, occ) in enumerate(basis)
        ns = ntuple(s -> sum(Int(occ[cell_index(m, s)]) for m in 1:n_modes), 3)
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
            Ns = sum(Int(occ[cell_index(m, s)]) for m in 1:n_modes)
            Ns_mean[s] += w * Ns
            Ns_diag[s] += w * Ns * (Ns - 1)
        end
    end
    @printf("\n<N_s>        = [%.6f, %.6f, %.6f]\n", Ns_mean...)
    @printf("<N_s(N_s-1)> = [%.6f, %.6f, %.6f]  (ρ₂(q=0) の理論値)\n", Ns_diag...)

    # <N_s>, <N_s(N_s-1)>  (rho2(q=0) の検証用)
    n_off = 0
    for (i, occ) in enumerate(basis)
        w = abs2(gs[i])
        for s in 1:3
            n_off += w * occ[cell_index(k_max + 1, s)]
        end
    end
    n_off = n_particles - n_off
    @printf("\n<N_off> = %.6f\n", n_off)

    return vals[1]
end

main()