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
    k_max = 4
    n_modes = 2 * k_max + 1
    n_particles = 10
    hbar2_over_2m = 1.0
    c0 = 0.0
    c1 = 8.0
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

    dirname = "./data/" * Dates.format(now(), "yyyymmdd") * "_exact"
    mkpath(dirname)
    filename = dirname * "/space_correlation_N$(n_particles)_k$(k_max)_c0$(c0)_c1$(@sprintf("%.3f", c1)).txt"
    logfilename = dirname * "/log_N$(n_particles)_k$(k_max)_c0$(c0)_c1$(@sprintf("%.3f", c1)).txt"

    open(filename, "w") do io
        @printf(io, "k_max = %d (モード数 %d), N = %d\n", k_max, n_modes, n_particles)
    end

    f = open(logfilename, "w")
    
    @printf(f, "k_max = %d (モード数 %d), N = %d\n", k_max, n_modes, n_particles)
    @printf(f, "c0 = %.3e, c1 = %.3e, hbar2/2m = %.3f\n", c0, c1, hbar2_over_2m)

    print("基底を列挙中... "); flush(stdout)
    t0 = time()
    basis = enumerate_basis(n_particles, n_modes, k_max, target_Mz, constrain_P)
    dim = length(basis)
    @printf("完了 (%.1f 秒)\n", time() - t0)
    @printf(f, "セクター次元: %d\n\n", dim)

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

    @printf(f, "基底エネルギー   E0 = %.10f\n", vals[1])
    if length(vals) >= 2
        @printf(f, "第1励起          E1 = %.10f  (gap = %.6f)\n", vals[2], vals[2] - vals[1])
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
        @printf(f, "\n基底状態の全運動量 P の分布:\n")
        for p in sort(collect(keys(p_weight)))
            w = p_weight[p]
            if w > 1e-8
                @printf(f, "  P = %+d : %.6f\n", p, w)
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
    @printf(f, "\n基底状態の (N₋₁, N₀, N₊₁) 分布 (上位10件):\n")
    for (ns, w) in first(sort(collect(sec), by = x -> -x[2]), 10)
        w > 1e-6 && @printf(f, "  %s : %.6f\n", ns, w)
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
    @printf(f, "\n<N_s>        = [%.6f, %.6f, %.6f]\n", Ns_mean...)
    @printf(f, "<N_s(N_s-1)> = [%.6f, %.6f, %.6f]  (ρ₂(q=0) の理論値)\n", Ns_diag...)

    # <N_s>, <N_s(N_s-1)>  (rho2(q=0) の検証用)
    n_off = 0
    for (i, occ) in enumerate(basis)
        w = abs2(gs[i])
        for s in 1:3
            n_off += w * occ[cell_index(k_max + 1, s)]
        end
    end
    n_off = n_particles - n_off
    @printf(f, "\n<N_off> = %.6f\n", n_off)

    close(f)

    eval_space_correlation(gs, basis, index, k_max, filename)

    return vals[1]
end

function eval_space_correlation(gs::Vector{Float64}, basis, index, k_max, filename)
    touch(filename)

    n_modes = 2k_max + 1
    q_range = -2k_max : 2k_max
    rho2_q = zeros(ComplexF64, length(q_range), 3, 3)
    cell(m, s) = (s - 1) * n_modes + m
    occ = zeros(Int8, n_modes * 3)

    for (j, occ_j) in enumerate(basis)
        cj = gs[j]
        abs(cj) < 1e-14 && continue

        for s1 in 1:3, s2 in 1:3, (qi, q) in enumerate(q_range)
            for m1 in 1:n_modes, m2 in 1:n_modes
                m1n = m1 + q;  m2n = m2 - q
                (1 <= m1n <= n_modes && 1 <= m2n <= n_modes) || continue

                copyto!(occ, occ_j)
                # 消滅: (m1,s1) → (m2,s2)  [順序に注意]
                i1 = cell(m1, s1)
                occ[i1] == 0 && continue
                amp = sqrt(Float64(occ[i1])); occ[i1] -= 1
                i2 = cell(m2, s2)
                occ[i2] == 0 && continue
                amp *= sqrt(Float64(occ[i2])); occ[i2] -= 1
                # 生成: (m2n,s2) → (m1n,s1)
                j2 = cell(m2n, s2); occ[j2] += 1; amp *= sqrt(Float64(occ[j2]))
                j1 = cell(m1n, s1); occ[j1] += 1; amp *= sqrt(Float64(occ[j1]))

                i = get(index, occ, 0)
                i == 0 && continue
                rho2_q[qi, s1, s2] += conj(gs[i]) * cj * amp
            end
        end
    end
   
    total = sum(real.(rho2_q[2 * k_max + 1, :, :]))
    @printf("Σ_ss' ρ₂(0) = %.10f \n", total)

    rho2_q_11 = rho2_q[:, 1, 1]
    rho2_q_22 = rho2_q[:, 2, 2]
    rho2_q_33 = rho2_q[:, 3, 3]
    ## 1, 2; 2, 3; 3, 1 の相関用
    rho2_q_12 = rho2_q[:, 1, 2]
    rho2_q_23 = rho2_q[:, 2, 3]
    rho2_q_31 = rho2_q[:, 3, 1]
 

    # フーリエ変換
    L_box = Float32(2 * π)
    x_grid = Float32.(range(-L_box/2, L_box/2, length=1000))
    k_list = Float32.((2 * π / L_box) .* q_range)
    W = exp.(-1.0f0im .* x_grid .* k_list')

    cor11_x_vec = real.(W * rho2_q_11) ./ L_box
    cor22_x_vec = real.(W * rho2_q_22) ./ L_box
    cor33_x_vec = real.(W * rho2_q_33) ./ L_box

    cor12_x_vec = real.(W * rho2_q_12) ./ L_box
    cor23_x_vec = real.(W * rho2_q_23) ./ L_box
    cor31_x_vec = real.(W * rho2_q_31) ./ L_box

    rho2_total = dropdims(sum(rho2_q, dims=(2,3)), dims=(2,3))   # [n_q]
    cord_x_vec = real.([sum(exp(im*q*x) * rho2_total[qi] for (qi,q) in enumerate(q_range))/(2π) for x in x_grid])
 
    open(filename, "a") do io
        @printf(io, "x, C11, C22, C33, C12, C23, C31, Cd,\n")
    end
    for x in 1:1000
        cor11_x = cor11_x_vec[x]
        cor22_x = cor22_x_vec[x]
        cor33_x = cor33_x_vec[x]
        cor12_x = cor12_x_vec[x]
        cor23_x = cor23_x_vec[x]
        cor31_x = cor31_x_vec[x]
        cord_x = cord_x_vec[x]
        open(filename, "a") do io
            @printf(io, "%6.3f, %6.9f, %6.9f, %6.9f, %6.9f, %6.9f, %6.9f, %6.9f\n", 
            x_grid[x], cor11_x, cor22_x, cor33_x, cor12_x, cor23_x, cor31_x, cord_x)
        end
    end
    
    @printf("n11_diag, n22_diag, n33_diag, max_correlation,\n") 
    @printf("%6.3f, %6.3f, %6.3f, %6.3f, \n", rho2_q_11[k_max + 1], rho2_q_22[k_max + 1], rho2_q_33[k_max + 1], maximum(abs.(cor11_x_vec - cor33_x_vec)))
    
    return nothing
end

main()