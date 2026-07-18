#!/usr/bin/env julia
"""
test/runtests.jl — spin1nqs 回帰テスト

【目的】
開発過程で踏んだバグと、確定した参照値をテストとして固定化する。
physics.jl / hilbert.jl / model.jl を変更するたびに実行すること。

【実行】
    julia --project test/runtests.jl
GPU が無い環境ではカーネル系テストは自動スキップされる。

【規約 (2026-07 確定, 共同研究者と合意済み)】
  - ハミルトニアン: 2体項は全モードインデックスが |l| <= k_max を満たす
    「すべての」運動量保存項を含む (射影ハミルトニアン PHP, 実効的に |q| <= 2*k_max)
  - 相互作用係数: g/(2π), 全チャネルに 1/2
  - 運動エネルギー: k = l (整数角運動量そのもの)
  - max_transitions = 3*(4*k_max+1)*min(N, 3*(2*k_max+1))^2

【参照値の履歴】
  旧規約 (|q|<=k_max, 廃止):  E0 = -0.2232018172 (k=1) / -0.2417806075 (k=5)
  新規約 (現行):              E0 = -0.2238283842 (k=1) / -0.2419300599 (k=5)
"""

using Test
using Random
using Statistics
using LinearAlgebra
using SparseArrays
using ComponentArrays
using Zygote

# ★ プロジェクト構成に合わせて調整 ------------------------------------
# ed_from_kernel.jl / ed_spin1.jl は、末尾の main() 呼び出しを
#   if abspath(PROGRAM_FILE) == @__FILE__; main(); end
# で囲ってから include すること (include 時に main が走らないように)。
include("../src/model.jl")
include("../src/hilbert.jl")
include("../src/physics.jl")
include("../src/jacobian.jl")
include("ed_from_kernel.jl")   # build_H_from_kernel, build_H_independent, enumerate_basis
using .Model
using .Hilbert
using .Physics
# --------------------------------------------------------------------

using CUDA
const GPU_OK = CUDA.functional()
GPU_OK || @warn "CUDA が利用不可: GPU カーネル系テストをスキップします"

Random.seed!(20260712)

# ============================================================
# 新規約の参照値 (独立EDで確定, 2026-07)
# ============================================================
const REF = (
    # k_max=1, N=6, g0=0, g1=0.2, Sz=0, P=0 (次元79)
    k1 = (
        E0 = -0.2238283842,
        E1 = -0.1255182073,
        sector = Dict((3,0,3) => 0.454566, (2,2,2) => 0.231794,
                      (1,4,1) => 0.172714, (0,6,0) => 0.140926),
        NsNs1 = [3.19098388, 6.76393553, 3.19098388],   # <N_s(N_s-1)> = rho2(q=0)
        n_offzero = 0.029472,                            # l≠0 総占有
        S2_tower = [0.0, 6.0, 20.0, 42.0],               # 低励起の <S^2> (S=0,2,4,6)
    ),
    # k_max=5, N=6, 同上 (次元23607)
    k5 = (
        E0 = -0.2419300599,
        E1 = -0.1419631312,
        NsNs1 = [3.189819, 6.759276, 3.189819],
        n_offzero = 0.033711,                            # l≠0 総占有
        S2_tower = [0.0, 6.0, 20.0, 42.0],               # 低励起の <S^2> (S=0,2,4,6)
    ),
)

## # ============================================================
## # 1. ハミルトニアン: カーネル vs 独立実装 (回帰の本丸)
## # ------------------------------------------------------------
## # 履歴: 打ち切り規約の不一致 (|q|<=k_max vs 2k_max) が共同研究者との
## #       6e-4 のずれ、および「ITCIが変分原理を破る」誤解の原因だった。
## #       このテストは physics.jl の行列要素が第一原理 (F·F) と一致する
## #       ことを恒常的に保証する。
## # ============================================================
## @testset "ハミルトニアン回帰 (k_max=5, 次元23067)" begin
##     basis = enumerate_basis(6)   # Sz=0, P=0
##     dim = length(basis)
##     @test dim == 23607
## 
##     H_indep = build_H_independent(basis)
## 
##     # 独立実装のエルミート性と参照値
##     @test maximum(abs.(H_indep - transpose(H_indep))) < 1e-12
##     
##     vals, vecs, info = eigsolve(H_indep, dim, 2, :SR;
##                                 issymmetric = true,
##                                 krylovdim = 60,
##                                 maxiter = 500,
##                                 tol = 1e-12)
##     @test isapprox(vals[1], REF.k5.E0; atol = 1e-9)
##     @test isapprox(vals[2], REF.k5.E1; atol = 1e-9)
## 
##     if GPU_OK
##         H_kernel = build_H_from_kernel(basis)
##         # Float32 カーネルなので許容誤差は 1e-6 オーダー
##         @test maximum(abs.(H_kernel - transpose(H_kernel))) < 1e-5
##         @test maximum(abs.(H_kernel - H_indep)) < 1e-5
##     end
## end

## # ============================================================
## # 2. 参照観測量 (独立EDの基底状態から)
## # ------------------------------------------------------------
## # NQS の収束判定に使うターゲット値そのもの。規約変更時はここを更新する。
## # ============================================================
## @testset "参照観測量 (k_max=1)" begin
##     basis = enumerate_basis(6)
##     H = Matrix(build_H_independent(basis))
##     F = eigen(Symmetric(H))
##     gs = F.vectors[:, 1]
## 
##     n_modes = 3
##     cell(m, s) = (s - 1) * n_modes + m
## 
##     # スピンセクター分布
##     sec = Dict{NTuple{3,Int}, Float64}()
##     for (i, occ) in enumerate(basis)
##         ns = ntuple(s -> sum(Int(occ[cell(m, s)]) for m in 1:n_modes), 3)
##         sec[ns] = get(sec, ns, 0.0) + abs2(gs[i])
##     end
##     for (ns, w_ref) in REF.k1.sector
##         @test isapprox(sec[ns], w_ref; atol = 1e-5)
##     end
##     # スピン反転対称性: (a,b,c) と (c,b,a) の重みは厳密に等しい
##     @test isapprox(sec[(3,0,3)], sec[(3,0,3)]; atol = 0) # 自明
##     @test isapprox(sec[(1,4,1)], sec[(1,4,1)]; atol = 0)
## 
##     # <N_s(N_s-1)>
##     NsNs1 = zeros(3)
##     for (i, occ) in enumerate(basis)
##         w = abs2(gs[i])
##         for s in 1:3
##             Ns = sum(Int(occ[cell(m, s)]) for m in 1:n_modes)
##             NsNs1[s] += w * Ns * (Ns - 1)
##         end
##     end
##     @test isapprox(NsNs1, REF.k1.NsNs1; atol = 1e-5)
##     @test isapprox(NsNs1[1], NsNs1[3]; atol = 1e-12)   # スピン反転対称
## 
##     # l≠0 総占有 (NQS 学習診断のターゲット)
##     n_off = sum(abs2(gs[i]) * Int(occ[cell(m, s)])
##                 for (i, occ) in enumerate(basis)
##                 for s in 1:3, m in 1:n_modes if m != 2)
##     @test isapprox(n_off, REF.k1.n_offzero; atol = 1e-5)
## end

# ============================================================
# 3. tower of states (物理構造の固定)
# ------------------------------------------------------------
# 履歴: 低励起状態が全スピン S=0,2,4,6 の回転子の塔であることを確認済
#       (2026-07)。ハミルトニアン変更でこの構造が壊れたら検出する。
#       ギャップ ≈ (g1/4π)·S(S+1) は単一モード近似の予言。
# ============================================================
@testset "tower of states (k_max=1)" begin
    basis = enumerate_basis(6)
    D = length(basis)
    n_modes = 3
    cell(m, s) = (s - 1) * n_modes + m
    H = Matrix(build_H_independent(basis))
    vals, vecs, info = eigsolve(H, D, 4, :SR;
                               issymmetric = true,
                               krylovdim = 60,
                               maxiter = 500,
                               tol = 1e-12)

    # S^2 演算子: S^2 = 2N + Σ_{l1,l2} T_{abcd} a†_{l1,a} a†_{l2,b} a_{l2,d} a_{l1,c}
    T = build_FF_tensor()
    index = Dict(occ => i for (i, occ) in enumerate(basis))
    S2 = zeros(D, D)
    new_occ = zeros(Int8, n_modes * 3)
    for (i, occ) in enumerate(basis)
        S2[i, i] += 2.0 * 6
        for l1 in 1:n_modes, l2 in 1:n_modes,
            a in 1:3, b in 1:3, c in 1:3, d in 1:3
            t = T[a, b, c, d]
            abs(t) < 1e-15 && continue
            copyto!(new_occ, occ)
            i1 = cell(l1, c); new_occ[i1] == 0 && continue
            amp = sqrt(Float64(new_occ[i1])); new_occ[i1] -= 1
            i2 = cell(l2, d); new_occ[i2] == 0 && continue
            amp *= sqrt(Float64(new_occ[i2])); new_occ[i2] -= 1
            j2 = cell(l2, b); new_occ[j2] += 1; amp *= sqrt(Float64(new_occ[j2]))
            j1 = cell(l1, a); new_occ[j1] += 1; amp *= sqrt(Float64(new_occ[j1]))
            j = get(index, new_occ, 0)
            j == 0 && continue
            S2[j, i] += t * amp
        end
    end

    for k in 1:4
        v = vecs[k]
        s2 = v' * S2 * v
        @test isapprox(s2, REF.k1.S2_tower[k]; atol = 1e-6)
    end
    # ギャップが回転子の予言 6*g1/(4π) の 10% 以内 (l≠0 補正込み)
    gap = vals[2] - vals[1]
    @test isapprox(gap, 6 * 0.2 / (4π); rtol = 0.10)

    if GPU_OK
        S2_kernel = build_S2_from_kernel(basis)
        # Float32 カーネルなので許容誤差は 1e-6 オーダー
        @test maximum(abs.(S2_kernel - transpose(S2_kernel))) < 1e-5
        @test maximum(abs.(S2_kernel - S2)) < 1e-5
    end
end
## 
## # ============================================================
## # 4. 初期状態の健全性
## # ------------------------------------------------------------
## # 履歴: initialize_states! が target_Mz < 0 で負の占有数を直接書き込む
## #       バグがあった (DomainError の原因の一つ)。修正後の動作を固定。
## # ============================================================
## @testset "初期状態 (target_Mz の全符号)" begin
##     if GPU_OK
##         for mz in (-2, 0, 2)
##             basis = SpinorBasis(1, 6, 100)          # ★ 実際のコンストラクタ名に合わせる
##             initialize_states!(basis, mz)
##             st = Array(basis.states)
##             @test minimum(st) >= 0                    # 負の占有数がない
##             @test all(vec(sum(st, dims = (1, 2))) .== 6)  # 粒子数保存
##             sz = [sum(st[m, s, w] * (s - 2) for m in 1:3, s in 1:3)
##                   for w in 1:size(st, 3)]
##             @test all(sz .== mz)                      # 磁化が指定通り
##         end
##     else
##         @test_skip "GPU なし"
##     end
## end
## 
## # ============================================================
## # 5. サンプラー: 乱数端点の防御
## # ------------------------------------------------------------
## # 履歴: CUDA.rand (cuRAND) は区間 (0,1] で 1.0f0 を含む。クランプなしだと
## #       粒子選択がフォールスルーし、負の占有数 → 数十エポック後に
## #       確率的な DomainError となった。クランプ式の動作を固定。
## # ============================================================
## @testset "粒子選択のクランプ式" begin
##     N = 6
##     # カーネル内の式と同一のロジック (変更したら両方直すこと)
##     pick(r, n) = min(trunc(Int32, r * n) + 1, n)
##     @test pick(1.0f0, N) == N                # 危険な端点
##     @test pick(prevfloat(1.0f0), N) == N
##     @test pick(0.0f0, N) == 1
##     @test pick(0.5f0, N) in 1:N
##     @test pick(1.0f0, N - 1) == N - 1        # 2粒子目の選択
## end
## 
## # ============================================================
## # 6. サンプラー: 詳細釣り合い (統計テスト)
## # ------------------------------------------------------------
## # 履歴: Hastings 因子に q の縮退度 (同スピン・異運動量の行き先で2通りの
## #       q が同一終状態を与える) が抜けており、一様ターゲットの定常分布が
## #       (3/23, 12/23, 8/23) からずれるバグがあった。修正後の分布を検証。
## #       N=4, k_max=1, ψ=const で P=0, Sz=0 セクター (23状態) 上の一様分布。
## # 注意: 統計テストなので固定シード + 緩い許容誤差。まれな失敗は再実行で
## #       判断し、系統的に失敗するなら詳細釣り合いの破れを疑う。
## # ============================================================
## @testset "詳細釣り合い (一様ターゲット, N=4)" begin
##     if GPU_OK
##         n_walkers = 2000
##         n_steps   = 2000        # 熱平衡化込み
##         basis = SpinorBasis(1, 4, n_walkers)     # ★ コンストラクタ名
##         initialize_states!(basis, 0)
## 
##         # ψ = const でサンプリング: 受理率 = min(1, h_factor)
##         # ★ 一様 ψ での MH ループは実装に合わせて書く。
##         #    sampler の log_psi を 0 に固定して sample_step! を呼ぶ形が簡単。
##         run_uniform_sampling!(basis, n_steps)    # ★ 要実装 or 既存関数流用
## 
##         st = Array(basis.states)
##         @test minimum(st) >= 0
##         # スピンセクター頻度
##         counts = Dict((0,4,0) => 0, (1,2,1) => 0, (2,0,2) => 0)
##         for w in 1:n_walkers
##             ns = ntuple(s -> sum(st[:, s, w]), 3)
##             counts[ns] = get(counts, ns, 0) + 1
##         end
##         p_ref = Dict((0,4,0) => 3/23, (1,2,1) => 12/23, (2,0,2) => 8/23)
##         for (ns, p) in p_ref
##             @test isapprox(counts[ns] / n_walkers, p; atol = 0.04)
##         end
##     else
##         @test_skip "GPU なし"
##     end
## end
## 
## # ============================================================
## # 7. max_transitions の解析的上限
## # ------------------------------------------------------------
## # 履歴: バッファ不足でカーネルが隣のウォーカーのスロットを破壊し、
## #       確率的な DomainError を起こした。q 範囲拡張 (|q|<=2k_max) 後の
## #       上限式 3*(4k+1)*min(N,3L)^2 が実際の最大遷移数を覆うことを検証。
## # ============================================================
## @testset "max_transitions 上限 (k_max=1, N=6)" begin
##     k_max = 1; N = 6; L = 2k_max + 1
##     basis = enumerate_basis(N)
##     cell(m, s) = (s - 1) * L + m
## 
##     function count_transitions(occ)   # カーネルの列挙を再現 (拡張q)
##         cnt = 0
##         for m1 in 1:L, s1 in 1:3
##             occ[cell(m1, s1)] == 0 && continue
##             for m2 in 1:L, s2 in 1:3
##                 n2 = occ[cell(m2, s2)]
##                 n2 == 0 && continue
##                 (m1 == m2 && s1 == s2 && n2 < 2) && continue
##                 for q in -2k_max:2k_max
##                     m1n, m2n = m1 + q, m2 - q
##                     (1 <= m1n <= L && 1 <= m2n <= L) || continue
##                     cnt += 1                                   # 密度チャネル
##                     if s1 == 2 && s2 == 2
##                         cnt += 2
##                     elseif (s1, s2) in ((1, 3), (3, 1))
##                         cnt += 1
##                     elseif abs(s1 - s2) == 1
##                         cnt += 1
##                     end
##                 end
##             end
##         end
##         return cnt
##     end
## 
##     bound = 3 * (4k_max + 1) * min(N, 3L)^2
##     max_actual = maximum(count_transitions(occ) for occ in basis)
##     @test max_actual <= bound
##     @info "max_transitions: 実測最大 $max_actual / 上限 $bound"
## end
## 
## # ============================================================
## # 8. 観測量の計算式: <N_s(N_s-1)>
## # ------------------------------------------------------------
## # 履歴: Σ_m <n_m(n_m-1)> (交差項が抜ける誤式) を <N_s(N_s-1)> と
## #       混同するバグがあった。凝縮が強いと数値が近く気づきにくい。
## #       交差項が効く合成データで正しい式を固定する。
## # ============================================================
## @testset "<N_s(N_s-1)> の計算式" begin
##     # 2ウォーカーの合成データ: [n_modes=3, spin=3, walkers=2]
##     st = zeros(Int32, 3, 3, 2)
##     st[1, 1, 1] = 2; st[3, 1, 1] = 2      # w1: sz=-1 が l=∓1 に2個ずつ (N_1=4)
##     st[2, 2, 1] = 2                        # w1: sz=0  が l=0 に2個     (N_2=2)
##     st[2, 1, 2] = 4                        # w2: sz=-1 が l=0 に4個     (N_1=4)
##     st[2, 2, 2] = 2                        # w2: sz=0  (N_2=2)
## 
##     # 正: モード和を先に取ってから N(N-1)
##     Ns = dropdims(sum(st, dims = 1), dims = 1)                    # [3, 2]
##     good = vec(sum(Ns .* (Ns .- Int32(1)), dims = 2)) ./ 2        # [3]
##     # w1: 4*3=12, w2: 4*3=12 → 平均12 (spin1)。spin2: 2*1=2 両方 → 2
##     @test good[1] == 12.0
##     @test good[2] == 2.0
## 
##     # 誤 (旧バグ): 各モードで n(n-1) してから和 → 交差項が抜ける
##     bad = vec(sum(st .* (st .- Int32(1)), dims = (1, 3))) ./ 2
##     # w1 spin1: 2*1+2*1=4 (正は12), w2: 4*3=12 → 平均8 ≠ 12
##     @test bad[1] == 8.0
##     @test bad[1] != good[1]   # 両式は合成データで区別可能であること
## end
## 
## # ============================================================
## # 9. パラメータ移植 (3モード → 11モード)
## # ------------------------------------------------------------
## # 履歴: ウォーミングアップの列マッピング (m_new = m_old + Δk_max)。
## #       乱数初期化した重みでも移植の正しさは検証できる。
## # ============================================================
## @testset "パラメータ移植" begin
##     rng = Random.default_rng()
##     model3  = build_model(1, 64)              # ★ モデル生成関数名に合わせる
##     model11 = build_model(5, 64)
##     ps3, st3 = Lux.setup(rng, model3)
##     ps3 = ComponentVector{Float32}(ps3)
##     ps11, st11 = expand_params(ps3, 1, 5, model11)
## 
##     for ns in [(3,0,3), (2,2,2), (1,4,1), (0,6,0)]
##         occ3  = zeros(Float32, 3, 3);  occ11 = zeros(Float32, 11, 3)
##         for s in 1:3
##             occ3[2, s] = ns[s]; occ11[6, s] = ns[s]
##         end
##         o3  = eval_complex_network(model3,  reshape(occ3,  :, 1), ps3,  st3)
##         o11 = eval_complex_network(model11, reshape(occ11, :, 1), ps11, st11)
##         @test isapprox(o3[1], o11[1]; atol = 1e-5)
##     end
## end
## 
## # ============================================================
## # 10. 11モード参照値 (重いので環境変数で選択実行)
## # ------------------------------------------------------------
## #     SPIN1NQS_SLOW_TESTS=1 julia --project test/runtests.jl
## # ============================================================
## if get(ENV, "SPIN1NQS_SLOW_TESTS", "0") == "1"
##     @testset "11モード参照値 (k_max=5, 次元23607)" begin
##         using KrylovKit
##         basis = enumerate_basis_kmax(5, 6)    # ★ k_max 可変版の列挙関数
##         @test length(basis) == 23607
##         H = build_H_independent_kmax(basis, 5) # ★ 同上
##         vals, _, _ = eigsolve(H, length(basis), 2, :SR;
##                               issymmetric = true, tol = 1e-12)
##         @test isapprox(vals[1], REF.k5.E0; atol = 1e-8)
##         @test isapprox(vals[2], REF.k5.E1; atol = 1e-8)
##     end
## end

## # ============================================================
## # 検証: Zygote との一致 (runtests.jl に移植すること)
## # ------------------------------------------------------------
## # 「動くけど間違う」対策。特に (a) 活性化微分, (b) ComponentVector の
## # 順序, (c) 規約Bの共役, の3点はこのテストでしか保証できない。
## # ============================================================
## @testset "Jacobianテスト" begin
##     rng = Xoshiro(42)
##     model = build_momentum_nqs(1, hidden_dim=64)
##     ps, st = initialize_model(model, rng)
##  
##     function test_O_bar(model, ps, st; nb = 7)
##         rng = Random.default_rng()
##         n_in = 33                                        # ★ n_modes*3 に合わせる
##         inputs = CUDA.rand(Float32, n_in, nb) .* 3      # 適当な入力
##     
##         Ō, logψ = compute_O_bar(inputs, ps)
##     
##         # Zygote 側: 実部・虚部の Jacobian を別々に (既存 optimise.jl と同じ方法)
##         inputs_h = Array(inputs); ps_h = ps |> cpu_device()   # ★ CPU比較が楽
##         f_re(p) = real.(vec_eval(model, inputs_h, p, st))     # ★ 実装のeval関数に合わせる
##         f_im(p) = imag.(vec_eval(model, inputs_h, p, st))
##         J_re = Zygote.jacobian(f_re, ps_h)[1]                 # [nb, n_params]
##         J_im = Zygote.jacobian(f_im, ps_h)[1]
##         O_true = transpose(J_re) .+ im .* transpose(J_im)     # 真の微分 [n_params, nb]
##         Ō_ref = conj.(O_true)                                 # 規約B
##     
##         @test isapprox(Array(Ō), Ō_ref; rtol = 1e-4)
##         # 前向きも一致するか
##         @test isapprox(Array(logψ), vec_eval(model, inputs_h, ps_h, st); rtol = 1e-5)
##         println("compute_O_bar: Zygote と一致 ✓")
##     end
## end


println("\n全テスト完了")
