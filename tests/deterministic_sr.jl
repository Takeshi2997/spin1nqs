#!/usr/bin/env julia
"""
deterministic_sr.jl — サンプリングを排除した厳密 SR 最適化

【目的】
セクション次元 23,607 は全列挙できるので、モンテカルロを使わずに
変分エネルギーと SR 更新を「無限サンプル極限」で厳密に計算する。

  到達 E → -0.2419300599  ⇒ ansatz の表現力は十分。
                             残差はすべてサンプリング起因と確定
  -0.2398 付近で停滞       ⇒ そこが hidden_dim の真の変分限界

【使うもの / 使わないもの】
  使う:   H (ed_from_kernel.jl の疎行列。独立実装と一致検証済み)
          ネットワーク (model, ps, st)
  使わない: ED の基底状態ベクトル (使ったら教師あり学習になる)
            サンプラー (hilbert.jl)
            散乱カーネル (physics.jl) ← Hψ の疎行列積が代替する

【MC 版との対応】
  一様重み 1/N_s  →  p(x) = |ψ(x)|² / Σ|ψ|²
  ψ比の明示的な和 →  E_loc = (Hψ)./ψ
  それ以外 (Ō の構築、規約B、中心化、正則化、solve) は完全に同じ。

【コスト】 1エポックあたり
  NN 前向き 23,607 回 × 2パス + 逆向き相当 (compute_O_bar) 23,607 回
  + 疎行列積 1 回 (2.3M 非ゼロ、CPU で数 ms)
  → 現在の MC 版 1 エポック (16,384 サンプル) と同程度
"""

using CUDA, LinearAlgebra, SparseArrays, Printf

# ============================================================
# 0. 前処理: 全配置の入力行列を一度だけ作る (ループの外)
# ============================================================
"""
    build_full_inputs(basis, n_modes) -> CuMatrix{Float32} [3*n_modes, n_configs]

basis :: Vector{Vector{Int8}}  (enumerate_basis の出力, cell_index = (s-1)*n_modes + m)
FlattenLayer の入力順序 (column-major) と一致させる。
"""
function build_full_inputs(basis::Vector{Vector{Int8}}, n_modes::Int)
    n_cfg = length(basis)
    A = Array{Float32}(undef, 3 * n_modes, n_cfg)
    for (i, occ) in enumerate(basis)
        @inbounds for k in 1:(3 * n_modes)
            A[k, i] = Float32(occ[k])
        end
    end
    return CuArray(A)          # [33, 23607] ≈ 3 MB
end

# ============================================================
# 1. パス1: 全配置の ψ, p(x), E_loc, E
# ============================================================
"""
    exact_state(model, ps, st, inputs_full, H; chunk) -> (p, E_loc, E, logψ)

p     :: Vector{Float64} [n_cfg]  厳密な確率重み (Σp = 1)
E_loc :: Vector{ComplexF64}       (Hψ)./ψ
E     :: Float64                  ψ†Hψ/ψ†ψ (= 厳密な変分エネルギー)

数値上の注意:
  - logψ の最大実部を引いてから exp (Float32 のオーバーフロー回避)
  - ψ, Hψ, E_loc は ComplexF64 (23,607 要素なので CPU で十分・安価)
  - |ψ|² が極小の配置では E_loc が巨大になるが、常に p(x) を掛けて
    使うので寄与は有限。マスクで落としてもよい (下記 pmask)
"""
function exact_state(model, ps, st, inputs_full::CuMatrix{Float32},
                     H::SparseMatrixCSC; chunk::Int = 4096)
    n_cfg = size(inputs_full, 2)
    logψ = Vector{ComplexF64}(undef, n_cfg)

    for c in Iterators.partition(1:n_cfg, chunk)
        out = eval_complex_network(model, view(inputs_full, :, c), ps, st)  # [nb]
        logψ[c] .= ComplexF64.(Array(out))
    end

    shift = maximum(real, logψ)
    ψ = exp.(logψ .- shift)                    # ComplexF64
    nrm2 = sum(abs2, ψ)
    p = abs2.(ψ) ./ nrm2                        # Σp = 1

    Hψ = H * ψ                                  # 疎行列積 (CPU, 数 ms)
    E_loc = Hψ ./ ψ
    E = real(sum(p .* E_loc))                   # = ψ†Hψ/ψ†ψ

    return p, E_loc, E, logψ
end

# ============================================================
# 2. パス2: p(x) 重みで S と g を厳密に構築 (規約B: Ō = conj(∂logψ))
# ============================================================
"""
    deterministic_sr_update(model, ps, st, inputs_full, p, E_loc, E;
                            chunk, lambda, epsilon) -> delta_p

MC 版 compute_SR_update と同一の式。重みが 1/N_s → p(x) に変わるだけ。

  Ō_mean = Σ_x p(x) Ō(x)
  g      = Σ_x p(x) Ō(x) [E_loc(x) − E]          ← 中心化はこの形で完結
  S      = Σ_x p(x) conj(Ō(x)) Ō(x)ᵀ − conj(Ō_mean) Ō_meanᵀ
"""
function deterministic_sr_update(model, ps, st, inputs_full::CuMatrix{Float32},
                                 p::Vector{Float64}, E_loc::Vector{ComplexF64},
                                 E::Float64;
                                 chunk::Int = 2048,
                                 lambda::Float64 = 1e-4,
                                 epsilon::Float64 = 1e-8,
                                 pmask::Float64 = 1e-30)
    n_cfg = size(inputs_full, 2)
    n_par = Lux.parameterlength(model)

    O_sum  = CUDA.zeros(ComplexF64, n_par)
    OE_sum = CUDA.zeros(ComplexF64, n_par)
    S_sum  = CUDA.zeros(ComplexF64, n_par, n_par)

    for c in Iterators.partition(1:n_cfg, chunk)
        # 寄与が完全に無視できる配置はスキップ (数値安全 + 高速化)
        all(<(pmask), @view p[c]) && continue

        Ō, _ = compute_O_bar(view(inputs_full, :, c), ps)      # [n_par, nb] ComplexF32
        Ōd = ComplexF64.(Ō)

        w  = CuArray(ComplexF64.(@view p[c]))                   # [nb]
        we = CuArray(ComplexF64.(p[c] .* (E_loc[c] .- E)))      # [nb]

        O_sum  .+= Ōd * w                                       # Σ p Ō
        OE_sum .+= Ōd * we                                      # Σ p Ō (E_loc − E) = g
        S_sum  .+= (conj.(Ōd) .* transpose(w)) * transpose(Ōd)  # Σ p Ō* Ōᵀ
    end

    Ō_mean = O_sum                                              # Σp = 1 なので平均そのもの
    g = OE_sum
    S = S_sum .- conj.(Ō_mean) * transpose(Ō_mean)

    # 正則化 (下記「正則化の変更点」参照。統計ノイズが無いので ε は極小で可)
    Sh = Array(S)
    d  = real.(diag(Sh))
    Sreg = Sh + lambda * Diagonal(ComplexF64.(d)) + epsilon * I

    delta = Sreg \ Array(g)
    return real.(delta)                                         # [n_par]
end

# ============================================================
# 3. メインループ
# ============================================================
function run_deterministic(model, ps, st, basis, H, n_modes;
                           n_epochs = 50000, lr = 1.5e-6,
                           lambda = 1e-4, epsilon = 1e-8,
                           chunk = 2048, log_every = 10)
    inputs_full = build_full_inputs(basis, n_modes)
    E_prev = Inf

    for epoch in 1:n_epochs
        p, E_loc, E, _ = exact_state(model, ps, st, inputs_full, H; chunk = chunk)
        Δ = deterministic_sr_update(model, ps, st, inputs_full, p, E_loc, E;
                                    chunk = chunk, lambda = lambda, epsilon = epsilon)

        # ★ 決定論的なので E は単調減少すべき。増えたら lr が大きすぎる
        if epoch % log_every == 0 || epoch == 1
            # 参考: Var(E_loc) も厳密に出る (収束の質の指標)
            V = real(sum(p .* abs2.(E_loc .- E)))
            flag = E > E_prev ? "  ← 上昇 (lr 過大)" : ""
            @printf("epoch %5d  E = %.10f  Var = %.3e  |Δ| = %.3e%s\n",
                    epoch, E, V, norm(Δ), flag)
        end
        E_prev = E

        ps_vec = ComponentArrays.getdata(ps)
        ps_vec .-= Float32(lr) .* CuArray(Float32.(Δ))
    end
    return ps
end

# ============================================================
# 検証手順 (実行前にこの順で確認すること)
# ------------------------------------------------------------
# V1. exact_state の E が、既に手で回した厳密評価 (-0.23976731) と
#     一致するか。ここがズレたら inputs の並び順 (FlattenLayer の
#     column-major) か H の配置順との対応がずれている。
# -> OK
# V2. 同じ ψ に対して、MC 版の (S, g) と本実装の (S, g) が
#     「サンプル数 → ∞」で近づくか。簡易版として、p(x) の代わりに
#     MC のサンプル頻度を入れると MC 版に一致するはず。
#     conj 規約 (規約B) の整合はこのテストでしか保証できない。
#
# V3. 最初の数エポックで E が単調減少するか。決定論的勾配なので
#     lr が適正なら必ず単調。振動したら lr を 1/3 に。
# -> OK 
# ============================================================

# ============================================================
# 正則化の変更点 (MC 版からの重要な差)
# ------------------------------------------------------------
# * 統計ノイズが存在しないので、ε の「ノイズゲート」としての役割は消える。
#   ε は純粋に数値的な条件数対策 → 1e-8 程度まで下げてよい。
#   (MC 版の ε = 1e-5〜1e-6 のままだと、尾部配置を担う極小 S_kk 方向を
#    不要に潰してしまい、まさに検証したい自由度が動かない)
#
# * 勾配ノルムクリッピングは無効化する。ノイズ由来のスパイクが無いので
#   発動する理由がなく、発動すれば単に歩幅を削るだけ。
#
# * diag(S) のゼロは激減する。全配置を重み付きで見るので、|l|=5 のセルも
#   微小重みで必ず現れる。残るゼロは b₂ (全体位相・規格化) の 2 個 =
#   物理的なゲージ自由度だけになるはず。ここは ε が処理する。
#
# * lr はアニーリングが有効 (決定論的なので下げれば単調に精度が上がる)。
#   例: 最後の 1000 エポックで lr を 1/10 に。
# ============================================================

# ============================================================
# メモリ見積もり
# ------------------------------------------------------------
#   S (ComplexF64):  n_par² × 16 バイト
#     hidden=128 (n_par=4610):  340 MB   ← 余裕
#     hidden=256 (n_par=9218):  1.36 GB  ← 3090 でも可、chunk と併せて注意
#   compute_O_bar の W1 ブロック: [n_h, n_in, chunk] complex
#     hidden=256, chunk=2048:   ~276 MB
#   inputs_full: 3 MB, ψ/E_loc 系: 各 0.4 MB (CPU)
# ============================================================