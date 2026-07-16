"""
manual_jacobian.jl — 対数微分行列 Ō の閉形式バッチ計算

【目的】
SR に必要な Ō_k(x) = conj(∂_k log ψ(x)) を、Zygote の逆伝播 (出力ごとに1回、
チャンクあたり数千回) の代わりに、行列演算数回で一括計算する。
測定: Zygote 版 ~3 s/chunk → 本実装は GEMM/broadcast のみで ~10 ms オーダー。

【対象アーキテクチャ】
    FlattenLayer → Dense(n_in → n_h, σ) → Dense(n_h → 2)
    出力: out[1] = log|ψ|, out[2] = arg ψ,  log ψ = out[1] + i·out[2]

【導出】
    z = W₁x + b₁,  h = σ(z),  out = W₂h + b₂
    ∂out_α/∂W₁[i,j] = W₂[α,i]·σ'(z_i)·x_j
    ∂out_α/∂b₁[i]   = W₂[α,i]·σ'(z_i)
    ∂out_α/∂W₂[α,j] = h_j          (他成分は0)
    ∂out_α/∂b₂[α]   = 1            (他成分は0)
    log ψ = out₁ + i·out₂ なので、複素微分は α=1 成分 + i·(α=2 成分)。

【規約B (既存 optimise.jl と同じ「共役済み」)】
    Ō = conj(∂ log ψ) を直接構築する:
        D̄ = (W₂[1,:] − i·W₂[2,:]) ⊙ σ'(z)      ← ここのマイナスが共役
    Ō_W₁[i,j] = D̄_i x_j,  Ō_b₁ = D̄,
    Ō_W₂[1,j] = h_j,  Ō_W₂[2,j] = −i·h_j,  Ō_b₂ = (1, −i)

【パラメータ順序 (ComponentVector と一致させること)】
    ps = [layer_2.weight (vec, column-major), layer_2.bias,
          layer_3.weight (vec, column-major), layer_3.bias]
    Julia の reshape/vec は column-major なので、[n_h, n_in, nb] を
    [n_h*n_in, nb] に reshape すれば vec(W₁) と同順になる。

【★ 要調整】
    - σ と σ' : model.jl の活性化関数に合わせる (下は tanh の例)
    - 層名 layer_2 / layer_3 : keys(ps) で確認
"""

using CUDA
using LinearAlgebra

# ★ 活性化関数とその微分 (model.jl に合わせて変更)
@inline act(z)  = tanh(z)
@inline dact_from_h(h) = 1 - h^2     # tanh: σ' = 1 − h² (h = σ(z) から計算できる)
# 例: σ = gelu なら dact は z から計算する形に変える

"""
    compute_O_bar(inputs, ps) -> (Ō, logψ)

inputs :: CuMatrix{Float32} [n_in, nb]   (Float32 化・平坦化済みの占有数)
ps     :: ComponentVector (GPU)

戻り値:
  Ō     :: CuMatrix{ComplexF32} [n_params, nb]  規約B (共役済み) の対数微分
  log_psi  :: CuVector{ComplexF32} [nb]            ついでに前向き評価も返す
          (eval_complex_network の呼び直しを省ける)
"""
function compute_O_bar(inputs::CuMatrix{Float32}, ps)
    W1 = ps.layer_2.weight        # [n_h, n_in]
    b1 = ps.layer_2.bias          # [n_h]
    W2 = ps.layer_3.weight        # [2, n_h]
    b2 = ps.layer_3.bias          # [2]

    n_h, n_in = size(W1)
    nb = size(inputs, 2)

    # ---- 前向き ----
    Z = W1 * inputs .+ b1                 # [n_h, nb]
    H = act.(Z)                           # [n_h, nb]
    out = W2 * H .+ b2                    # [2, nb]
    log_psi = ComplexF32.(out[1, :], out[2, :])   # [nb]

    # ---- 逆向き (閉形式) ----
    Sp = dact_from_h.(H)                  # σ'(z) [n_h, nb]
    # 規約B: 共役済みの「デルタ」  D̄ = (W₂[1,:] − i W₂[2,:]) ⊙ σ'
    w2c = ComplexF32.(W2[1, :], -W2[2, :])          # [n_h]  (虚部にマイナス = conj)
    D̄ = w2c .* ComplexF32.(Sp)                      # [n_h, nb] (列方向ブロードキャスト)

    # W₁ ブロック: Ō[i,j,s] = D̄[i,s] * x[j,s]  (サンプルごとの外積をブロードキャストで)
    O_W1 = reshape(D̄, n_h, 1, nb) .* reshape(ComplexF32.(inputs), 1, n_in, nb)
    O_W1 = reshape(O_W1, n_h * n_in, nb)            # vec(W₁) と同順 (column-major)

    # b₁ ブロック
    O_b1 = D̄                                        # [n_h, nb]

    # W₂ ブロック: Ō[(α,j), s] : α=1 → h_j,  α=2 → −i·h_j
    # vec(W₂) の順序は α が速い (column-major, W₂ は [2, n_h])
    Hc = ComplexF32.(H)
    O_W2 = similar(Hc, ComplexF32, 2 * n_h, nb)
    O_W2[1:2:end, :] .= Hc                           # α=1 行 (奇数行)
    O_W2[2:2:end, :] .= -im .* Hc                    # α=2 行 (偶数行)

    # b₂ ブロック: (1, −i) を全サンプルに
    O_b2 = CuMatrix{ComplexF32}(undef, 2, nb)
    O_b2[1, :] .= 1.0f0 + 0.0f0im
    O_b2[2, :] .= 0.0f0 - 1.0f0im

    Ō = vcat(O_W1, O_b1, O_W2, O_b2)                 # [n_params, nb]
    return Ō, log_psi
end

