module Optimise

using Lux
using Zygote
using LinearAlgebra

export compute_SR_update

"""
SR法によるパラメータ更新ベクトルを計算する
"""
function compute_SR_update(model, ps_flat, st, states, E_loc, epsilon=1.0f-3)
    n_walkers = size(states, ndims(states))

    # 1. ヤコビアン O_k(x) の計算のためのラッパー関数
    function eval_log_psi_real(p)
        # log_psi の評価 (バッチ全体)
        out, _ = Lux.apply(model, states, p, st)
        # 複素NQSの場合は、ここで実部と虚部を合成した複素数ベクトルを返す
        return out[1, :]
    end

    function eval_log_psi_imag(p)
        # log_psi の評価 (バッチ全体)
        out, _ = Lux.apply(model, states, p, st)
        # 複素NQSの場合は、ここで実部と虚部を合成した複素数ベクトルを返す
        return out[1, :]
    end


    # Zygoteでヤコビアンを取得 O: [n_walkers, n_params]
    # ※ Zygote.jacobian は (出力次元, 入力次元) で返ります
    O_real = Zygote.jacobian(eval_log_psi_real, ps_flat)[1] 
    O_imag = Zygote.jacobian(eval_log_psi_imag, ps_flat)[1] 
    O_matrix = O_real .+ im .* O_imag # 複素数のヤコビアン行列
    
    # 計算の都合上、転置して [n_params, n_walkers] にする
    O = transpose(O_matrix)

    # 2. 中心化（平均を引いてゆらぎにする）
    O_mean = sum(O, dims=2) ./ n_walkers
    E_mean = sum(E_loc) ./ n_walkers

    O_centered = O .- O_mean
    E_centered = E_loc .- E_mean

    # 3. S行列とFベクトルの構築
    # O_centered * O_centered' はパラメータ次元の行列になる
    S = real.((conj.(O_centered) * transpose(O_centered)) ./ n_walkers)
    
    # E_centered は1次元配列なので、O_centeredとの積でベクトルになる
    F = real.((conj.(O_centered) * E_centered) ./ n_walkers)

    # 4. 正則化（対角成分を少し底上げして逆行列計算を安定化）
    # S = S + ε * I の処理
    S_reg = S + epsilon * I

    # 5. 連立方程式 S * Δp = F を解く ( \ 演算子を使用)
    # GPU上(CuArray)のままでLU分解やコレスキー分解が走ります
    delta_p = S_reg \ F

    return delta_p
end

end # module Optimise