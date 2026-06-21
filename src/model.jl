module Model

using Lux
using Random
using NNlib # 活性化関数（tanh, geluなど）を使用するために必要

export build_momentum_nqs, initialize_model

"""
波数空間のスピン1ボゾン系向けNQSを構築する関数
"""
function build_momentum_nqs(k_max::Int; hidden_dim::Int=64)
    n_modes = 2 * k_max + 1
    input_features = n_modes * 3 # (波数モード数) × (スピン3成分)
    
    # Lux.Chain でネットワークを定義
    model = Chain(
        # 1. テンソルの平坦化
        # 入力 [n_modes, 3, n_walkers] を [input_features, n_walkers] に変換
        FlattenLayer(), 
        
        # 2. 全結合層（非局所的な波数間の相関を学習）
        # 活性化関数には、最適化が安定しやすい滑らかな関数（tanhなど）を採用
        Dense(input_features => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim, tanh),
        
        # 3. 出力層
        # 各ウォーカーに対して1つのスカラー値（対数振幅 logΨ）を出力
        Dense(hidden_dim => 2) 
    )
    
    return model
end

"""
モデルの初期化と、GPU(CUDA)への転送準備を行うヘルパー関数
"""
function initialize_model(model, rng::AbstractRNG)
    # Luxでは、パラメータ(ps)と状態(st)を明示的に初期化します
    ps, st = Lux.setup(rng, model)
    return ps, st
end

end # module Model