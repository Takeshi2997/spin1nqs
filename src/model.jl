module Model

using CUDA
using Lux
using Zygote
using JLD2
using Functors
using Random
using NNlib 
using ComponentArrays
using Combinatorics

export build_momentum_nqs, save_nqs_model, load_nqs_model, initialize_model, 
eval_complex_network, eval_complex_network_real, eval_complex_network_imag, expand_params, 
JastrowIndex, to_gpu, npair, build_U, jastrow_O, jastrow_logpsi, NqsParametersWithJastrow, 
save_jastrow, load_jastrow

struct NqsParametersWithJastrow
    nqs_params::ComponentVector{Float32}   # NQSのパラメータ
    U::AbstractMatrix{Float32}     # ジャストローパラメータ
    N::Int                         # 系の粒子数
end

"""
波数空間のスピン1ボゾン系向けNQSを構築する関数
"""
function build_momentum_nqs(k_max::Int; hidden_dim::Int=32)
    n_modes = 2 * k_max + 1
    input_features = n_modes * 3 # (波数モード数) × (スピン3成分)
    
    # Lux.Chain でネットワークを定義
    model = Chain(
        # 1. テンソルの平坦化
        # 入力 [n_modes, 3, n_walkers] を [input_features, n_walkers] に変換
        FlattenLayer(), 

        # 2. 全結合層
        Dense(input_features => hidden_dim, tanh),
        ## Dense(input_features => hidden_dim0, relu),
        ## Dense(hidden_dim0 => hidden_dim, tanh),

        # 3. 出力層
        # 各ウォーカーに対して対数振幅 logΨの実部と虚部を出力
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

"""
与えられた入力に対して、複素数の波動関数を評価する関数
"""
function eval_complex_network(model, inputs, ps::ComponentVector, st)
    # 入力は [n_modes, 3, n_walkers] の形状を想定
    # Lux.apply は [2, n_walkers] の出力を返す（1行目: log|Ψ|の実部、2行目: log|Ψ|の虚部）
    outputs, _ = Lux.apply(model, inputs, ps, st)
    
    # 複素数の波動関数 Ψ を構築
    log_psi_real = outputs[1, :]
    log_psi_imag = outputs[2, :]

    return log_psi_real .+ im .* log_psi_imag
end

function eval_complex_network_real(model, inputs, ps::ComponentVector, st)
    return real.(eval_complex_network(model, inputs, ps, st))
end

function eval_complex_network_imag(model, inputs, ps::ComponentVector, st)
    return imag.(eval_complex_network(model, inputs, ps, st))
end

"""
3モードで学習した ps_old (ComponentVector) の重みを、
11モード用に新しく初期化した ps_new に移植する。
- layer_2: Dense(入力→hidden)。列をオフセット付きでコピー、新規列はゼロ
- layer_3: Dense(hidden→2)。そのままコピー
"""
function expand_params(ps_old::ComponentVector, k_max_old::Int, k_max_new::Int,
                       model_new, rng = Random.default_rng())
    n_old = 2 * k_max_old + 1        # 3
    n_new = 2 * k_max_new + 1        # 11
    offset = k_max_new - k_max_old # 4

    # 通常どおり初期化 (構造を得るため)
    ps_new, st_new = Lux.setup(rng, model_new)
    ps_new = ComponentVector{Float32}(ps_new)

    # --- layer_2: Dense(3*n → hidden) ---
    W_old = ps_old.layer_2.weight          # [hidden, 3*n_old] の view
    ps_new.layer_2.weight .= 0.0f0         # 新規列 (l=±2..±5) はゼロ
    for s in 1:3, m in 1:n_old
        row_old = (s - 1) * n_old + m
        row_new = (s - 1) * n_new + m + offset
        ps_new.layer_2.weight[row_new, :] .= W_old[row_old, :]
    end
    ps_new.layer_2.bias .= ps_old.layer_2.bias

    # --- layer_3: Dense(hidden → 2) はそのまま ---
    ps_new.layer_3.weight .= ps_old.layer_3.weight
    ps_new.layer_3.bias   .= ps_old.layer_3.bias

    return ps_new, st_new
end

struct JastrowIndex{V<:AbstractVector{Int}}
    n::Int
    lin::V              # 上三角 (i ≤ j) の列優先線形インデックス (j-1)n + i
end
 
JastrowIndex(n::Int) = JastrowIndex(n, [(j - 1) * n + i for j in 1:n for i in 1:j])
to_gpu(ji::JastrowIndex) = JastrowIndex(ji.n, CuArray(ji.lin))
npair(ji::JastrowIndex) = length(ji.lin)
 
# ------------------------------------------------------------
# θ → U (パラメータ更新ごとに1回。カーネル1回の散布)
# ------------------------------------------------------------
function build_U(J::AbstractVector{T}, ji::JastrowIndex) where {T}
    U = fill!(similar(J, ji.n, ji.n), zero(T))
    U[ji.lin] .= J
    return U
end
 
# 占有数 → スケール済み入力 [n, cols]
scaled_input(states::AbstractArray{Float32,3}, s::Real) =
    reshape(states, :, size(states, 3)) ./ Float32(s)

# ------------------------------------------------------------
# 評価: log ψ_J = xᵀ U x  (特徴行列なし)
# ------------------------------------------------------------
function jastrow_logpsi(states::AbstractArray, U::AbstractMatrix, s::Real) 
    X = states ./ Float32(s)
    vec(sum((U * X) .* X, dims = 1))
end
 
# ------------------------------------------------------------
# SR 用: O_J [npair, nb]  (サンプルした配置についてのみ呼ぶ)
# ------------------------------------------------------------
function jastrow_O(states::AbstractMatrix, ji::JastrowIndex, s::Real)
    X = states ./ Float32(s)
    n, nb = size(X)
    outer = reshape(X, n, 1, nb) .* reshape(X, 1, n, nb)       # outer[i,j,b] = x_i x_j
    return reshape(outer, n * n, nb)[ji.lin, :]                # 上三角の行だけ
end

"""
与えられた入力に対して、複素数の波動関数を評価する関数
"""
function eval_complex_network(model, inputs, nqs_params::NqsParametersWithJastrow, st)
    # 入力は [n_modes, 3, n_walkers] の形状を想定
    # Lux.apply は [2, n_walkers] の出力を返す（1行目: log|Ψ|の実部、2行目: log|Ψ|の虚部）
    outputs, _ = Lux.apply(model, inputs, nqs_params.nqs_params, st)
    
    # 複素数の波動関数 Ψ を構築
    log_psi_real = outputs[1, :]
    log_psi_imag = outputs[2, :]

    X = scaled_input(inputs, nqs_params.N / 3)
    log_psi_jastrow = jastrow_logpsi(X, nqs_params.U, nqs_params.N / 3)

    return log_psi_real .+ im .* log_psi_imag .+ log_psi_jastrow
end

function eval_complex_network_real(model, inputs, nqs_params::NqsParametersWithJastrow, st)
    return real.(eval_complex_network(model, inputs, nqs_params, st))
end

function eval_complex_network_imag(model, inputs, nqs_params::NqsParametersWithJastrow, st)
    return imag.(eval_complex_network(model, inputs, nqs_params, st))
end

"""
保存処理 学習完了後やチェックポイント
"""
function save_nqs_model(dirname, epoch, ps_gpu, st_gpu)
    # 1. GPU (CuArray) から CPU (標準のArray) へ変換
    ps_cpu = fmap(Array, ps_gpu)
    st_cpu = fmap(Array, st_gpu)
    
    # 2. JLD2でファイルに保存
    num_params = Lux.parameterlength(ps_cpu)
    filename = dirname * "/nqs_model_$(num_params)_epoch$(epoch).jld2"
    @save filename ps_cpu st_cpu
    println("モデルを $(filename) に保存しました。")
end

"""
読み込み処理 計算の再開やデータ解析時
"""
function load_nqs_model(filename)
    # 1. JLD2ファイルからCPUメモリへ読み込み
    @load filename ps_cpu st_cpu
    
    println("モデルを $(filename) から読み込みました。")
    return ps_cpu, st_cpu
end

"""
ジャストローパラメータ保存処理 学習完了後やチェックポイント
"""
function save_jastrow(dirname, epoch, J_gpu)
    # 1. GPU (CuArray) から CPU (標準のArray) へ変換
    J_cpu = fmap(Array, J_gpu)
    
    # 2. JLD2でファイルに保存
    num_params = length(J_cpu)
    filename = dirname * "/jastrow_$(num_params)_epoch$(epoch).jld2"
    @save filename J_cpu
    println("ジャストローパラメータを $(filename) に保存しました。")
end

"""
ジャストローパラメータ読み込み処理 計算の再開やデータ解析時
"""
function load_jastrow(filename)
    # 1. JLD2ファイルからCPUメモリへ読み込み
    @load filename J_cpu
    
    println("ジャストローパラメータを $(filename) から読み込みました。")
    return J_cpu
end

end # module Model